import cv2
import json
import numpy as np
import random
import torch
from PIL import Image

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 


_CLASSNAMES = ['robot', 'coke can', 'pepsi can', 'redbull can', '7up can', 'blue plastic bottle', 
               'apple', 'orange', 'sponge', 'bottom drawer', 'middle drawer', 'top drawer',
               'eggplant', 'spoon', 'carrot', 'plate', 'towel', 'yellow basket', 'green cube', 'yellow cube', 'goal_site']

_BACKGROUND_CLASSNAMES = ['floor', 'wall', 'ceiling']

_COLORS = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in _CLASSNAMES]

_NAME_TO_ALIAS = {
    '7up can': '7up pop can',
    'redbull can': 'redbull pop can',
    'coke can': 'coke pop can',
    'pepsi can': 'pepsi pop can',
    'robot': 'robot manipulator',
    'sponge': 'green sponge',
    'top drawer': 'dresser',
    'middle drawer': 'dresser',
    'bottom drawer': 'dresser',
}

_NAME_TO_ALIAS_SED = {
    'blue plastic bottle': 'mineral water bottle with label',
    'top drawer': 'dresser',
    'middle drawer': 'dresser',
    'bottom drawer': 'dresser',
}

_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_SAM2_CHECKPOINT = "pretrained/sam2.1_hiera_large.pt"
_GROUNDING_DINO_CHECKPOINT = "pretrained/grounding-dino-base"

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)

def _load_image_numpy_as_tensor(image, image_size):
    # img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    # cv2 resize
    video_height, video_width = image.shape[:2]
    image = cv2.resize(image, (image_size, image_size))
    if image.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        image = image / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {image.dtype}")
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image, video_height, video_width


def load_images_numpy(images, image_size, offload_video_to_cpu, compute_device):
    img_mean = torch.tensor(_IMAGE_MEAN, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(_IMAGE_STD, dtype=torch.float32)[:, None, None]
    num_frames = len(images)

    image_tensors = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, image in enumerate(images):
        image_tensors[n], video_height, video_width = _load_image_numpy_as_tensor(image, image_size)
    if not offload_video_to_cpu:
        image_tensors = image_tensors.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    image_tensors -= img_mean
    image_tensors /= img_std
    return image_tensors, video_height, video_width


def dilate_mask(mask, kernel_size):
    if mask is None:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def index_all(l, e):
    # find all indexs of e in l
    return [i for i, x in enumerate(l) if x == e]


def equal_all(l1, l2):
    if len(l1) != len(l2):
        return False
    return all(x == y for x, y in zip(l1, l2))


def mask_to_bbox(mask):
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])


def postprocess_mask(mask):
    if mask is None or not mask.any():
        return None
    
    # only keep the largest connected component
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    if num_labels > 1:
        # for each component, if area is smaller than 0.1 * largest_area, set it to 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 0.1 * largest_area:
                mask[labels == i] = 0

    # get counter of mask and fill poly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, contours, 1)
    return (mask > 0)


def visualize_multi_objects(image, masks, names, filename, points=None, boxes=None):
    for idx, (mask, name) in enumerate(zip(masks, names)):
        if mask is None or not mask.any():
            continue
        color = _COLORS[_CLASSNAMES.index(name)]
        mask_rgb = cv2.merge([mask.astype(np.uint8) * color[i] for i in range(3)])
        # draw mask on image
        image_with_mask = cv2.addWeighted(image, 0.5, mask_rgb, 0.5, 0)
        image[mask] = image_with_mask[mask]
        image = image.copy()
        
        y_center, x_center = np.where(mask)
        y_center = int(y_center.mean())
        x_center = int(x_center.mean())
        cv2.putText(image, name, (x_center, y_center), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        deeper_color = [c * 0.5 for c in color]
        if points is not None:
            # (2, 1) -> (2,)
            for point in points[idx]:
                cv2.circle(image, (int(point[0]), int(point[1])), 5, deeper_color, -1)
            
        if boxes is not None:
            xmin, ymin, xmax, ymax = boxes[idx]
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), deeper_color, 2)
        
    # save image to self.save_dir/visualize/name/filename
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


class GroundedSAMPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_CHECKPOINT)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(_GROUNDING_DINO_CHECKPOINT).to('cuda:0')
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image_pil = Image.fromarray(image.copy())
        
        # all_prompts = prompts + [name for name in _BACKGROUND_CLASSNAMES if name not in prompts]
        all_prompts = prompts
        text = '. '.join([_NAME_TO_ALIAS.get(p, p) for p in all_prompts]) + '.'
        
        inputs = self.processor(image_pil, text=text, return_tensors="pt").to('cuda:0')
        outputs = self.grounding_model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        input_boxes = result['boxes'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        labels = []

        # avoid one alias to multi name mapping
        alias_to_name = {v: k for k, v in _NAME_TO_ALIAS.items() if k in prompts}

        for l in result['labels']:
            # label will remove word "can"
            if l.endswith('pop'):
                l += ' can'
            labels.append(alias_to_name.get(l, l))

        self.image_predictor.set_image(image.copy())
        masks, mask_scores, mask_logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        output_masks = []
        for name in prompts:
            if name in labels:
                indexs = index_all(labels, name)
                selected_scores = scores[indexs]
                argmax_scores = selected_scores.argmax()
                output_masks.append(masks[indexs[argmax_scores]] > 0)
            else:
                output_masks.append(None)

        return [postprocess_mask(mask) for mask in output_masks]


class SEDPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
        from .SED.demo.predictor import VisualizationDemo as SEDDemo
        from .SED.sed import add_sed_config

        with open('segmentation_models/SED/datasets/simpler.json', 'w') as f:
            self.classnames = [_NAME_TO_ALIAS_SED.get(name, name) for name in _CLASSNAMES] + _BACKGROUND_CLASSNAMES
            self.classnames = list(set(self.classnames))
            json.dump(self.classnames, f)

        def setup_cfg():
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_sed_config(cfg)
            cfg.merge_from_file('segmentation_models/SED/configs/convnextL_768.yaml')
            cfg.merge_from_list(['MODEL.WEIGHTS', 'segmentation_models/SED/sed.pth'])
            cfg.freeze()
            return cfg
        cfg = setup_cfg()
        self.model = SEDDemo(cfg)

        sam2_checkpoint = "pretrained/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sem_seg = self.model.predictor(image)['sem_seg'].argmax(dim=0)
        
        masks = []
        for prompt in prompts:
            idx = self.classnames.index(_NAME_TO_ALIAS_SED.get(prompt, prompt))
            mask = (sem_seg == idx).detach().cpu().numpy()
            if mask.any():
                masks.append(postprocess_mask(mask))
            else:
                masks.append(None)
        
        self.image_predictor.set_image(image)
        # if mask is None, set box [0, 0, 1, 1]
        input_boxes = np.array([mask_to_bbox(mask) if mask is not None 
                                else np.array([0.0, 0.0, 1.0, 1.0]) for mask in masks])
        refined_masks, mask_scores, mask_logits = self.image_predictor.predict(point_coords=None, point_labels=None,
                                                                               box=input_boxes, multimask_output=False)
        if refined_masks.ndim == 4:
            refined_masks = refined_masks.squeeze(1)
        
        # add None to refined_masks
        refined_masks = [(refined_masks[i] > 0) if mask is not None else None for i, mask in enumerate(masks)]
        return [postprocess_mask(mask) for mask in refined_masks]


class VisualPromptPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.points = None
        self.boxes = None
    
    def set_points(self, points):
        # points is a list contains None
        self.points = np.array([point for point in points if point is not None])
        self.mask_index = [point is not None for point in points]
        
    def set_boxes(self, boxes):
        self.boxes = np.array([box for box in boxes if box is not None])
        self.mask_index = [box is not None for box in boxes]
    
    def predict(self, image, prompts):
        # only support one of points or boxes
        assert self.points is not None or self.boxes is not None, 'points or boxes must be provided!'
        assert self.points is None or self.boxes is None, 'only one of points or boxes can be provided!'
        
        self.image_predictor.set_image(image.copy())
        if self.points is not None:
            point_labels = np.ones(self.points.shape[:-1], dtype=int)
            masks, _, _ = self.image_predictor.predict(point_coords=self.points, point_labels=point_labels, 
                                                       box=None, multimask_output=False)
        elif self.boxes is not None:
            masks, _, _ = self.image_predictor.predict(point_coords=None, point_labels=None, 
                                                       box=self.boxes, multimask_output=False)
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        out_masks = []
        count = 0
        for mask_index in self.mask_index:
            if mask_index:
                out_masks.append(masks[count])
                count += 1
            else:
                out_masks.append(None)
        return [postprocess_mask(mask) for mask in masks]


class TrackingPredictor:
    def __init__(self, predictor):
        from .grounded_sam_2.sam2.build_sam import build_sam2_video_predictor

        self.predictor = predictor
        self.video_predictor = build_sam2_video_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)

        self.images = []
        self.masks_record = []
        self.has_masks_record = []
        self.inference_state = None
        self.start_tracking = False

        self.init_masks = None
    
    def predict(self, image, prompts):
        if not self.start_tracking:
            # if not start tracking, init or detect masks
            masks = self.predictor.predict(image, prompts)
            
            if all(mask is None for mask in masks):
                return masks
            
            # if there is at least one mask, start tracking
            self.reset_tracking_state_and_masks(image, masks, prompts)
            self.start_tracking = True
            return masks
        
        masks = self.tracking_next(image, prompts)
        # update tracking state
        self.reset_tracking_state_and_masks(image, masks, prompts)
        return masks
    
    def reset_tracking_state_and_masks(self, image, masks, prompts):
        self.images.append(image)
        self.masks_record.append(masks)
        self.has_masks_record.append([mask is not None for mask in masks])
        
        has_masks_init = self.has_masks_record[0]
        selected_index = [i for i, has_masks in enumerate(self.has_masks_record) if equal_all(has_masks, has_masks_init)]
        selected_images = [self.images[i] for i in selected_index]
        selected_masks = [self.masks_record[i] for i in selected_index]
        self.inference_state = self.video_predictor.init_state_from_images([selected_images[-1]])

        for mask, prompt in zip(selected_masks[-1], prompts):
            obj_id = _CLASSNAMES.index(prompt) + 1
            if mask is not None:
                self.video_predictor.add_new_mask(self.inference_state, 0, obj_id, mask)
    
    def tracking_next(self, image, prompts):
        images, _, _ = load_images_numpy([image], self.video_predictor.image_size, False, self.video_predictor.device)
        self.inference_state['images'] = torch.cat([self.inference_state['images'], images], dim=0)
        self.inference_state['num_frames'] = len(self.inference_state['images'])

        for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
            if frame_idx != self.inference_state['num_frames'] - 1:
                continue
            name2mask = dict()
            for idx, obj_id in enumerate(obj_ids):
                classname = _CLASSNAMES[obj_id - 1]
                mask = (mask_logits[idx].cpu().numpy() > 0).squeeze(0)
                if not mask.any():
                    mask = None
                name2mask[classname] = mask
        
        masks = [name2mask.get(p, None) for p in prompts]
        return masks
    
    def reset(self):
        self.images = []
        self.masks_record = []
        self.has_masks_record = []
        self.inference_state = None
        self.start_tracking = False


def build_predictor(predictor_name):
    if predictor_name == 'grounded_sam':
        return GroundedSAMPredictor()
    if predictor_name == 'grounded_sam_tracking':
        return TrackingPredictor(GroundedSAMPredictor())
    if predictor_name == 'sed':
        return SEDPredictor()
    if predictor_name == 'sed_tracking':
        return TrackingPredictor(SEDPredictor())
    if predictor_name == 'point_tracking':
        return TrackingPredictor(VisualPromptPredictor())
    if predictor_name == 'box_tracking':
        return TrackingPredictor(VisualPromptPredictor())
    raise ValueError(f'predictor_name {predictor_name} is not supported')


def predict_masks_with_predictor(image, prompts, predictor):
    masks = predictor.predict(image, prompts)
    visualize_multi_objects(image, masks, prompts, 'test.jpg', boxes=predictor.predictor.boxes)
    return [dilate_mask(mask, 5) for mask in masks]
