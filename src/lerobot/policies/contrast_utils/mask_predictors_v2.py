import cv2
import json
import numpy as np
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from .utils import *


_CLASSNAMES = ['robot', 'blue cup', 'green cup', 'red cup', 'yellow cup']

_BACKGROUND_CLASSNAMES = ['floor', 'wall', 'ceiling']

_NAME_TO_ALIAS_GDINO = {
    'robot': 'robot manipulator',
}
_NAME_TO_ALIAS_SED = {
    'robot': 'robot manipulator',
}

_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_SAM2_CHECKPOINT = "pretrained/sam2.1_hiera_large.pt"
_GROUNDING_DINO_CHECKPOINT = "pretrained/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def my_print(*args):
    # get gpu id
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    if gpu_id == '0':
        print(*args)


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


class GroundedSAMPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.device = 'cuda:0'
        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_CHECKPOINT)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(_GROUNDING_DINO_CHECKPOINT).to(self.device)
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image_pil = Image.fromarray(image.copy())
        
        # all_prompts = prompts + [name for name in _BACKGROUND_CLASSNAMES if name not in prompts]
        all_prompts = prompts
        text = '. '.join([_NAME_TO_ALIAS_GDINO.get(p, p) for p in all_prompts]) + '.'
        
        inputs = self.processor(image_pil, text=text, return_tensors="pt").to(self.device)
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
        alias_to_name = {v: k for k, v in _NAME_TO_ALIAS_GDINO.items() if k in prompts}

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

        with open('contrast_utils/SED/datasets/simpler.json', 'w') as f:
            self.classnames = [_NAME_TO_ALIAS_SED.get(name, name) for name in _CLASSNAMES] + _BACKGROUND_CLASSNAMES
            self.classnames = list(set(self.classnames))
            json.dump(self.classnames, f)

        def setup_cfg():
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_sed_config(cfg)
            cfg.merge_from_file('contrast_utils/SED/configs/convnextL_768.yaml')
            cfg.merge_from_list(['MODEL.WEIGHTS', 'pretrained/sed.pth'])
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
        
        return [postprocess_mask(mask) for mask in out_masks]


# class TrackingPredictor:
#     def __init__(self, predictor):
#         from .grounded_sam_2.sam2.build_sam import build_sam2_video_predictor

#         self.predictor = predictor
#         self.video_predictor = build_sam2_video_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)

#         self.images = []
#         self.masks_record = []
#         self.has_masks_record = []
#         self.inference_state = None
#         self.start_tracking = False

#         self.init_masks = None
    
#     def predict(self, image, prompts):
#         if not self.start_tracking:
#             # if not start tracking, init or detect masks
#             masks = self.predictor.predict(image, prompts)
            
#             if all(mask is None for mask in masks):
#                 return masks
            
#             # if there is at least one mask, start tracking
#             self.reset_tracking_state_and_masks(image, masks, prompts)
#             self.start_tracking = True
#             return masks
        
#         masks = self.tracking_next(image, prompts)
#         # update tracking state
#         self.reset_tracking_state_and_masks(image, masks, prompts)
#         return masks
    
#     def reset_tracking_state_and_masks(self, image, masks, prompts):
#         self.images.append(image)
#         self.masks_record.append(masks)
#         self.has_masks_record.append([mask is not None for mask in masks])
        
#         has_masks_init = self.has_masks_record[0]
#         selected_indexs = [i for i, has_masks in enumerate(self.has_masks_record) if equal_all(has_masks, has_masks_init)]
#         selected_indexs = self.uniform_select(selected_indexs, 3, 4)
#         selected_images = [self.images[i] for i in selected_indexs]
#         selected_masks_list = [self.masks_record[i] for i in selected_indexs]
#         self.inference_state = self.video_predictor.init_state_from_images(selected_images)

#         for frame_idx, masks in enumerate(selected_masks_list):
#             for mask, prompt in zip(masks, prompts):
#                 obj_id = _CLASSNAMES.index(prompt) + 1
#                 if mask is not None:
#                     self.video_predictor.add_new_mask(self.inference_state, frame_idx, obj_id, mask)
    
#     def uniform_select(self, indexs, max_num, step):
#         num = len(indexs)
#         if num <= max_num:
#             return indexs
#         if num <= max_num * step:
#             step = num // max_num
#         return indexs[::step][:max_num]
    
#     def tracking_next(self, image, prompts):
#         images, _, _ = load_images_numpy([image], self.video_predictor.image_size, False, self.video_predictor.device)
#         self.inference_state['images'] = torch.cat([self.inference_state['images'], images], dim=0)
#         self.inference_state['num_frames'] = len(self.inference_state['images'])

#         for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
#             if frame_idx != self.inference_state['num_frames'] - 1:
#                 continue
#             name2mask = dict()
#             for idx, obj_id in enumerate(obj_ids):
#                 classname = _CLASSNAMES[obj_id - 1]
#                 mask = (mask_logits[idx].cpu().numpy() > 0).squeeze(0)
#                 if not mask.any():
#                     mask = None
#                 name2mask[classname] = mask
        
#         masks = [name2mask.get(p, None) for p in prompts]
#         return masks
    
#     def reset(self):
#         self.images = []
#         self.masks_record = []
#         self.has_masks_record = []
#         self.inference_state = None
#         self.start_tracking = False


class TrackingPredictorV2:
    def __init__(self, predictor):
        from .grounded_sam_2.sam2.build_sam import build_sam2_camera_predictor
        self.predictor = predictor
        self.video_predictor = build_sam2_camera_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT, device='cuda')
        self.start_tracking = False
        self.objects = []
        
    def predict(self, image, prompts):
        for prompt in prompts:
            if prompt not in self.objects:
                self.objects.append(prompt)
        
        if not self.start_tracking:
            masks = self.predictor.predict(image, prompts)
            if any(mask is None for mask in masks):
                return masks
            
            self.video_predictor.load_first_frame(image)
            for mask, prompt in zip(masks, prompts):
                obj_id = self.objects.index(prompt) + 1
                if mask is not None:
                    self.video_predictor.add_new_mask(0, obj_id, mask)
                    
            self.start_tracking = True
            return masks
        
        obj_ids, mask_logits = self.video_predictor.track(image)
        name2mask = dict()
        for idx, obj_id in enumerate(obj_ids):
            classname = self.objects[obj_id - 1]
            mask = (mask_logits[idx].cpu().numpy() > 0).squeeze(0)
            if not mask.any():
                mask = None
            name2mask[classname] = mask
        
        masks = [name2mask.get(p, None) for p in prompts]
        return masks
    
    def reset(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2_camera_predictor
        self.video_predictor = build_sam2_camera_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.start_tracking = False
        self.objects = []


def build_predictor(predictor_name):
    if predictor_name == 'grounded_sam':
        return GroundedSAMPredictor()
    if predictor_name == 'grounded_sam_tracking':
        return TrackingPredictorV2(GroundedSAMPredictor())
    if predictor_name == 'sed':
        return SEDPredictor()
    if predictor_name == 'sed_tracking':
        return TrackingPredictorV2(SEDPredictor())
    if predictor_name == 'point_tracking':
        return TrackingPredictorV2(VisualPromptPredictor())
    if predictor_name == 'box_tracking':
        return TrackingPredictorV2(VisualPromptPredictor())
    raise ValueError(f'predictor_name {predictor_name} is not supported')


def predict_masks_with_predictor(image, prompts, predictor):
    masks = predictor.predict(image, prompts)
    # visualize_multi_objects(image, masks, prompts, 'test.jpg')
    return masks
