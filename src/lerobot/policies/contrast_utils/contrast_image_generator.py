import numpy as np

from .instruction_templates import get_objects_from_instruction


class ContrastImageGenerator:
    def __init__(self, 
                 by="grounded_sam_tracking",
                 inpaint_mode="lama",
                 color="auto",
                 sigma=5):
        from .mask_predictors_v2 import build_predictor
        from .inpainters import build_inpainter
        
        self.by = by
        self.color = color
        self.sigma = sigma
        
        self.mask_objects = None
        self.keep_objects = None
        self.task_description = None
        self.predictor = build_predictor(self.by)
        self.inpainter = build_inpainter(inpaint_mode)
    
    def generate(self, image, task_description):
        if task_description != self.task_description:
            self.reset_mask_and_keep_object_names(task_description)
            self.task_description = task_description
        
        mask, excluded_mask = self.get_mask_by_predictor(image, reverse_mask=False)
        image = self.inpainter.inpaint(image, mask, excluded_mask)
        return image
    
    def reset(self, points=None, boxes=None):
        self.task_description = None

        if self.by == "point_tracking":
            if points is None:
                raise ValueError("Points must be provided for point tracking.")
            self.predictor.predictor.set_points(points)
        
        if self.by == "box_tracking":
            if boxes is None:
                raise ValueError("Boxes must be provided for box tracking.")
            self.predictor.predictor.set_boxes(boxes)
        
        self.predictor.reset()
        
    def reset_mask_and_keep_object_names(self, task_description):
        self.mask_objects = get_objects_from_instruction(task_description)
        self.keep_objects = ["robot"]
    
    def get_mask_by_predictor(self, image, reverse_mask=False):
        from .mask_predictors_v2 import predict_masks_with_predictor
        
        objs = self.mask_objects + self.keep_objects
        masks = predict_masks_with_predictor(image, objs, self.predictor)
        mask_obj_masks, keep_obj_masks = masks[:len(self.mask_objects)], masks[len(self.mask_objects):len(self.mask_objects) + len(self.keep_objects)]
        robot_mask = masks[objs.index('robot')] if 'robot' in objs else None
        mask = self._add_reserve_keep_mask(image.shape[:2], mask_obj_masks, reverse_mask, keep_obj_masks)
        return mask, robot_mask

    def _add_reserve_keep_mask(self, shape, masks, reverse_mask, keep_masks):
        mask = np.zeros(shape, dtype=bool)
        for obj_mask in masks:
            if obj_mask is not None:
                mask |= obj_mask

        if reverse_mask:
            mask = ~mask

        for obj_mask in keep_masks:
            if obj_mask is not None:
                mask[obj_mask] = False

        return mask
