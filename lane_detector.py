from camera_geometry import CameraGeometry
import numpy as np
import cv2
import torch
import os
import sys

# Add detector library paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir + '/detectors/CLRerNet/')
sys.path.append(parent_dir)

from detectors.CLRerNet.configs.clrernet.culane.dataset_culane_clrernet import crop_bbox
from detectors.CLRerNet.libs.api.inference import *
from detectors.CLRerNet.libs.utils.visualizer import *
from mmdet.apis import init_detector

# Recreate the inference api here for an img array input to avoid modifying the detector repo
def inference_one_image_array(model, img):
    """Inference on an image with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img (np.array): Image array.
    Returns:
        img (np.ndarray): Image data with shape (width, height, channel).
        preds (List[np.ndarray]): Detected lanes.
    """
    ori_shape = img.shape
    # The data pipelines crop the input image in half (to focus on the road).
    # Detection is heavily impaired if the img is not resized to the original culane size
    # (the crop will be in a different place or not even possible)
    culane = cv2.resize(img, (crop_bbox[2], crop_bbox[3]))
    data = dict(
        filename=None,
        sub_img_name=None,
        img=culane,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=culane.shape,
        ori_shape=ori_shape,
    )
    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False
    device = next(model.parameters()).device  # model device
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    lanes = results[0]['result']['lanes']
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])
    return img, preds

# Recreate the visualize api here to color the lanes in an arbitrary way to avoid modifying the detector repo
def visualize_lanes_colors(
    src,
    preds,
    colors,
    annos=list(),
    concat_src=False,
    save_path=None,
):
    """
    visualize lane markers from prediction results and ground-truth labels
    Args:
        src (np.ndarray): Source image.
        preds (List[np.ndarray]): Lane predictions.
        colors (List[tuple]): Lane annotations.
        annos (List[np.ndarray]): Lane annotations.
        concat_src (bool): Concatenate the original and overlaid images vertically.
        save_path (str): The output image file path.
    Returns:
        dst (np.ndarray): Output image.
    """
    dst = copy.deepcopy(src)
    for anno in annos:
        dst = draw_lane(anno, dst, dst.shape, width=4, color=GT_COLOR)
    for pred, color in zip(preds, colors):
        dst = draw_lane(pred, dst, dst.shape, width=4, color=color)
    if concat_src:
        dst = np.concatenate((src, dst), axis=0)
    if save_path:
        cv2.imwrite(save_path, dst)
    return dst

# Lane detector used https://github.com/hirotomusiker/CLRerNet
# https://github.com/thomasfermi/Algorithms-for-Automated-Driving
class LaneDetector():
    def __init__(self, config, checkpoint, device, cam_geom=CameraGeometry()):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.model = init_detector(config, checkpoint, device)
        
    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image   

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def detect(self, img_array):
        src, preds = inference_one_image_array(self.model, img_array)
        return src, preds
       
    def fit_poly(self, pred):

        mask_idx = np.clip(pred.astype(np.int32), [0,0], [self.cg.image_width_pix-1, self.cg.image_height_pix-1]).dot(np.array([1, self.cg.image_width_pix]))
        mask = np.full(self.cg.image_height_pix*self.cg.image_width_pix, False)
        mask[mask_idx]=True
        
        mask_grid = mask[self.cut_v*self.cg.image_width_pix:]
        if mask_grid.any():
            coeffs = np.polyfit(self.grid[:,0][mask_grid], self.grid[:,1][mask_grid], deg=3)
        else:
            coeffs = np.array([0.,0.,0.,0.])
        return np.poly1d(coeffs)

    def __call__(self, image):
        if isinstance(image, str):
            image = self.read_imagefile_to_array(image)
        fits, preds = self.get_fit_and_probs(image)
        return fits, preds

    def get_fit_and_probs(self, img):
        _, preds = self.detect(img)
        fits = [self.fit_poly(pred) for pred in preds]
        return fits, preds
    
    def draw_lanes(self, frame, preds):
        return visualize_lanes(src=frame, preds=preds)
    
    def draw_lanes_colors(self, frame, preds, colors):
        return visualize_lanes_colors(src=frame, preds=preds, colors=colors)
    