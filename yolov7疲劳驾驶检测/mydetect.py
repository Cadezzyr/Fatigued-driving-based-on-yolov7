import numpy as np
import torch
import cv2

from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# def detect(model, img, stride, imgsz, device, conf_thres=0.25, iou_thres=0.45):
#     img = letterbox(img, imgsz, stride=stride)[0]
#     img = img.transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device)
#     img = img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     pred = model(img)[0]
#     pred = non_max_suppression(pred, conf_thres, iou_thres)
#
#     boxes = []
#     for i, det in enumerate(pred):  # detections per image
#         if len(det):
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
#
#             for *xyxy, conf, cls in reversed(det):
#                 box = [int(cls.item()), *xyxy]
#                 boxes.append(box)
#     return boxes

# class LoadImagesNDArray:  # for numpy.ndarray type images
#     def __init__(self, img, img_size=640):
#         self.img = img
#         self.img_size = img_size
#         self.nf = 1  # number of files
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         img0 = self.img
#         img = letterbox(img0, new_shape=self.img_size)[0]
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#
#         self.count += 1
#         return "numpy_img", img, img0, None
#
#     def __len__(self):
#         return self.nf
# #
# def detect(model, frame, imgsz, params):
#     bboxes = []  # This list will store the bounding box results
#     save_img = not params.get("nosave", False)
#     img = frame.copy()
#     dataset = LoadImagesNDArray(img, img_size=imgsz)
#     # Initialize
#     device = select_device(params['device'])
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#
#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # Inference
#         pred = model(img, augment=params['augment'])[0]
#
#         # Apply NMS
#         pred = non_max_suppression(pred, params['conf_thres'], params['iou_thres'], classes=params['classes'],
#                                    agnostic=params['agnostic_nms'])
#
#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
#
#                 # Save results with class information
#                 for *xyxy, conf, cls in reversed(det):
#                     bbox_with_class = (
#                     cls.item(), xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item())
#                     bboxes.append(bbox_with_class)
#
#         return bboxes


def TracedModel(model, imgsz=640):
    print(" Convert model to Traced-model... ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stride = int(model.stride.max())
    imgsz = torch.tensor([imgsz, imgsz]).to(device)

    model.eval()
    example_input = torch.randn(1, 3, imgsz[0], imgsz[1]).to(device)
    traced_model = torch.jit.trace(model, example_input)
    print(" model is traced! \n")

    return traced_model


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img, img_size):
    img0 = img.copy()
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def postprocess_detections(pred, img_shape, im0_shape):
    bboxes = []
    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], im0_shape).round()
            # Save results with class information
            for *xyxy, conf, cls in reversed(det):
                bbox_with_class = (
                cls.item(), xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item())
                bboxes.append(bbox_with_class)
    return bboxes


# def detect(model, frame, imgsz):
#     model.eval()
#     with torch.no_grad():
#         img = preprocess_image(frame, imgsz)
#         pred = model(img, augment=False)[0]
#         bboxes = postprocess_detections(pred, img.shape, frame.shape)
#     return bboxes

def detect(model, frame, imgsz, stride=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    model.eval()
    with torch.no_grad():
        img = preprocess_image(frame, imgsz)
        img = img.to(device)
        pred = model(img, augment=False)[0]
        bboxes = postprocess_detections(pred, img.shape, frame.shape)
    return bboxes
