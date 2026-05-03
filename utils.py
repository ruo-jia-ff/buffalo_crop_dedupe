import numpy as np
import cv2
from itertools import product as product
from math import ceil

import torch
import torch.nn.functional as F


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase="train"):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1] for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0] for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline.
    Args:
        dets: detections before nms
        thresh: nms threshold
    Return:
        keep: index after nms
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )
    return landms


def pad_image(image, h, w, size, padvalue):
    pad_image = image.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                    pad_w, cv2.BORDER_CONSTANT,
                                    value=padvalue)
    return pad_image


def resize_image(image, re_size, keep_ratio=True):
    """Resize image
    Args: 
        image: origin image
        re_size: resize scale
        keep_ratio: keep aspect ratio. Default is set to true.
    Returns:
        re_image: resized image
        resize_ratio: resize ratio
    """
    if not keep_ratio:
        re_image = cv2.resize(image, (re_size[0], re_size[1])).astype('float32')                                             
        return re_image, 0, 0 
    ratio = re_size[0] * 1.0 / re_size[1] 
    h, w = image.shape[0:2]
    if h * 1.0 / w <= ratio:
        resize_ratio = re_size[1] * 1.0 / w
        re_h, re_w = int(h * resize_ratio), re_size[1] 
    else:
        resize_ratio = re_size[0] * 1.0 / h
        re_h, re_w = re_size[0], int(w * resize_ratio)
    
    re_image = cv2.resize(image, (re_w, re_h)).astype('float32')                                              
    re_image = pad_image(re_image, re_h, re_w, re_size, (0.0, 0.0, 0.0))
    return re_image, resize_ratio


def preprocess(img_raw, input_size, device):
    """preprocess
    Args:
        img_raw: origin image 
    Returns: 
        img: resized image
        scale: resized image scale
        resize: resize ratio
    """
    img = np.float32(img_raw)
    # resize image
    img, resize = resize_image(img, input_size)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.numpy()
    scale = scale.to(device)
    return img, scale, resize





def postprocess(cfg, img, outputs, scale, resize, confidence_threshold, nms_threshold, device, input_size=None):
    """
    Post-process RetinaFace outputs to get final boxes and landmarks.

    Args:
        cfg: RetinaFace config
        img: preprocessed image
        outputs: list of outputs from ONNX model [loc, conf, landms]
        scale: torch.Tensor with [w, h, w, h] for scaling boxes
        resize: resize ratio
        confidence_threshold: minimum confidence to keep detection
        nms_threshold: NMS IoU threshold
        device: torch device
        input_size: tuple (H, W) for prior box generation
    Returns:
        dets: numpy array of [x1, y1, x2, y2, score, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
    """

    # Determine image size
    if input_size is None:
        im_height, im_width = img.shape[2], img.shape[3]
    else:
        im_height, im_width = input_size

    # Convert outputs to torch tensors
    loc = torch.from_numpy(outputs[0]).to(device)
    conf = torch.from_numpy(outputs[1]).to(device)
    landms = torch.from_numpy(outputs[2]).to(device)

    # Softmax on confidence
    conf = torch.nn.functional.softmax(conf, dim=-1)

    # Generate priors
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    print(priors)
    print("Priors Shape: ", priors.shape)

    # Decode boxes and landmarks
    boxes_decoded = decode(loc.squeeze(0), priors, cfg["variance"])
    landms_decoded = decode_landm(landms.squeeze(0), priors, cfg["variance"])

    # Scale boxes to original image
    boxes_decoded = boxes_decoded * scale / resize
    boxes_decoded = boxes_decoded.cpu().numpy()

    # Scale landmarks to original image
    scale_landms = torch.tensor([
        scale[0], scale[1],
        scale[2], scale[3],
        scale[0], scale[1],
        scale[2], scale[3],
        scale[0], scale[1]
    ], device=device, dtype=torch.float32).view(1, 10)  # shape [1,10]

    landms_decoded = landms_decoded * scale_landms / resize
    landms_decoded = landms_decoded.cpu().numpy()

    # Confidence scores
    scores = conf.squeeze(0).cpu().numpy()[:, 1]

    # Filter by confidence
    inds = np.where(scores > confidence_threshold)[0]
    boxes_decoded = boxes_decoded[inds]
    landms_decoded = landms_decoded[inds]
    scores = scores[inds]

    # Sort by scores
    order = scores.argsort()[::-1]
    boxes_decoded = boxes_decoded[order]
    landms_decoded = landms_decoded[order]
    scores = scores[order]

    # Combine boxes and scores for NMS
    dets = np.hstack((boxes_decoded, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms_decoded = landms_decoded[keep]

    # Concatenate landmarks to boxes
    dets = np.concatenate((dets, landms_decoded), axis=1)

    return dets


# def postprocess(cfg, img, outputs, scale, resize, confidence_threshold, nms_threshold, device, input_size):
#     """post_process
#     Args:
#         img: resized image
#         outputs: forward outputs
#         scale: resized image scale
#         resize: resize ratio
#         confidence_threshold: confidence threshold
#         nms_threshold: non-maximum suppression threshold
#     Returns: 
#         detetcion results
#     """
#     # _,  im_height, im_width, _= img.shape
#     if input_size is None:
#         # img is NCHW: (1,3,H,W)
#         im_height, im_width = img.shape[2], img.shape[3]
#     else:
#         im_height, im_width = input_size
#     loc = torch.from_numpy(outputs[0]).to(device)
#     conf = torch.from_numpy(outputs[1]).to(device)
#     landms = torch.from_numpy(outputs[2]).to(device)
#     # softmax
#     conf = F.softmax(conf, dim=-1)

#     priorbox = PriorBox(cfg, image_size=(im_height, im_width))
#     priors = priorbox.forward()
#     priors = priors.to(device)
#     prior_data = priors.data
#     boxes = decode(loc.squeeze(0), prior_data, cfg["variance"])
#     boxes = boxes * scale / resize 
#     boxes = boxes.cpu().numpy()
#     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
#     landms = decode_landm(landms.squeeze(0), prior_data, cfg["variance"])
#     scale1 = torch.Tensor(
#         [img.shape[2], img.shape[1], img.shape[2], img.shape[1], img.shape[2],
#          img.shape[1], img.shape[2], img.shape[1], img.shape[2], img.shape[1],]
#     )
#     scale1 = scale1.to(device)
#     landms = landms * scale1 / resize 
#     landms = landms.cpu().numpy()

#     # ignore low scores
#     inds = np.where(scores > confidence_threshold)[0]
#     boxes = boxes[inds]
#     landms = landms[inds]
#     scores = scores[inds]

#     # keep top-K before NMS
#     order = scores.argsort()[::-1]
#     boxes = boxes[order]
#     landms = landms[order]
#     scores = scores[order]

#     # do NMS
#     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#     keep = py_cpu_nms(dets, nms_threshold)
#     dets = dets[keep, :]
#     landms = landms[keep]
#     dets = np.concatenate((dets, landms), axis=1)
#     return dets