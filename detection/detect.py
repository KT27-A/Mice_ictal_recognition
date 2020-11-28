import torch
import numpy as np
from detection.utils.general import non_max_suppression
from detection.utils.torch_utils import select_device
from detection.utils.datasets import letterbox


def detect(img, model):
    # assume img is raw image from cv2 with shape (height, widht, 3) and BGR
    # Then we will need to transpose it ot (channels, height, widht) and RGB
    origin_height, origin_width, _ = img.shape
    img = letterbox(img)
    # img = letterbox(img)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # Initialize
    device = select_device()
    # half precision only supported on CUDA
    # device = 'gpu'
    half = device != 'cpu'

    # img = img -  [144.7748, 107.7354, 99.4750]
    # img[0] = img[0] - 114.7748
    img[0] = img[0] - 114.7748
    img[1] = img[1] - 107.7354
    img[2] = img[2] - 99.4750
    # invert RGB to tensor
    img = torch.from_numpy(img).to(device)
    # img = torch.from_numpy(img)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.01, 0.5, classes=0, agnostic=True)
    # Process detections
    # only one img
    det = pred[0]
    if det is not None and len(det):
        det = det.cpu().numpy()
        det = det[det.argmax(axis=0)[-2]]
        # Write results
        x_center = (det[0] + det[2]) / 2
        y_center = (det[1] + det[3]) / 2
        # convert it to original size position
        x_center = round(x_center/640 * origin_width)
        y_center = round(y_center/384 * origin_height)
        return int(x_center), int(y_center)
