# 开发时间：2023/07/03 14:13
import cv2
import torch
import numpy as np
from skimage.transform import warp

from model import FAN
from lib.utils.utils import *
import matplotlib.pyplot as plt
import os
import dlib
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#1. initialize model and weights
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = 'best_checkpoint.pth.tar'# checkpoint path
model = FAN(3,68)
state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

#2. load image and perform dlib face detection and dlib face alignment
predictons = []

image = cv2.imread('obama.jpg')[:,:,::-1]
detector = dlib.get_frontal_face_detector()
# download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
dets = detector(image, 1)


def get_canonical_shape(keypoints, res):
    dst = np.array([[0, 20], [10, 20], [20, 20]])
    L = keypoints.shape[0]
    if L == 68:
        l, r = 36, 45
    elif L == 98:
        l, r = 60, 72
    else:
        raise ValueError("")
    src = np.array([keypoints[l], keypoints[l] * 0.5 + keypoints[r] * 0.5, keypoints[r]])
    d, z, tform = procrustes(dst, src)
    keypoints = np.dot(keypoints, tform['rotation']) * tform['scale'] + tform['translation']
    gtbox = get_gtbox(keypoints)
    xmin, ymin, xmax, ymax = gtbox
    keypoints -= [xmin, ymin]
    keypoints *= [res / (xmax - xmin), res / (ymax - ymin)]

    return keypoints
def warp(image, src, dst, res, keypoints=None):
    d, Z, meta = procrustes(dst, src)
    M = np.zeros([2, 3], dtype=np.float32)
    M[:2, :2] = meta['rotation'].T * meta['scale']
    M[:, 2] = meta['translation']
    img = cv2.warpAffine(image, M, (res, res))
    if keypoints is not None:
        keypoints = np.dot(keypoints, meta['rotation']) * meta['scale'] + meta['translation']
    return img, keypoints, meta

def crop_from_box(image, box, res, keypoints=None):
    xmin, ymin, xmax, ymax = box
    src = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    dst = np.array([[0, 0], [0, res - 1], [res - 1, 0], [res - 1, res - 1]])

    return warp(image, src, dst, res, keypoints)

def transform_keypoints(kps, tform, inverse=False):
    if inverse:
        new_kps = np.dot(kps - tform['translation'], np.linalg.inv(tform['rotation'] * tform['scale']))
    else:
        new_kps = np.dot(kps, tform['rotation']) * tform['scale'] + tform['translation']

    return new_kps

def show_preds(image, preds):
    plt.figure()
    plt.imshow(image)
    for pred in preds:
        plt.scatter(pred[:, 0], pred[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
    plt.ion()  # 开启交互模式

for idx, det in enumerate(dets):
    shape = landmark_predictor(image, det)
    dlib_shape = []
    for i in range(68):
        dlib_shape.append([shape.part(i).x, shape.part(i).y])
    dlib_shape = np.array(dlib_shape)
    can_shape = get_canonical_shape(dlib_shape, 112)
    can_shape += [8, 8]
    img, _, meta = warp(image, dlib_shape, can_shape, 128)
    # convert to tensor
    img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255)
    img.unsqueeze_(0)
    img = img.to(device)
    with torch.no_grad():
        out = model(img)
        out = get_preds(out)
        if out.is_cuda:
            out = out.cpu()
        pred = out.squeeze(0).numpy()
    pred = transform_keypoints(pred, meta, inverse=True)
    predictons.append(pred)

# 3. plot landmarks
show_preds(image, predictons)