import cv2
import numpy as np
from scipy.special import hermite
from math import factorial

def rotate_image_in_memory(img, angle):
    # Takes angle input in degrees
    if angle == 0:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def mdghm_kernel(order, size, sigma):
    x = np.linspace(-1, 1, size)
    x_grid, y_grid = np.meshgrid(x, x)

    Hx = hermite(order)(x_grid / sigma)
    Hy = hermite(order)(y_grid / sigma)
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2)) * Hx * Hy

    norm_factor = np.sqrt(2**order * factorial(order) * np.pi * sigma**2)
    return kernel / norm_factor

def mdghm_response(img, order=3, size=9, sigma=0.5):
    kernel = mdghm_kernel(order, size, sigma)
    response = cv2.filter2D(img.astype(np.float32), -1, kernel)
    return response

def get_keypoints_mdghm(img, order=3, size=9, sigma=0.5, threshold=0.1, max_kps=500):
    response = mdghm_response(img, order, size, sigma)

    local_max = (response == cv2.dilate(response, np.ones((3, 3), np.uint8)))
    local_min = (response == cv2.erode(response, np.ones((3, 3), np.uint8)))
    extrema = np.logical_or(local_max, local_min)

    mask = np.abs(response) > threshold
    keypoint_mask = np.logical_and(extrema, mask)

    ys, xs = np.where(keypoint_mask)
    strengths = np.abs(response[ys, xs])
    sorted_idx = np.argsort(-strengths)[:max_kps]

    keypoints = [cv2.KeyPoint(float(xs[i]), float(ys[i]), 3) for i in sorted_idx]
    return keypoints

def calculate_descriptors(img, keypoints, patch_size=8):
    half = patch_size // 2
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)

    descriptors = []
    valid_kps = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        px, py = x + half, y + half
        patch = padded[py - half:py + half, px - half:px + half]
        if patch.shape == (patch_size, patch_size):
            descriptors.append(patch.flatten())
            valid_kps.append(kp)

    return np.array(descriptors, dtype=np.float32), valid_kps


def calculate_image_offset_mift(img_path1, img_path2, gsd=None, rotation_angle=0):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("One of the images couldn't be loaded.")

    if rotation_angle != 0:
        img2 = rotate_image_in_memory(img2, rotation_angle)

    kp1 = get_keypoints_mdghm(img1)
    kp2 = get_keypoints_mdghm(img2)

    des1, kp1 = calculate_descriptors(img1, kp1)
    des2, kp2 = calculate_descriptors(img2, kp2)

    if len(des1) < 4 or len(des2) < 4:
        raise RuntimeError("Not enough descriptors.")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    if len(matches) < 4:
        raise RuntimeError("Not enough good matches to compute transform.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is None:
        raise RuntimeError("Could not compute transform between images.")

    dx_px, dy_px = M[0, 2], M[1, 2]

    if gsd is not None:
        dx_m = dx_px * gsd
        dy_m = dy_px * gsd
        return {
            "pixel_offset": (dx_px, dy_px),
            "distance_offset_m": (dx_m, dy_m),
            "method": f"MIFT + Affine (matches: {len(matches)})"
        }
    else:
        return {
            "pixel_offset": (dx_px, dy_px),
            "method": f"MIFT + Affine (matches: {len(matches)})"
        }
