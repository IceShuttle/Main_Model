import cv2
import numpy as np
import time
from pyproj import Geod

# Earth's radius in meters (approximate)
EARTH_RADIUS = 6378137  
geod = Geod(ellps="WGS84")


def rotate_image_in_memory(img, angle):
    """Rotate an image by a given angle without saving to disk."""
    if angle == 0:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated


def compute_orb_affine_offset(img1, img2):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return None, None, len(matches)

    good_matches = matches[:50]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return None, None, len(matches)

    dx, dy = M[0, 2], M[1, 2]
    return dx, dy, len(matches)


def smart_offset_estimation(img1_path, img2_path,prev_lat,prev_lon, rotation_angle=0):

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("One or both images could not be loaded.")

    # Rotate second image in memory if angle ≠ 0
    img2 = rotate_image_in_memory(img2, rotation_angle)

    dx, dy, num_matches = compute_orb_affine_offset(img1, img2)

    if dx is None or dy is None or num_matches <= 15:
        raise ValueError("ORB-based alignment failed or not enough matches.")

    method = f"ORB + Affine (matches: {num_matches})"
    print(f"Estimated Offset (pixels): X = {dx:.2f}, Y = {dy:.2f} using method: {method}")
    dx*=0.03125
    dy*=0.03125
    length = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))
    new_lon,new_lat,_ = geod.fwd(prev_lon,prev_lat,angle,length)
    return new_lat,new_lon
    
    
    