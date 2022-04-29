import mmcv
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import os
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate homography matrix.')
    parser.add_argument('ann_file', help='the annotation file of dataset')
    parser.add_argument('save_file', help='file stores homography matrix')
    args = parser.parse_args()

    return args

def compute_homography(img_1, img_2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [[m] for m, n in matches if m.distance < 0.5* n.distance]
    if len(good) < 4:
        return np.eye(3), False
    pts1, pts2 = [], []
    for f, in good:
        pts1.append(kp1[f.queryIdx].pt)
        pts2.append(kp2[f.trainIdx].pt)
    H, _ = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv2.RANSAC)
    if H is not None:
        return H, True
    else:
        return np.eye(3), False

def main():
    args = parse_args()
    anns = mmcv.load(args.ann_file)
    dirname = osp.dirname(osp.abspath(args.ann_file))
    h_mat_list = []
    for img1_p, img2_p in tqdm(anns):
        img1 = mmcv.imread(osp.join(dirname, img1_p))
        img2 = mmcv.imread(osp.join(dirname, img2_p))
        h_mat, flag = compute_homography(img1, img2)
        if not flag:
            print(img1_p)
        h_mat_list.append(h_mat)
    print('save homography matrix to file: ', osp.abspath(args.save_file))
    mmcv.dump(h_mat_list, args.save_file)
    
if __name__ == '__main__':
    main()