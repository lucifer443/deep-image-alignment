import mmcv
import argparse
from tqdm import tqdm
import torch
import sys
import os
import os.path as osp
sys.path.insert(0, os.getcwd())

from core import two_images_warp


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate homography matrix.')
    parser.add_argument('H_mat_file', help='file stores homography matrix')
    parser.add_argument('ann_file', help='the annotation file of dataset')
    parser.add_argument('--blending-scale', type=float, default=0.5, help='blending scale.')
    parser.add_argument('--result-save-dir', type=str, default=None, help='save warp images.')
    args = parser.parse_args()

    return args


def reprojection_error(img1, img2, mask1, mask2):
    mask = mask1*mask2
    error = ((img1-img2).abs().mean(dim=0)*mask).sum() / mask.sum()
    flag = torch.isnan(error).item()
    return error, flag


def simple_blending(img1, img2, mask1, mask2, blending_scale):
    overlap_mask = mask1*mask2
    new_image = (mask1 - overlap_mask)*img1 + (mask2 - overlap_mask)*img2 \
                + (blending_scale*img1 + (1-blending_scale)*img2)*overlap_mask
    return new_image


def save_warp_result(result, filename, dirname, blending_scale):
    img1, img2, mask1, mask2 = result
    mmcv.imwrite(img1[0].permute(1, 2, 0).cpu().numpy(), osp.join(dirname, 'wimg1/'+filename))
    mmcv.imwrite(img2[0].permute(1, 2, 0).cpu().numpy(), osp.join(dirname, 'wimg2/'+filename))
    mmcv.imwrite(mask1[0].permute(1, 2, 0).cpu().numpy(), osp.join(dirname, 'mask1/'+filename))
    mmcv.imwrite(mask2[0].permute(1, 2, 0).cpu().numpy(), osp.join(dirname, 'mask2/'+filename))
    mmcv.imwrite(simple_blending(*result, blending_scale)[0].permute(1, 2, 0).cpu().numpy(), osp.join(dirname, 'blending/'+filename))
    
def main():
    args = parse_args()
    if args.result_save_dir is not None:
        save_dirname = osp.abspath(args.result_save_dir)
        mmcv.mkdir_or_exist(save_dirname+'/wimg1')
        mmcv.mkdir_or_exist(save_dirname+'/wimg2')
        mmcv.mkdir_or_exist(save_dirname+'/mask1')
        mmcv.mkdir_or_exist(save_dirname+'/mask2')
        mmcv.mkdir_or_exist(save_dirname+'/blending')
    H_matrix_list = mmcv.load(args.H_mat_file)
    anns = mmcv.load(args.ann_file)
    dirname = osp.dirname(osp.abspath(args.ann_file))
    re_list = []
    nan_list = []
    for (img1_p, img2_p), H_mat in tqdm(zip(anns, H_matrix_list)):
        img1 = mmcv.imread(osp.join(dirname, img1_p))
        img2 = mmcv.imread(osp.join(dirname, img2_p))
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float().cuda()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float().cuda()
        H_mat_tensor = torch.from_numpy(H_mat).float().cuda()
        with torch.no_grad():
            wimg1, wimg2, mask1, mask2 = two_images_warp(img1_tensor, img2_tensor, H_mat_tensor)
            error, isnan = reprojection_error(wimg1, wimg2, mask1, mask2)
            if not isnan:
                re_list.append(error)
                if args.result_save_dir is not None:
                    save_warp_result([wimg1, wimg2, mask1, mask2], osp.basename(img1_p), save_dirname, args.blending_scale)
            else:
                nan_list.append(img1_p)
    # import pdb
    # pdb.set_trace()
    print('Mean reprojection error: ', sum(re_list).item()/len(re_list))
    print('Total nan pairs: ', len(nan_list))
    print('Nan image pairs: ')
    print('\n'.join(nan_list))


if __name__ == '__main__':
    main()