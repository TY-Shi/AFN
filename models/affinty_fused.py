import torch
import torchvision.transforms.functional as F


def make_gt_toaffinity(gt, size):
    # N 1 H W
    # Check
    gt_pad = F.pad(gt, (size, size, size, size, 0, 0, 0, 0), "constant")
    # print("gt_pad_shape",gt_pad.shape)

    feature_pad_group = [torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, :-2*size, size:-size])),  # left
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, 2*size:, size:-size])),  # right
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, size:-size, :-2*size])),  # up
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, size:-size, 2*size:])),  # down
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, 2*size:, :-2*size])),  # right_up
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, 2*size:, 2*size:])),  # right_down
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, :-2*size, :-2*size])),  # left_up
                         torch.logical_not(torch.logical_xor(gt, gt_pad[:, :, :-2*size, 2*size:]))]  # left_down

    return feature_pad_group


def get_groupfeature(feature, size):
    # Check
    feature_pad_group = [feature[:, :, 2*size:, size:-size],  # right
                         feature[:, :, :-2*size, size:-size],  # left
                         feature[:, :, size:-size, :-2*size],  # up
                         feature[:, :, size:-size, 2*size:],  # down
                         feature[:, :, 2*size:, :-2*size],  # right_up
                         feature[:, :, 2*size:, 2*size:],  # right_down
                         feature[:, :, :-2*size, :-2*size],  # left_up
                         feature[:, :, :-2*size, 2*size:]]  # left_down
    
    return feature_pad_group
