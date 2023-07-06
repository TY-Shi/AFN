import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.vgg import get_vgg16_layer, vgg16_bn
from models.backbone.base import get_base
from models.modules import GAU, Super_multaffSegdecoder
from models.loss_function.ASCLoss import ASCLoss
import numpy as np
from PIL import Image


# make unet like supervise affinity:Vggunet_saff

def make_gt_to_affinity(gt_mask, tmp_size):
    # h, w = gt_mask.shape[2], gt_mask.shape[3] NCHW
    # gt_pad = F.pad(gt_mask,(tmp_size,tmp_size,tmp_size,tmp_size,0,0,0,0),"constant")
    affinity_gt = torch.zeros([gt_mask.shape[0], 8, gt_mask.shape[2], gt_mask.shape[3]])
    gt_pad = F.pad(gt_mask, [tmp_size, tmp_size, tmp_size, tmp_size], mode='constant', value=0)
    # print("gt_pad_shape_value",gt_pad.shape)
    gt_pad_bool = gt_pad.bool()
    gt_array_bool = gt_mask.bool()
    right = torch.bitwise_xor(gt_pad_bool[:, :, 2*tmp_size:, tmp_size:-tmp_size], gt_array_bool)
    left = torch.bitwise_xor(gt_pad_bool[:, :, :-2*tmp_size, tmp_size:-tmp_size], gt_array_bool)
    up = torch.bitwise_xor(gt_pad_bool[:, :, tmp_size:-tmp_size, :-2*tmp_size], gt_array_bool)
    down = torch.bitwise_xor(gt_pad_bool[:, :, tmp_size:-tmp_size, 2*tmp_size:], gt_array_bool)
    diag1 = torch.bitwise_xor(gt_pad_bool[:, :, 2*tmp_size:, :-2*tmp_size], gt_array_bool)  # right_up
    diag2 = torch.bitwise_xor(gt_pad_bool[:, :, 2*tmp_size:, 2*tmp_size:], gt_array_bool)  # right_down
    diag3 = torch.bitwise_xor(gt_pad_bool[:, :, :-2*tmp_size, :-2*tmp_size], gt_array_bool)  # left_up
    diag4 = torch.bitwise_xor(gt_pad_bool[:, :, :-2*tmp_size, 2*tmp_size:], gt_array_bool)  # left_up
    affinity_gt[:, 0, :, :] = 1 + (-1) * right[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 1, :, :] = 1 + (-1) * left[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 2, :, :] = 1 + (-1) * up[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 3, :, :] = 1 + (-1) * down[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 4, :, :] = 1 + (-1) * diag1[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 5, :, :] = 1 + (-1) * diag2[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 6, :, :] = 1 + (-1) * diag3[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 7, :, :] = 1 + (-1) * diag4[:, 0, :, :].type(affinity_gt.type())

    return affinity_gt


def make_gt_to_affinity_mult(gt_mask, tmp_size=3):
    # h, w = gt_mask.shape[2], gt_mask.shape[3] NCHW
    # gt_pad = F.pad(gt_mask,(tmp_size,tmp_size,tmp_size,tmp_size,0,0,0,0),"constant")
    affinity_gt = torch.zeros([gt_mask.shape[0], 8 * tmp_size, gt_mask.shape[2], gt_mask.shape[3]])
    # for i_size in range(1, tmp_size + 1):
    mult_size = [1, 4, 7]
    for i in range(tmp_size):
        i_size = mult_size[i]
        gt_pad = F.pad(gt_mask, [i_size, i_size, i_size, i_size], mode='constant', value=0)
        gt_pad_bool = gt_pad.bool()
        gt_array_bool = gt_mask.bool()

        right = torch.bitwise_xor(gt_pad_bool[:, :, 2*i_size:, i_size:-i_size], gt_array_bool)
        left = torch.bitwise_xor(gt_pad_bool[:, :, :-2*i_size, i_size:-i_size], gt_array_bool)
        up = torch.bitwise_xor(gt_pad_bool[:, :, i_size:-i_size, :-2*i_size], gt_array_bool)
        down = torch.bitwise_xor(gt_pad_bool[:, :, i_size:-i_size, 2*i_size:], gt_array_bool)
        diag1 = torch.bitwise_xor(gt_pad_bool[:, :, 2*i_size:, :-2*i_size], gt_array_bool)  # right_up
        diag2 = torch.bitwise_xor(gt_pad_bool[:, :, 2*i_size:, 2*i_size:], gt_array_bool)  # right_down
        diag3 = torch.bitwise_xor(gt_pad_bool[:, :, :-2*i_size, :-2*i_size], gt_array_bool)  # left_up
        diag4 = torch.bitwise_xor(gt_pad_bool[:, :, :-2*i_size, 2*i_size:], gt_array_bool)  # left_down

        affinity_gt[:, 0 + i * 8, :, :] = 1 + (-1) * right[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 1 + i * 8, :, :] = 1 + (-1) * left[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 2 + i * 8, :, :] = 1 + (-1) * up[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 3 + i * 8, :, :] = 1 + (-1) * down[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 4 + i * 8, :, :] = 1 + (-1) * diag1[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 5 + i * 8, :, :] = 1 + (-1) * diag2[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 6 + i * 8, :, :] = 1 + (-1) * diag3[:, 0, :, :].type(affinity_gt.type())
        affinity_gt[:, 7 + i * 8, :, :] = 1 + (-1) * diag4[:, 0, :, :].type(affinity_gt.type())

    return affinity_gt


class Vggunet_saff_mult(nn.Module):
    def __init__(self, backbone, in_ch, use_fim, affinity, affinity_supervised, up, classes=1, steps=3, reduce_dim=False, tmp_size=3):
        super(Vggunet_saff_mult, self).__init__()
        # if use the pretrain need to set True
        # pretrained = True
        pretrained = False
        assert backbone in ['vgg16', 'base']
        assert classes == 1
        assert len(use_fim) == 4
        self.backbone = backbone
        self.use_fim = use_fim
        self.steps = steps
        self.up = up
        self.reduce_dim = reduce_dim
        self.tmp_size = tmp_size
        self.bce_loss = nn.BCELoss()
        self.asc_loss = ASCLoss()
        # contral the affinity fuse take in which layer 0 is top layer

        # self.affinity_layer = [True, True, True, True]
        # self.affinity_layer = [True, True, True, True]
        self.affinity = affinity
        self.affinity_supervised = affinity_supervised
        # self.multsize_layer = [True, False, False, False] if affinity

        if self.backbone == 'vgg16':
            print('INFO: Using VGG_16 bn')
            vgg16 = vgg16_bn(pretrained=pretrained)
            filters = [64, 128, 256, 512, 512]
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif self.backbone == 'base':
            print('INFO: Using base backbone')
            filters = [64, 128, 256, 512, 512]
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_base(in_ch, filters)
        else:
            raise RuntimeError('Backbone ', backbone, 'is not implemented.')

        if reduce_dim:
            reduce_filters = [32, 32, 32, 32, 32]
        else:
            reduce_filters = filters

        self.skip_blocks = []
        for i in range(5):
            self.skip_blocks.append(GAU(filters[i], True, reduce_dim, reduce_filters[i]))

        self.decoder = []
        index = use_fim.index(False) if False in use_fim else len(use_fim) - 1
        print("index:", index)
        # decoder[0] on the top layer
        for i in range(4):
            if i == (index):
                self.decoder.append(
                    Super_multaffSegdecoder(reduce_filters[i + 1], reduce_filters[i], reduce_filters[i], use_fim[i],
                                            up[i], affinity=self.affinity[i], bottom=True,
                                            fuse_layer=self.affinity_supervised[i]))
            else:
                self.decoder.append(
                    Super_multaffSegdecoder(reduce_filters[i + 1], reduce_filters[i], reduce_filters[i], use_fim[i],
                                            up[i], affinity=self.affinity[i], fuse_layer=self.affinity_supervised[i]))
        self.skip_blocks = nn.ModuleList(self.skip_blocks)
        self.decoder = nn.ModuleList(self.decoder)

        self.filters = filters
        self.reduce_filters = reduce_filters

    def forward(self, batch_dict):
        img = batch_dict['img']
        bs, c, h, w = img.shape

        output_dict = {}
        y = torch.zeros([bs, 1, h, w], device=img.device)
        bc_x1 = self.layer0(img)
        bc_x2 = self.layer1(bc_x1)
        bc_x3 = self.layer2(bc_x2)
        bc_x4 = self.layer3(bc_x3)
        bc_x5 = self.layer4(bc_x4)
        # print(img.shape, bc_x1.shape, bc_x2.shape, bc_x3.shape, bc_x4.shape, bc_x5.shape)
        feature_dict = {}
        for i in range(self.steps):
            x1 = self.skip_blocks[0](bc_x1, y)
            x2 = self.skip_blocks[1](bc_x2, y)
            x3 = self.skip_blocks[2](bc_x3, y)
            x4 = self.skip_blocks[3](bc_x4, y)
            x5 = self.skip_blocks[4](bc_x5, y)

            x5_s = x5
            x5_b = x5

            x4_s, x4_b, s4_cls, b4_cls, _, Tfeature4, Ffeature4 = self.decoder[-1](x5_s, x5_b, x4)
            x3_s, x3_b, s3_cls, b3_cls, _, Tfeature3, Ffeature3 = self.decoder[-2](x4_s, x4_b, x3)
            x2_s, x2_b, s2_cls, b2_cls, _, Tfeature2, Ffeature2 = self.decoder[-3](x3_s, x3_b, x2)
            x1_s, x1_b, s1_cls, b1_cls, weight, Tfeature1, Ffeature1 = self.decoder[-4](x2_s, x2_b, x1)
            output_dict['step_' + str(i) + '_output_mask'] = [s1_cls, s2_cls, s3_cls, s4_cls]
            output_dict['step_' + str(i) + '_output_affinity'] = [b1_cls, b2_cls, b3_cls, b4_cls]
            output_dict['step_' + str(i) + "weight"] = weight
            if i == 2:
                feature_dict['step_' + str(i) + "Tfeature"] = [Tfeature1, Tfeature2, Tfeature3, Tfeature4]
                feature_dict['step_' + str(i) + "Ffeature"] = [Ffeature1, Ffeature2, Ffeature3, Ffeature4]
                feature_dict['step_' + str(i) + "weight"] = weight
            y = s1_cls

        output_dict['output'] = y

        return output_dict

    def compute_objective(self, output_dict, batch_dict):
        loss_dict = {}

        gt_mask = batch_dict['anno_mask']
        # gt_boundary = batch_dict['anno_boundary']
        h, w = gt_mask.shape[2], gt_mask.shape[3]
        # need change gt to affinity
        total_loss = None
        lamba_asc = 5
        # lamba_asc = 6
        for i in range(self.steps):
            pred_mask = output_dict['step_' + str(i) + '_output_mask']  # list
            # pred_boundary = output_dict['step_' + str(i) + '_output_boundary'] # list
            pred_affinity = output_dict['step_' + str(i) + '_output_affinity']  # list
            weight = output_dict['step_' + str(i) + "weight"]
            # with torch.no_grad():
            #    print("weight",torch.unique(weight))
            step_mask_loss = None
            for k in range(len(pred_mask)):
                inner_pred = pred_mask[k]
                if inner_pred is None:
                    continue
                inner_pred = F.interpolate(inner_pred, (h, w), mode='bilinear', align_corners=True)
                mask_loss = self.bce_loss(inner_pred, gt_mask.float())
                if step_mask_loss is None:
                    step_mask_loss = torch.zeros_like(mask_loss).to(mask_loss.device)
                step_mask_loss = step_mask_loss + mask_loss
            step_mask_loss = step_mask_loss / len(pred_mask)

            step_topo_loss = None
            for k in range(len(pred_affinity)):
                if self.affinity_supervised[k] is True:

                    inner_pred = pred_affinity[k]
                    if inner_pred is None:
                        continue

                    gt_affintiy_or = gt_mask.clone()

                    if isinstance(self.affinity[k], list) and len(self.affinity[k]) > 1:
                        gt_affinity = make_gt_to_affinity_mult(gt_affintiy_or, tmp_size=self.tmp_size)
                    else:
                        gt_affinity = make_gt_to_affinity(gt_affintiy_or, tmp_size=2)

                    gt_affinity = gt_affinity.to(device='cuda')

                    inner_pred = F.interpolate(inner_pred, (h, w), mode='bilinear', align_corners=True)

                    bce_loss = self.bce_loss(inner_pred, gt_affinity.float())
                    if k == 0:
                        asc_loss = self.asc_loss(inner_pred, gt_affinity.float())

                        topo_loss = bce_loss + lamba_asc * asc_loss

                    else:
                        topo_loss = bce_loss
                    if step_topo_loss is None:
                        step_topo_loss = torch.zeros_like(topo_loss).to(topo_loss.device)
                    step_topo_loss = step_topo_loss + topo_loss
            if step_topo_loss is not None:
                step_topo_loss = step_topo_loss / len(pred_affinity)
            else:
                step_topo_loss = torch.zeros_like(step_mask_loss).to(step_mask_loss.device)

            if total_loss is None:
                total_loss = torch.zeros_like(step_mask_loss).to(step_mask_loss.device)
            total_loss = total_loss + step_mask_loss + step_topo_loss

            loss_dict['step_' + str(i) + 'total_loss'] = step_mask_loss + step_topo_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def train_mode(self):
        self.train()

    def test_mode(self):
        self.eval()
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()
