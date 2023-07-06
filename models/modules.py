import torch
import torch.nn as nn
import torch.nn.functional as F
from models.affinty_fused import make_gt_toaffinity, get_groupfeature
import numpy as np


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(1, 1, kernel_size=(5, 5), padding=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        out = self.activate(out)
        return out


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class GAU(nn.Module):
    def __init__(self, in_channels, use_gau=True, reduce_dim=False, out_channels=None):
        super(GAU, self).__init__()
        self.use_gau = use_gau
        self.reduce_dim = reduce_dim

        if self.reduce_dim:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            in_channels = out_channels

        if self.use_gau:
            self.sa = SpatialAttention()
            self.ca = ChannelAttention(in_channels)

            self.reset_gate = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, y):
        if self.reduce_dim:
            x = self.down_conv(x)

        if self.use_gau:
            y = F.interpolate(y, x.shape[-2:], mode='bilinear', align_corners=True)

            comx = x * y
            resx = x * (1 - y)  # bs, c, h, w

            x_sa = self.sa(resx)  # bs, 1, h, w
            x_ca = self.ca(resx)  # bs, c, 1, 1

            O = self.reset_gate(comx)
            M = x_sa * x_ca

            RF = M * x + (1 - M) * O
        else:
            RF = x
        return RF


class Self_fusegt(nn.Module):
    def __init__(self, temp_size=1, affinity=3, fuse_type='Size'):
        super(Self_fusegt, self).__init__()
        # input channel_32
        self.temp_size = temp_size
        self.affinity = affinity
        self.fuse_type = fuse_type

    def forward(self, input_feature, gt, weights=None):
        if self.temp_size > 1:
            weight = weights.cuda()
            fuse_feature = input_feature
            mult_size = self.affinity
            for i in range(3):
                size = (mult_size[i] - 1) // 2
                feature_pad = F.pad(input_feature, (size, size, size, size, 0, 0, 0, 0), "constant")
                feature_pad_group = get_groupfeature(feature_pad, size)
                for direction in range(0, 8):

                    aff_feature = gt[:, direction + 8 * i, :, :].unsqueeze(1)
                    aff_feature = aff_feature.float()
                    aff_direction = aff_feature

                    if weights != None:
                        new_weight = torch.unsqueeze(weights[:, i, :, :], dim=1)
                        fuse_direction = feature_pad_group[direction] * aff_direction * new_weight
                    else:
                        # weight[N,3,H,W]
                        weight = F.softmax(weight, dim=1)
                        new_weight = torch.unsqueeze(weights[:, i, :, :], dim=1)
                        fuse_direction = feature_pad_group[direction] * aff_direction * new_weight

                    fuse_feature = fuse_feature + fuse_direction
        else:
            size = (self.affinity - 1) // 2

            feature_pad = F.pad(input_feature, (size, size, size, size, 0, 0, 0, 0),
                                "constant")
            feature_pad_group = get_groupfeature(feature_pad, size)
            fuse_feature = input_feature
            for direction in range(0, 8):
                aff_direction = gt[:, direction, :, :].unsqueeze(1)
                aff_direction = aff_direction.float()
                fuse_direction = feature_pad_group[direction] * aff_direction
                fuse_feature = fuse_feature + fuse_direction
            weights = None

        return weights, fuse_feature


class BaseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, f_channels):
        super(BaseDecoder, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, f_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(f_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, rf):
        x = self.up_conv(x, output_size=rf.size())

        # padding
        diffY = rf.size()[2] - x.size()[2]
        diffX = rf.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        y = self.conv1(torch.cat([x, rf], dim=1))
        y = self.conv2(y)

        return y


# mult affinity to fuse feature
class Super_multaffSegdecoder(nn.Module):

    def __init__(self, in_channels, out_channels, f_channels, use_topo=True, up=True, bottom=False,
                 affinity=3, fuse_layer=True):
        super(Super_multaffSegdecoder, self).__init__()
        self.use_topo = use_topo
        self.up = up
        self.bottom = bottom

        self.affinity = affinity

        if affinity is not None:
            self.affinity_layer = True
        else:
            self.affinity_layer = False

        self.fuse_layer = fuse_layer

        if isinstance(affinity, list) and len(affinity) > 1:
            self.multsize_layer = True
            self.tmp_size = len(affinity)
        else:
            self.multsize_layer = False
            self.tmp_size = 1

        # topo mean use the
        if self.up:
            # segmentation
            self.up_s = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            # boundary
            self.up_t = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_s = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # same size
        # seg accept the connect information from encoder
        self.decoder_s = nn.Sequential(
            nn.Conv2d(out_channels + f_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # segmentation final layer for different size
        self.inner_s = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        if self.bottom:
            self.st = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        if self.use_topo:
            self.decoder_t = nn.Sequential(
                nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.s_to_t = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.res_s = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            # use affinity boundary final layer
            if self.multsize_layer:
                self.fuse_sf_with_a = Self_fusegt(temp_size=self.tmp_size, affinity=self.affinity)
                self.inner_t = nn.Sequential(
                    nn.Conv2d(out_channels, 8 * self.tmp_size, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid()
                )

                self.weight_layer = nn.Sequential(
                    nn.Conv2d(out_channels, 3, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid()
                )
            else:
                self.fuse_sf_with_8 = Self_fusegt(temp_size=self.tmp_size, affinity=self.affinity)
                self.inner_t_8 = nn.Sequential(
                    nn.Conv2d(out_channels, 8, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid()
                )

    # rf:connect feature x_s:segmentation branch x_t:affinity branch
    def forward(self, x_s, x_t, rf):
        w_cls = None
        s_res = None
        if self.use_topo:
            if self.bottom:
                x_t = self.st(x_t)
            # bs, c, h, w = x_s.shape
            x_s = self.up_s(x_s)
            x_t = self.up_t(x_t)

            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x_t = F.pad(x_t, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            #print(x_s.shape, rf.shape)
            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)

            s_t = self.s_to_t(s)  # the fig don't have conv  s to t need to use?

            t = torch.cat((x_t, s_t), dim=1)
            x_t = self.decoder_t(t)
            # t_s = self.t_to_s(x_t)
            w_cls = None
            if self.multsize_layer:
                t_cls = self.inner_t(x_t)  # mult_size to use the affinity map to fuse
                w_cls = self.weight_layer(x_t)
            else:
                t_cls = self.inner_t_8(x_t)  # single_size
                # w_cls = None

            if self.affinity_layer == True:
                if self.multsize_layer == True:  # with affinity gt and mult layer
                    mean_3 = torch.mean(t_cls[:, :8, :, :], dim=1, keepdim=True)
                    mean_5 = torch.mean(t_cls[:, 8:16, :, :], dim=1, keepdim=True)
                    mean_7 = torch.mean(t_cls[:, 16:, :, :], dim=1, keepdim=True)
                    enhance_feature = torch.zeros(t_cls.shape).cuda()
                    enhance_feature[:, :8, :, :] = t_cls[:, :8, :, :] - mean_3
                    enhance_feature[:, 8:16, :, :] = t_cls[:, 8:16, :, :] - mean_5
                    enhance_feature[:, 16:, :, :] = t_cls[:, 16:, :, :] - mean_7
                    enhance_feature[enhance_feature >= 0] = 1.0
                    enhance_feature[enhance_feature < 0] = 0
                    # N,8,H,W
                    # weight, fuse_segf = self.fuse_sf_with_a(s, size_3_feature)
                    # enhance_feature = 1 - enhance_feature
                    weight, fuse_segf = self.fuse_sf_with_a(s, enhance_feature, w_cls)
                    fuse_segf = torch.sigmoid(fuse_segf)

                else:
                    mean_3 = torch.mean(t_cls, dim=1, keepdim=True)
                    enhance_feature = t_cls - mean_3
                    enhance_feature[enhance_feature >= 0] = 1.0
                    enhance_feature[enhance_feature < 0] = 0
                    # enhance_feature = 1 - enhance_feature
                    weight, fuse_segf = self.fuse_sf_with_8(s, enhance_feature)

                    w_cls = None
            else:
                # fuse_segf = s
                fuse_segf = s
                # fuse_segf = torch.sigmoid(fuse_segf)
            s_res = self.res_s(fuse_segf)

            x_s = s + s_res
            # x_s = s * s_res
            # t:affinity s:segmentation
            s_cls = self.inner_s(x_s)
        else:
            x_s = self.up_s(x_s)
            # x_b = self.up_b(x_b)
            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)
            x_s = s
            x_t = x_s
            t_cls = None
            s_cls = self.inner_s(x_s)
        # return x_s, x_t, s_cls, t_cls, w_cls
        return x_s, x_t, s_cls, t_cls, w_cls, s_res, s



