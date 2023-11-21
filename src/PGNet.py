import torch
import torch.nn as nn
import torch.nn.functional as F
from Res import resnet50
from Swin import Swintransformer
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

Act = nn.ReLU


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU, Act, nn.AdaptiveAvgPool2d, nn.Softmax)):
            pass
        else:
            m.initialize()


class CatFeatures(nn.Module):
    def __init__(self):
        super(CatFeatures, self).__init__()
        self.cat1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, s1, s2, s3, s4, r2, r3, r4, r5):
        # print(s1.size(), r2.size(), "s1_size"*30)
        c1 = torch.cat((s1, r2), 1)
        c1 = self.cat1(c1)

        c2 = torch.cat((s2, r3), 1)
        c2 = self.cat2(c2)

        c3 = torch.cat((s3, r4), 1)
        c3 = self.cat3(c3)

        c4 = torch.cat((s4, r5), 1)
        c4 = self.cat4(c4)

        return c1, c2, c3, c4

    def initialize(self):
        weight_init(self)


class ConvLarge(nn.Module):
    def __init__(self):
        super(ConvLarge, self).__init__()
        self.cat1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat2 = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat3 = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat4 = nn.Sequential(
            nn.Conv2d(1536, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, s1, s2, s3, s4):
        # print(s1.size(), r2.size(), "s1_size"*30)
        c1 = self.cat1(s1)

        c2 = self.cat2(s2)

        c3 = self.cat3(s3)

        c4 = self.cat4(s4)

        return c1, c2, c3, c4

    def initialize(self):
        weight_init(self)


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

        self.conv1g = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1g = nn.BatchNorm2d(64)


    def forward(self, left, down, glob=None):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')



        out1h = self.bn1h(self.conv1h(left ))
        out2h = self.bn2h(self.conv2h(out1h))
        out1v = self.bn1v(self.conv1v(down ))
        out2v = self.bn2v(self.conv2v(out1v))
        fuse  = out2h * out2v

        if glob != None:
            if glob.size()[2:] != left.size()[2:]:
                glob = F.interpolate(glob, size=left.size()[2:], mode='bilinear')
            out1g = self.bn1g(self.conv1g(glob))
            out3h = self.bn3h(self.conv3h(fuse)) + out1h + out1g
            out3v = self.bn3v(self.conv3v(fuse)) + out1v + out1g
        else:
            out3h = self.bn3h(self.conv3h(fuse)) + out1h
            out3v = self.bn3v(self.conv3v(fuse)) + out1v
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, glob, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')

            out5v = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v, glob)
            out3h, out3v = self.cfm34(out3h+refine3, out4v, glob)
            out2h, pred  = self.cfm23(out2h+refine2, out3v, glob)
        else:
            out4h, out4v = self.cfm45(out4h, out5v, glob)
            out3h, out3v = self.cfm34(out3h, out4v, glob)
            out2h, pred  = self.cfm23(out2h, out3v, glob)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)


class PGNet(nn.Module):
    def __init__(self, cfg=None):
        super(PGNet, self).__init__()
        self.cfg = cfg
        self.catfeature = CatFeatures()
        self.conv_large = ConvLarge()
        self.squeeze5 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


        # self.resnet = resnet50()
        # self.resnet.load_state_dict(torch.load('../pre/resnet50.pth'), strict=False)
        # # swin_base_224
        # self.swin = Swintransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        #          embed_dim=128, depths=[2, 2, 18,2], num_heads=[4, 8, 16, 32],
        #          window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2)
        # self.swin.load_state_dict(torch.load('../pre/swin_base_patch4_window7_224_22k.pth')['model'], strict=False)

        #
        # # swin_base_384
        self.swin = Swintransformer(img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                                    embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                    window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2)
        self.swin.load_state_dict(torch.load('../pre/swin_base_patch4_window12_384_22k.pth')['model'], strict=False)

        # # swin_large_224
        # self.swin = Swintransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        #          embed_dim=192, depths=[2, 2, 18,2], num_heads=[6, 12, 24, 48],
        #          window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2)
        # self.swin.load_state_dict(torch.load('../pre/swin_large_patch4_window7_224_22k.pth')['model'], strict=False)

        # # swin_large_384
        # self.swin = Swintransformer(img_size=384, patch_size=4, in_chans=3, num_classes=1000,
        #                             embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
        #                             window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2)
        # self.swin.load_state_dict(torch.load('../pre/swin_large_patch4_window12_384_22k.pth')['model'], strict=False)


        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain = torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k, v in pretrain.items():
                new_state_dict[k[7:]] = v
            self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x, shape=None, mask=None):
        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)

        # r2, r3, r4, r5 = self.resnet(x)
        # print(r2.size(), r3.size(), r4.size(), r5.size(), "r2-r5--size=============================================")
        s1, s2, s3, s4 = self.swin(y)

       # out2h, out3h, out4h, out5v = self.squeeze2(s1), self.squeeze3(s2), self.squeeze4(s3), self.squeeze5(s4)
        #

        # out2h, out3h, out4h, out5v = self.catfeature(s1, s2, s3, s4, r2, r3, r4, r5)
        out2h, out3h, out4h, out5v = self.catfeature(s1, s2, s3, s4, s1, s2, s3, s4)
        # out2h, out3h, out4h, out5v = self.conv_large(s1, s2, s3, s4)
        glob = out5v
        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v, glob)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, glob, pred1)

        # print("out2h, out3h, out4h, out5v")

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        # print(pred1.size(), pred2.size(), out2h.size(), out3h.size(), out4h.size(), out5h.size(), "="*100)

        return pred1, pred2, out2h, out3h, out4h, out5h




