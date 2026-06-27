import torch
import torch.nn as nn
import torch.nn.functional as F
class Rep3DConv(nn.Module):
    def __init__(self):
        super(Rep3DConv, self).__init__()

        self.T = 20
        self.t1 = 16
        self.t2 = 6
        self.t3 = 1

        self.conv3d_to_res_1_2 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.ReLU(),
        )

        self.conv3d_to_res_1_4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(0, 1, 1), stride=(2, 2, 2)),
            nn.ReLU(),
        )

        self.conv3d_to_res_1_8 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=(5, 3, 3), padding=(0, 1, 1), stride=(2, 2, 2)),
            nn.ReLU(),
        )

        self.conv_intra_1_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_fuse_1_2 = nn.Conv2d(16 * self.t1, 32, kernel_size=3, padding=1)
        self.deconv_res1_2 = nn.ConvTranspose2d(32 + 64, 64, kernel_size=4, padding=1, stride=2)

        self.conv_intra_1_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_fuse_1_4 = nn.Conv2d(32 * self.t2, 48, kernel_size=3, padding=1)
        self.deconv_res1_4 = nn.ConvTranspose2d(48 + 64, 64, kernel_size=4, padding=1, stride=2)

        self.conv_intra_1_8 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_fuse_1_8 = nn.Conv2d(48 * self.t3, 64, kernel_size=3, padding=1)
        self.deconv_res1_8 = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)

        self.out_layer = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def trans_3d_to_2d_separate(self, x):
        B, C, T, H, W = x.shape
        self.T = T
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        return x

    def trans_2d_to_2d_for_fuse(self, x):
        BT, C, H, W = x.shape
        B = BT // self.T
        x = x.view(B, self.T, C, H, W).view(B, self.T * C, H, W)
        return x

    def trans_2d_separate_to_3d(self, x):
        BT, C, H, W = x.shape
        B = BT // self.T
        x = x.view(B, self.T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return x

    def forward(self, x):
        f3d = x.unsqueeze(dim=1)
        f3d = self.conv3d_to_res_1_2(f3d)

        f2d = self.trans_3d_to_2d_separate(f3d)
        intra_feat2 = f2d + self.conv_intra_1_2(f2d)
        feat2_for_fuse = self.trans_2d_to_2d_for_fuse(intra_feat2)
        cat_feat2 = self.conv_fuse_1_2(feat2_for_fuse)

        f3d = self.conv3d_to_res_1_4(self.trans_2d_separate_to_3d(intra_feat2))
        f2d = self.trans_3d_to_2d_separate(f3d)
        intra_feat4 = f2d + self.conv_intra_1_4(f2d)
        feat4_for_fuse = self.trans_2d_to_2d_for_fuse(intra_feat4)
        cat_feat4 = self.conv_fuse_1_4(feat4_for_fuse)

        f3d = self.conv3d_to_res_1_8(self.trans_2d_separate_to_3d(intra_feat4))
        f2d = self.trans_3d_to_2d_separate(f3d)
        intra_feat8 = f2d + self.conv_intra_1_8(f2d)
        feat8_for_fuse = self.trans_2d_to_2d_for_fuse(intra_feat8)
        cat_feat8 = self.conv_fuse_1_8(feat8_for_fuse)

        feat8 = self.deconv_res1_8(cat_feat8)
        if feat8.shape[-2:] != cat_feat4.shape[-2:]:
            feat8 = F.interpolate(feat8, size=cat_feat4.shape[-2:], mode='bilinear', align_corners=True)

        feat4 = self.deconv_res1_4(torch.cat([feat8, cat_feat4], dim=1))
        if feat4.shape[-2:] != cat_feat2.shape[-2:]:
            feat4 = F.interpolate(feat4, size=cat_feat2.shape[-2:], mode='bilinear', align_corners=True)

        feat2 = self.deconv_res1_2(torch.cat([feat4, cat_feat2], dim=1))
        out = self.out_layer(feat2)

        return [out, cat_feat2, cat_feat4, cat_feat8]


backwarp_tenGrid = {}


def resize_flow(flow, size):
    B, C, h, w = flow.shape
    H, W = size

    if (h, w) == (H, W):
        return flow

    scale_x = W / w
    scale_y = H / h

    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
    flow[:, 0:1] *= scale_x
    flow[:, 1:2] *= scale_y
    return flow


def bwarp(tenInput, tenFlow):
    device = tenInput.device
    B, C, H, W = tenInput.shape

    k = (str(device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
        tenVertical = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], dim=1)

    flow_norm = torch.cat([
        tenFlow[:, 0:1] / ((W - 1.0) / 2.0),
        tenFlow[:, 1:2] / ((H - 1.0) / 2.0)
    ], dim=1)

    grid = (backwarp_tenGrid[k] + flow_norm).permute(0, 2, 3, 1)

    return F.grid_sample(
        input=tenInput,
        grid=grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

def get_norm(norm='none'):
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'layer':
        norm_layer = nn.LayerNorm
    elif norm == 'none':
        norm_layer = nn.Identity
    else:
        print("=====Wrong norm type!======")
    return norm_layer

def conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm = 'none'):
    norm_layer = get_norm(norm)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,padding=padding),
        norm_layer(out_ch),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

def deconv(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm = 'none'):
    norm_layer = get_norm(norm)
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,padding=padding),
        norm_layer(out_ch),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv(in_ch=in_ch, out_ch=in_ch, kernel_size=3),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,stride=1,padding=1),
        )
    def forward(self, x):
        res = self.conv(x)
        x = x + res
        return x

class downConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv1 = ResBlock(in_ch)
        self.conv2 = conv(in_ch, in_ch, kernel_size=1,stride=1,padding=0)
        self.down = conv(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x_skip = self.conv2(x)
        x = self.down(x)
        return x, x_skip

class upConv(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.deconv = deconv(in_ch, out_ch)
        self.conv_skip = conv(skip_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv1 = conv(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = ResBlock(out_ch)
        
    def forward(self, x, x_skip):
        x = self.deconv(x)
        x_skip = self.conv_skip(x_skip)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

class Unet(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 3, base_ch = 32, depth = 3) -> None:
        super().__init__()
        self.depth = depth
        self.head = nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1)

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        for i in range(self.depth):
            self.down_path.append(downConv(base_ch*2**i, base_ch*2**(i+1)))
        
        self.bottom = nn.Sequential(
            ResBlock(base_ch*2**self.depth),
            ResBlock(base_ch*2**self.depth),
        )

        for i in range(1,self.depth+1):
            self.up_path.append(upConv(base_ch*2**i, base_ch*2**(i-1), base_ch*2**(i-1)))

        self.pred = nn.Conv2d(base_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x_skip_list = []
        for i in range(self.depth):
            x, x_skip = self.down_path[i](x)
            x_skip_list.append(x_skip)
        x = self.bottom(x)
        for i in range(self.depth-1, -1, -1):
            x = self.up_path[i](x, x_skip_list[i])
        x = self.pred(x)
        return x



class ProjImage(nn.Module):
    def __init__(self, input_dim=128):
        super(ProjImage, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class FlowUpsampleRefine(nn.Module):
    def __init__(self, feat_ch=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 + feat_ch * 2, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(hidden),
            ResBlock(hidden),
            nn.Conv2d(hidden, 2, 3, 1, 1)
        )

    def forward(self, flow_low, feat1, feat2, out_size):
        flow_up = resize_flow(flow_low, out_size)
        feat1_up = F.interpolate(feat1, size=out_size, mode='bilinear', align_corners=True)
        feat2_up = F.interpolate(feat2, size=out_size, mode='bilinear', align_corners=True)

        delta = self.net(torch.cat([flow_up, feat1_up, feat2_up], dim=1))
        return flow_up + delta

class SpikeFlowNet(nn.Module):
    def __init__(self, base_ch=32, depth=3):
        super().__init__()
        self.rep = Rep3DConv()
        self.proj_img = ProjImage(input_dim=128)
        self.proj_img1 = ProjImage(input_dim=32)
        self.proj_img2 = ProjImage(input_dim=48)
        self.proj_img3 = ProjImage(input_dim=64)

        self.flow_head = Unet(
            in_ch=128*3 + 1,   
            out_ch=2,
            base_ch=base_ch,
            depth=depth
        )
        self.flow_up_refine = FlowUpsampleRefine(feat_ch=128)
        

    def forward(self, spk1, spk2, img1, distance):
        feat1_list = self.rep(spk1)
        feat2_list = self.rep(spk2)

        feat1 = feat1_list[0]
        feat2 = feat2_list[0]

        flow_spk = self.flow_head(torch.cat([feat2, feat1, feat1 - feat2, distance], dim=1))
        flow_img = self.flow_up_refine(
            flow_spk,
            feat1,
            feat2,
            img1.shape[-2:]
        )
        warp_img = bwarp(img1, flow_img)

        rec1 = [self.proj_img(feat1_list[0]),
                self.proj_img1(feat1_list[1]),
                self.proj_img2(feat1_list[2]),
                self.proj_img3(feat1_list[3]),] if self.training else None
        rec2 = [self.proj_img(feat2_list[0]),
                self.proj_img1(feat2_list[1]),
                self.proj_img2(feat2_list[2]),
                self.proj_img3(feat2_list[3]),] if self.training else None

        return {
            'warp_img': warp_img,
            'flow_img': flow_img,
            'flow_spk': flow_spk,
            'rec1': rec1,
            'rec2': rec2,
        }