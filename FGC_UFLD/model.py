import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np




try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe(x, normed_mask, kernel_size, group=1, up=1):
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        assert m_h == up * h
        assert m_w == up * w
        pad = kernel_size // 2
        # print(pad)
        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        # print(pad_x.shape)
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
        # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
        res = unfold_x * normed_mask
        res = res.sum(dim=2).reshape(b, c, m_h, m_w)

        return res

    def normal_init(module, mean=0, std=0.001, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d

class PFAM(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 compressed_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=False,  # use offset generator or not
                 feature_resample_group=4,
                 comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,  # for regularization, do not matter really
                 feature_resample_norm=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(  # ALPF generator
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample

        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp',
                                                   groups=feature_resample_group, use_direct_scale=True,
                                                   kernel_size=encoder_kernel, norm=feature_resample_norm)
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(  # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(
                hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(
                hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))  # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):  # use check_point to save GPU memory
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        global mask_hr
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)  # 从hr_feat得到初始高通滤波特征
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel,
                                                          hamming=self.hamming_highpass)  # kernel归一化得到初始高通滤波
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat,
                                                                                          mask_hr_init,
                                                                                          self.highpass_kernel,
                                                                                          self.up_group,
                                                                                          1)  # 利用初始高通滤波对压缩hr_feat的高频增强 （x-x的低通结果=x的高通结果）

                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)  # 从hr_feat得到初始低通滤波特征
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel,
                                                          hamming=self.hamming_lowpass)  # kernel归一化得到初始低通滤波

                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)  # 从hr_feat得到另一部分初始低通滤波特征
                    mask_lr_lr_feat = F.interpolate(  # 利用初始低通滤波对另一部分初始低通滤波特征上采样
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                        size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat  # 将两部分初始低通滤波特征合在一起

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel,
                                                          hamming=self.hamming_lowpass)  # 得到初步融合的初始低通滤波
                    mask_hr_lr_feat = F.interpolate(  # 使用初始低通滤波对lr_feat处理，分辨率得到提高
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel,
                               self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat  # 最终高通滤波特征
                else:
                    raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
                    self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:],
                                         mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf

        if self.feature_resample:
            # print(lr_feat.shape)
            lr_feat = self.dysampler(hr_x=compressed_hr_feat,
                                     lr_x=compressed_lr_feat, feat2sample=lr_feat)
        # total=hr_feat+ lr_feat
        return hr_feat, lr_feat

class LocalSimGuidedSampler(nn.Module):
    """
    offset generator in FreqFusion
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, use_direct_scale=True, kernel_size=1, local_window=3,
                 sim_type='cos', norm=True, direction_feat='sim_concat'):
        super().__init__()
        assert scale == 2
        assert style == 'lp'

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size // 2)
        elif self.direction_feat == 'sim_concat':
            self.offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)
        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                              padding=kernel_size // 2)
            elif self.direction_feat == 'sim_concat':
                self.direct_scale = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels,
                                              kernel_size=kernel_size, padding=kernel_size // 2)
            else:
                raise NotImplementedError
            constant_init(self.direct_scale, val=0.)

        out_channels = 2 * groups
        if self.direction_feat == 'sim':
            self.hr_offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                                       padding=kernel_size // 2)
        elif self.direction_feat == 'sim_concat':
            self.hr_offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                                       padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.hr_direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 padding=kernel_size // 2)
            elif self.direction_feat == 'sim_concat':
                self.hr_direct_scale = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels,
                                                 kernel_size=kernel_size, padding=kernel_size // 2)
            else:
                raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset, scale=None):
        if scale is None: scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)

    def forward(self, hr_x, lr_x, feat2sample):
        global hr_sim, lr_sim
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == 'sim':
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        # offset = self.get_offset(hr_x, lr_x)
        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return self.sample(feat2sample, offset)

    # def get_offset_lp(self, hr_x, lr_x):
    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'direct_scale'):
            # offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (
                        self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x),
                                                                    self.scale)).sigmoid() + self.init_pos
            # offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_sim), self.scale)).sigmoid() + self.init_pos
        else:
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量，形状为[B, C, H, W]
    - k: 范围大小，表示周围KxK范围内的点

    返回：
    - 输出张量，形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape
    # 使用零填充来处理边界情况
    # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

    # 展平输入张量中每个点及其周围KxK范围内的点
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # B, CxKxK, HW
    # print(unfold_tensor.shape)
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

    # 计算余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将结果重塑回[B, KxK-1, H, W]的形状
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity



#############################################################
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        planes = int(out_planes / 2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        # 调整通道数
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class Upsample_shn(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsample_shn, self).__init__()
        self.upsample = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.upsample(x)

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Overlapping Cross-Attention (OCA)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)#(1,16,128,128)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)#(256,64,16)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))#(1,2304,256)
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))#(256,144,16)

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))##(256,64,16),(256,144,16),(256,144,16)

        # attention
        qs = qs * self.scale#(256,64,16)
        spatial_attn = (qs @ ks.transpose(-2, -1))#(256,64,144)
        spatial_attn += self.rel_pos_emb(qs)#(256,64,144)
        spatial_attn = spatial_attn.softmax(dim=-1)#(256,64,144)

        out = (spatial_attn @ vs)#(256,64,16)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)
        # (1,16,128,128)
        # merge spatial and channel
        out = self.project_out(out)#(1,48,128,128)

        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()


        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

###############################################################################
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 使用深度卷积进行查询、键、值投影
        self.query_conv = DepthConv(dim, dim)
        self.key_conv = DepthConv(dim, dim)
        self.value_conv = DepthConv(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        # 输入形状: [batch, channels, height, width]
        b, c, h, w = query.shape

        # 通过深度卷积提取特征
        q = self.query_conv(query)
        k = self.key_conv(context)
        v = self.value_conv(context)

        # 重塑为多头注意力格式 [batch, heads, seq_len, head_dim]
        q = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # [b, heads, seq_len, head_dim]
        k = k.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # 应用注意力
        out = torch.matmul(attn, v)  # [b, heads, seq_len, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)  # 恢复形状
        return out


# 膨胀残差块（Dilated Residual Block）
class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedResBlock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.point(self.depth(x))


class DiC_Wave(nn.Module):

    def __init__(self, in_channels, compressed_channels):
        super().__init__()
        self.head_dim = compressed_channels // 4

        self.dwt_conv = nn.Conv2d(in_channels * 3, compressed_channels, kernel_size=1)

        # 高频特征提取（深度卷积）
        self.feat_extract_hl = DepthConv(in_channels, in_channels)
        self.feat_extract_lh = DepthConv(in_channels, in_channels)
        self.feat_extract_hh = DepthConv(in_channels, in_channels)

        # 高频增强的交叉注意力
        self.cross_attn_hl_to_hh = CrossAttention(in_channels, num_heads=8)
        self.cross_attn_lh_to_hh = CrossAttention(in_channels, num_heads=8)

        # 高频融合卷积
        self.hf_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1)

        # 膨胀残差块
        self.dilated_res_hl = DilatedResBlock(in_channels, in_channels)
        self.dilated_res_lh = DilatedResBlock(in_channels, in_channels)
        self.dilated_res_hh = DilatedResBlock(in_channels, in_channels)

        self.hf_output = DepthConv(in_channels * 3, compressed_channels)

        # 下采样低频（输入下采样）
        self.downsample = Downsample(in_channels)

        self.q_ll = DepthConv(compressed_channels, compressed_channels)
        self.k_hf = DepthConv(compressed_channels, compressed_channels)
        self.v_hf = DepthConv(compressed_channels, compressed_channels)
        self.ll_proj = nn.Conv2d(compressed_channels, compressed_channels, 1)
        self.dropout = nn.Dropout(0.0)


    def reshape_heads(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 4, self.head_dim, H * W)
        return x.permute(0, 1, 3, 2)

    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_HL,x_LH,x_HH

    def forward(self, x):
        # ----------------
        # 1. DWT 高频
        hl, lh, hh = self.dwt_init(x)

        # ----------------
        # 2. 高频增强
        hl_feat = self.feat_extract_hl(hl)
        lh_feat = self.feat_extract_lh(lh)
        hh_feat = self.feat_extract_hh(hh)

        # 交叉注意力增强 HH
        hh_enh_v = self.cross_attn_hl_to_hh(hh_feat, hl_feat)
        hh_enh_h = self.cross_attn_lh_to_hh(hh_feat, lh_feat)
        hh_combined = self.hf_conv(torch.cat([hh_enh_v, hh_enh_h], dim=1))

        # 膨胀残差块恢复
        hl_out = self.dilated_res_hl(hl_feat)
        lh_out = self.dilated_res_lh(lh_feat)
        hh_out = self.dilated_res_hh(hh_combined)

        # 高频输出
        hf_enh = self.hf_output(torch.cat([hl_out, lh_out, hh_out], dim=1))

        # ----------------
        # 3. 下采样低频
        x_down = self.downsample(x)  # 下采样输入 x
        # 对齐通道
        # if x_down.shape[1] != hf_enh.shape[1]:
        #     x_down = nn.Conv2d(x_down.shape[1], hf_enh.shape[1], 1)(x_down)

        # ----------------
        B, C, H, W = x_down.shape
        # 4. 高频引导
        Q_x = self.reshape_heads(self.q_ll(x_down))
        K_hf = self.reshape_heads(self.k_hf(hf_enh))
        V_hf = self.reshape_heads(self.v_hf(hf_enh))

        attn = torch.matmul(Q_x, K_hf.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ll_enh = torch.matmul(attn, V_hf)
        ll_enh = ll_enh.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        x_fused = self.ll_proj(x_down + ll_enh)

        return x_fused

#############################################################################
# @ARCH_REGISTRY.register()
class FAN(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=124,
        dim = 48,
        num_enc_blocks = [1, 2, 2, 4],
        num_dec_blocks=[1, 2, 4, 4],
        num_refinement_blocks = 1,
        channel_heads = [1,2,4,8],
        spatial_heads = [1,2,4,8],
        overlap_ratio=[0.5, 0.5, 0.5, 0.5],
        window_size = 8,
        spatial_dim_head = 16,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        scale = 1
    ):

        super(FAN, self).__init__()
        print("Initializing XRestormer")
        self.scale = scale

        self.conv1 = nn.Conv2d(inp_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = nn.Sequential(  # 直接做 H/4
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1, bias=bias)
        )

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[0])])

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[1])])

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[2])])

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), window_size = window_size, overlap_ratio=overlap_ratio[3],  num_channel_heads=channel_heads[3], num_spatial_heads=spatial_heads[3], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_dec_blocks[3])])

        self.reduce_chan_level3 = nn.Conv2d(int(dim*12), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_dec_blocks[2])])

        self.reduce_chan_level2 = nn.Conv2d(int(dim*6), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_dec_blocks[1])])

        self.reduce_chan_level1 = nn.Conv2d(int(dim * 3), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_dec_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.dic_wave1 = DiC_Wave(dim, dim * 2)
        self.dic_wave2 = DiC_Wave(dim * 2, dim * 4)
        self.dic_wave3 = DiC_Wave(dim * 4, dim * 8)

        self.pfam1 = PFAM(
            hr_channels=int(dim * 4),
            lr_channels=int(dim * 8),
            compressed_channels=(int(dim * 4) + int(dim * 8)) // 8,
        )

        self.pfam2 = PFAM(
            hr_channels=int(dim * 2),
            lr_channels=int(dim * 4),
            compressed_channels=(int(dim * 2) + int(dim * 4)) // 8,

        )
        self.pfam3 = PFAM(
            hr_channels=int(dim * 1),
            lr_channels=int(dim * 2),
            compressed_channels=(int(dim * 1) + int(dim * 2)) // 8,

        )


        self.out_head = nn.Sequential(
            ConvBlock(96, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(256, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inp_img):

        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)

        x = self.conv1(inp_img)  # [bs, 64, 128, 128]
        x = self.conv2(x)  # [bs, 128, 128, 128]
        inp_enc_level1 = self.conv3(x)  # [bs, 48, 64, 64]

        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [B, C1, 64, 64]
        inp_enc_level2 = self.dic_wave1(out_enc_level1)  #

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.dic_wave2(out_enc_level2)  #

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.dic_wave3(out_enc_level3)  #

        latent = self.latent(inp_enc_level4)#[bs, 384, 8, 8]

        hr_feat_out1, lr_feat_out1 = self.pfam1(out_enc_level3, latent)
        inp_dec_level3 = torch.cat([hr_feat_out1, lr_feat_out1], 1)  # [bs, 384, 16, 16]
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # [bs, 192, 16, 16]

        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # [bs, 192, 16, 16]
        hr_feat_out2, lr_feat_out2 = self.pfam2(out_enc_level2, out_dec_level3)
        inp_dec_level2 = torch.cat([hr_feat_out2, lr_feat_out2], 1)  # [bs, 192, 32, 32]
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # [bs, 192, 32, 32]

        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [bs, 96, 32, 32]
        hr_feat_out3, lr_feat_out3 = self.pfam3(out_enc_level1, out_dec_level2)
        inp_dec_level1 = torch.cat([hr_feat_out3, lr_feat_out3], 1)  # [bs, 96, 64, 64]
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # [bs, 96, 64, 64]
        out_dec_level1 = self.refinement(out_dec_level1)  # [bs, 96, 64, 64]

        x = self.out_head(out_dec_level1)  # [B, 256,  H/4, W/4] -> [B,256,64,64]  若输入为256×256
        x = self.out_conv(x)  # [B, 4*out_channels, H/4, W/4]
        out = self.pixel_shuffle(x)  # [B, out_channels, H/2, W/2] -> 对应你示例为 [B,124,128,128]

        return out