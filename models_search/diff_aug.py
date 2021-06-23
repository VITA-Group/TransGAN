# Code Heavily borrowed from Differentiable Augmentation and StyleGAN-ADA

import torch
import torch.nn.functional as F
import numpy as np
import random
from models_search.ada import *
# from models_search import conv2d_gradfix
import scipy.signal
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from torch_utils.ops import grid_sample_gradfix
from torch_utils.ops import conv2d_gradfix
    
wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

# global Hz_fbank
# Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
# Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
# Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
# Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
# Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
# for i in range(1, Hz_fbank.shape[0]):
#     Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
#     Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
#     Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
# Hz_fbank = torch.as_tensor(Hz_fbank, dtype=torch.float32)


def DiffAugment(x, policy='', channels_first=True, affine=None):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, affine=affine)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_crop(x, affine=None):
    b, _, h, w = x.shape
    x_large = torch.nn.functional.interpolate(x, scale_factor=1.2, mode='bicubic')
    _, _, h_large, w_large = x_large.size()
    h_start, w_start = random.randint(0, (h_large - h)), random.randint(0, (w_large - w))
    x_crop = x_large[:, :, h_start:h_start+h, w_start:w_start+w]
    assert x_crop.size() == x.size()
    output = torch.where(torch.rand([b, 1, 1, 1], device=x.device) < 0.2, x_crop, x)
    return output
    

def rand_filter(images, affine=None):
    ratio = 0.25
   
    
    _, Hz_fbank = affine
    Hz_fbank = Hz_fbank.to(images.device)
    imgfilter_bands = [1,1,1,1]
    batch_size, num_channels, height, width = images.shape
    device = images.device
    num_bands = Hz_fbank.shape[0]
    assert len([1,1,1,1]) == num_bands
    expected_power = constant(np.array([10, 1, 1, 1]) / 13, device=device) # Expected power spectrum (1/f).

    # Apply amplification for each band with probability (imgfilter * strength * band_strength).
    g = torch.ones([batch_size, num_bands], device=device) # Global gain vector (identity).
    for i, band_strength in enumerate(imgfilter_bands):
        t_i = torch.exp2(torch.randn([batch_size], device=device) * 1)
        t_i = torch.where(torch.rand([batch_size], device=device) < ratio * band_strength, t_i, torch.ones_like(t_i))
#         if debug_percentile is not None:
#             t_i = torch.full_like(t_i, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * 1)) if band_strength > 0 else torch.ones_like(t_i)
        t = torch.ones([batch_size, num_bands], device=device)                  # Temporary gain vector.
        t[:, i] = t_i                                                           # Replace i'th element.
        t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt() # Normalize power.
        g = g * t                                                               # Accumulate into global gain.

    # Construct combined amplification filter.
    Hz_prime = g @ Hz_fbank                                    # [batch, tap]
    Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])   # [batch, channels, tap]
    Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1]) # [batch * channels, 1, tap]

    # Apply filter.
    p = Hz_fbank.shape[1] // 2
    images = images.reshape([1, batch_size * num_channels, height, width])
    images = torch.nn.functional.pad(input=images, pad=[p,p,p,p], mode='reflect')
    images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(2), groups=batch_size*num_channels)
    images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(3), groups=batch_size*num_channels)
    images = images.reshape([batch_size, num_channels, height, width])
    return images
            
def rand_hue(images, affine=None):
    batch_size, num_channels, height, width = images.shape
    device = images.device
    I_4 = torch.eye(4, device=device)
    C = I_4
    v = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device) # Luma axis.

    # Apply hue rotation with probability (hue * strength).
    if num_channels > 1:
        theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * 1
        theta = torch.where(torch.rand([batch_size], device=device) < 0.5, theta, torch.zeros_like(theta))
#         if debug_percentile is not None:
#             theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * 1)
        C = rotate3d(v, theta) @ C # Rotate around v.

    # Apply saturation with probability (saturation * strength).
#     if self.saturation > 0 and num_channels > 1:
#         s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
#         s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s, torch.ones_like(s))
#         if debug_percentile is not None:
#             s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.saturation_std))
#         C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

    # ------------------------------
    # Execute color transformations.
    # ------------------------------

    # Execute if the transform is not identity.
    if C is not I_4:
        images = images.reshape([batch_size, num_channels, height * width])
        if num_channels == 3:
            images = C[:, :3, :3] @ images + C[:, :3, 3:]
        elif num_channels == 1:
            C = C[:, :3, :].mean(dim=1, keepdims=True)
            images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
        else:
            raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
        images = images.reshape([batch_size, num_channels, height, width])
    return images

def rand_geo(images, affine=None):
    batch_size, num_channels, height, width = images.shape
    device = images.device
    
    Hz_geom, _ = affine
    Hz_geom = Hz_geom.to(images.device)
    
    I_3 = torch.eye(3, device=device)
    G_inv = I_3

    # Apply x-flip with probability (xflip * strength).
    if 1:
        i = torch.floor(torch.rand([batch_size], device=device) * 2)
        i = torch.where(torch.rand([batch_size], device=device) < 1, i, torch.zeros_like(i))
#         if debug_percentile is not None:
#             i = torch.full_like(i, torch.floor(debug_percentile * 2))
        G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

#     # Apply 90 degree rotations with probability (rotate90 * strength).
#     if self.rotate90 > 0:
#         i = torch.floor(torch.rand([batch_size], device=device) * 4)
#         i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * P, i, torch.zeros_like(i))
#         if debug_percentile is not None:
#             i = torch.full_like(i, torch.floor(debug_percentile * 4))
#         G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

    # Apply integer translation with probability (xint * strength).
#     if self.xint > 0:
#         t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
#         t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * P, t, torch.zeros_like(t))
#         if debug_percentile is not None:
#             t = torch.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
#         G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * width), torch.round(t[:,1] * height))

    # --------------------------------------------------------
    # Select parameters for general geometric transformations.
    # --------------------------------------------------------

    # Apply isotropic scaling with probability (scale * strength).
    if 1:
        s = torch.exp2(torch.randn([batch_size], device=device) * 0.2)
        s = torch.where(torch.rand([batch_size], device=device) < 0.3, s, torch.ones_like(s))
#         if debug_percentile is not None:
#             s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.scale_std))
        G_inv = G_inv @ scale2d_inv(s, s)

#     # Apply pre-rotation with probability p_rot.
#     p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1)) # P(pre OR post) = p
#     if self.rotate > 0:
#         theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
#         theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
#         if debug_percentile is not None:
#             theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
#         G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.

#     Apply anisotropic scaling with probability (aniso * strength).
    if 1:
        s = torch.exp2(torch.randn([batch_size], device=device) * 0.2)
        s = torch.where(torch.rand([batch_size], device=device) < 0.3, s, torch.ones_like(s))
#         if debug_percentile is not None:
#             s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
        G_inv = G_inv @ scale2d_inv(s, 1 / s)

#     # Apply post-rotation with probability p_rot.
#     if self.rotate > 0:
#         theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
#         theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
#         if debug_percentile is not None:
#             theta = torch.zeros_like(theta)
#         G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.

    # Apply fractional translation with probability (xfrac * strength).
    if 1:
        t = torch.randn([batch_size, 2], device=device) * 0.125
        t = torch.where(torch.rand([batch_size, 1], device=device) < 0.3, t, torch.zeros_like(t))
#         if debug_percentile is not None:
#             t = torch.full_like(t, torch.erfinv(debug_percentile * 2 - 1) * 0.125)
        G_inv = G_inv @ translate2d_inv(t[:,0] * width, t[:,1] * height)

    # ----------------------------------
    # Execute geometric transformations.
    # ----------------------------------

    # Execute if the transform is not identity.
    if G_inv is not I_3:

        # Calculate padding.
        cx = (width - 1) / 2
        cy = (height - 1) / 2
        cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device) # [idx, xyz]
        cp = G_inv @ cp.t() # [batch, xyz, idx]
        Hz_pad = Hz_geom.shape[0] // 4
        margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
        margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
        margin = margin + constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
        margin = margin.max(constant([0, 0] * 2, device=device))
        margin = margin.min(constant([width-1, height-1] * 2, device=device))
        mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

        # Pad image and adjust origin.
        images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
        G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

        # Upsample.
        images = upfirdn2d.upsample2d(x=images, f=Hz_geom, up=2)
        G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
        G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

        # Execute transformation.
        shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
        G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
        grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
        images = grid_sample_gradfix.grid_sample(images, grid)

        # Downsample and crop.
        images = upfirdn2d.downsample2d(x=images, f=Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
    return images




def rand_brightness(x, affine=None):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x, affine=None):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x, affine=None):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.2, affine=None):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_translation_1(x, ratio=0.1, affine=None):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_x = translation_x*2 - 1
    translation_y = translation_y*2 - 1
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_strong_translation(x, ratio=0.125, affine=None):
    ratio = 0.125
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5, affine=None):
    if random.random() < 0.3:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
    return x

def rand_erase(x, ratio=0.5, affine=None):
    ratio_x = random.randint(20, x.size(2)//2 + 20)
    ratio_y = random.randint(20, x.size(3)//2 + 20)
    if random.random() < 0.3:
#         cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
        cutout_size = ratio_x, ratio_y
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
    return x

def rand_erase_ratio(x, ratio=0.5, affine=None):
    ratio_x = random.randint(int(x.size(2)*0.2), int(x.size(2)*0.7))
    ratio_y = random.randint(int(x.size(3)*0.2), int(x.size(3)*0.7))
    if random.random() < 0.3:
#         cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
        cutout_size = ratio_x, ratio_y
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
    return x

def rand_erase2_ratio(x, ratio=0.5, affine=None):
    ratio_x = random.randint(int(x.size(2)*0.2), int(x.size(2)*0.7))
    ratio_y = random.randint(int(x.size(3)*0.2), int(x.size(3)*0.7))
    if random.random() < 0.3:
#         cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
        cutout_size = ratio_x, ratio_y
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
        
        cutout_size = ratio_x, ratio_y
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
    return x

def rand_rand_erase_ratio(x, ratio=0.5, affine=None):
    ratio_x = random.randint(int(x.size(2)*0.2), int(x.size(2)*0.7))
    ratio_y = random.randint(int(x.size(3)*0.2), int(x.size(3)*0.7))
#     if random.random() < 0.3:
#         cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
    cutout_size = ratio_x, ratio_y
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x[:int(x.size(0)*0.3)] = x[:int(x.size(0)*0.3)] * mask[:int(x.size(0)*0.3)].unsqueeze(1)
    return x

def rand_cutmix(x, affine=None):
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = lam
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
#         if random.random()<0.5:
#             cx = 0
#         else:
#             cx = int(W*0.6)
#         if random.random()<0.5:
#             cy = 0
#         else:
#             cy = int(H*0.6)
            
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx, 0, W)
        bby1 = np.clip(cy, 0, H)
        bbx2 = np.clip(cx + cut_w, 0, W)
        bby2 = np.clip(cy + cut_h, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    lam = 0.45 + 0.1*random.random()
    rand_index = torch.randperm(x.size()[0]).cuda()
#     for i in range(10000):
#         if rand_index[0].item() == 0:
#             rand_index = torch.randperm(x.size()[0]).cuda()
#         else:
#             break
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x

# def rand_erase(x, ratio=0.5):
#     ratio_x = random.randint(20, x.size(2)//2 + 20)
#     ratio_y = random.randint(20, x.size(3)//2 + 20)
#     cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
#     offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
#     offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    
    
#     if random.random() < 0.3:
#         cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
#         offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
#         offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
#         grid_batch, grid_x, grid_y = torch.meshgrid(
#             torch.arange(x.size(0), dtype=torch.long, device=x.device),
#             torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
#             torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
#         )
#         grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
#         grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
#         del offset_x
#         del offset_y
#         mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
#         mask[grid_batch, grid_x, grid_y] = 0
#         x = x * mask.unsqueeze(1)
#         del mask
#         del grid_x
#         del grid_y
#         del grid_batch
#     return x

def rand_rotate(x, ratio=0.5, affine=None):
    k = random.randint(1,3)
    if random.random() < ratio:
        x = torch.rot90(x, k, [2,3])
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'translation_1': [rand_translation_1],
    'strong_translation': [rand_strong_translation],
    'cutout': [rand_cutout],
    'erase': [rand_erase],
    'erase_ratio': [rand_erase_ratio],
    'erase2_ratio': [rand_erase2_ratio],
    'rand_erase_ratio': [rand_rand_erase_ratio],
    'rotate': [rand_rotate],
    'cutmix': [rand_cutmix],
    'hue': [rand_hue],
    'filter': [rand_filter],
    'geo': [rand_geo],
    'crop': [rand_crop],
}
