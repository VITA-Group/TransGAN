import torch
import torch.nn as nn
import math
import numpy as np


from models_search.ViT_helper import DropPath, to_2tuple, trunc_normal_
from models_search.diff_aug import DiffAugment

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])

class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu
        
    def forward(self, x):
        return self.act_layer(x)
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size
        if self.window_size != 0:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.window_size != 0:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).clone()].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)
        
    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0,2,1)).permute(0,2,1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=8):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class StageBlock(nn.Module):

    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=8):
        super().__init__()
        self.depth = depth
        self.block = nn.ModuleList([
                        Block(
                        dim=dim, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop, 
                        attn_drop=attn_drop, 
                        drop_path=drop_path, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=window_size
                        ) for i in range(depth)])

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

class Generator(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(Generator, self).__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2, embed_dim//4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2, embed_dim//16))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule
        self.blocks = StageBlock(
                        depth=depth[0],
                        dim=embed_dim, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=args.bottom_width,
                        )
        self.upsample_blocks = nn.ModuleList([
                    StageBlock(
                        depth=depth[1],
                        dim=embed_dim//4, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=args.bottom_width*2,
                        ),
                    StageBlock(
                        depth=depth[2],
                        dim=embed_dim//16, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=args.bottom_width*4,
                        )
                    ])
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//16, 3, 1, 1, 0)
        )
#         self.apply(self._init_weights)
    
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Conv2d) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.BatchNorm1d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.InstanceNorm1d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
            
    def set_arch(self, x, cur_stage):
        pass

    def forward(self, z, epoch):
        if self.args.latent_norm:
            latent_size = z.size(-1)
            z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size()
        H, W = self.bottom_width, self.bottom_width
        x = self.blocks(x)
        for index, blk in enumerate(self.upsample_blocks):
            # x = x.permute(0,2,1)
            # x = x.view(-1, self.embed_dim, H, W)
            x, H, W = pixel_upsample(x, H, W)
            x = x + self.pos_embed[index+1].to(x.get_device())
            x = blk(x)
            # _, _, H, W = x.size()
            # x = x.view(-1, self.embed_dim, H*W)
            # x = x.permute(0,2,1)
        output = self.deconv(x.permute(0, 2, 1).view(-1, self.embed_dim//16, H, W))
        return output


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class DisBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x*self.gain + self.drop_path(self.attn(self.norm1(x)))*self.gain
        x = x*self.gain + self.drop_path(self.mlp(self.norm2(x)))*self.gain
        return x


class Discriminator(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, args, img_size=32, patch_size=None, in_chans=3, num_classes=1, embed_dim=None, depth=7,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = args.df_dim  
        
        depth = args.d_depth
        self.args = args
        patch_size = args.patch_size
        norm_layer = args.d_norm
        act_layer = args.d_act
        mlp_ratio = args.d_mlp
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (args.img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DisBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for i in range(depth)])
        
        self.norm = CustomNorm(norm_layer, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Conv2d) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        if "None" not in self.args.diff_aug:
            x = DiffAugment(x, self.args.diff_aug, True)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).permute(0,2,1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:,0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
