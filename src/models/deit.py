# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import numpy as np
from mindspore import Parameter, nn, ops, Tensor
from mindspore import dtype as mstype
from mindspore import context

from src.models.initializer import trunc_normal_
from src.models.layers.identity import Identity
from src.models.vision_transformer import VisionTransformer

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 'deit_small_patch16_64', 'deit_small_distilled_patch16_64'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = Parameter(Tensor(np.zeros([1, 1, self.embed_dim]), dtype=mstype.float32))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = Parameter(Tensor(np.zeros([1, num_patches + 2, self.embed_dim]), dtype=mstype.float32))
        self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if self.num_classes > 0 else Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.init_weights()

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        cls_tokens = ops.Tile()(self.cls_token, (x.shape[0], 1, 1))  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = ops.Tile()(self.dist_token, (x.shape[0], 1, 1))
        x = ops.Concat(1)((cls_tokens, dist_token, x))

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return x, x_dist


def deit_small_distilled_patch16_64(**kwargs):
    network = DistilledVisionTransformer(
        img_size=64, patch_size=16, embed_dim=64, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_small_patch16_64(**kwargs):
    network = VisionTransformer(
        img_size=64, patch_size=16, embed_dim=64, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network

def deit_tiny_patch16_224(**kwargs):
    network = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_small_patch16_224(**kwargs):
    network = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_base_patch16_224(**kwargs):
    network = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_tiny_distilled_patch16_224(**kwargs):
    network = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_small_distilled_patch16_64(**kwargs):
    network = DistilledVisionTransformer(
        img_size=64, patch_size=16, embed_dim=64, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network

def deit_small_distilled_patch16_224(**kwargs):
    network = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_base_distilled_patch16_224(**kwargs):
    network = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_base_patch16_384(**kwargs):
    network = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


def deit_base_distilled_patch16_384(**kwargs):
    network = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return network


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([2, 3, 224, 224]), dtype=mstype.float32)
    model = deit_tiny_patch16_224()
    out = model(data)
    print(out.shape)
    params = 0.
    for name, param in model.parameters_and_names():
        params += np.prod(param.shape)
        print(name)
    assert params == 5717416
