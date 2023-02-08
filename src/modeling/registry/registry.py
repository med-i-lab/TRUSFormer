# From solo-learn development team.

import copy
from dataclasses import dataclass
import os
from numpy import identity

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple

# from timm.models.convnext import _create_convnext
from timm.models.swin_transformer import _create_swin_transformer
from timm.models.vision_transformer import _create_vision_transformer

from ..resnets import resnet10 as _create_resnet10
from ..resnets import resnet18 as _create_resnet18
from ..resnets import resnet50 as _create_resnet50
from ..resnets import resnet_feature_extractor

from ..attention import MultiheadAttention, AttentionMIL
from ..simple_dense_net import SimpleAggregNet


_MODELS = {}


def register_model(factory):
    _MODELS[factory.__name__] = factory

    return factory


def create_model(model_name, **kwargs):
    if model_name not in _MODELS:
        raise ValueError(f"Model <{model_name}> not registered.")

    return _MODELS[model_name](**kwargs)


def list_models():
    return list(_MODELS.keys())


@register_model
def swin_tiny(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer("swin_tiny_patch4_window7_224", **model_kwargs)


@register_model
def swin_small(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_small_patch4_window7_224", pretrained=False, **model_kwargs
    )


@register_model
def swin_base(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_base_patch4_window7_224", pretrained=False, **model_kwargs
    )


@register_model
def swin_large(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_large_patch4_window7_224", pretrained=False, **model_kwargs
    )


@register_model
def vit_tiny(patch_size=16, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=0,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_tiny_patch16_224", pretrained=False, **model_kwargs
    )
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=0,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=False, **model_kwargs
    )
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=0,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=False, **model_kwargs
    )
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=0,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=False, **model_kwargs
    )
    return model


# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Taken from https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py
# and slightly adapted


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 0,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "poolformer_s": _cfg(crop_pct=0.9),
    "poolformer_m": _cfg(crop_pct=0.95),
}


class PatchEmbed(nn.Module):
    """Patch Embedding that is implemented by a layer of conv.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        patch_size=16,
        stride=16,
        padding=0,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """LayerNorm only for Channel Dimension.

    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(
            -1
        ).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """Group Normalization with 1 group.

    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """Implementation of pooling for PoolFormer.

    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """Implementation of MLP with 1*1 convolutions.

    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """Implementation of one PoolFormer block.

    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=GroupNorm,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(
    dim,
    index,
    layers,
    pool_size=3,
    mlp_ratio=4.0,
    act_layer=nn.GELU,
    norm_layer=GroupNorm,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            PoolFormerBlock(
                dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):
    """PoolFormer, the main class of our model.

    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalizaiotn and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_faat: whetehr output features of the 4 stages, for dense prediction
    --init_cfg，--pretrained:
        for mmdetection and mmsegmentation to load pretrianfed weights
    """

    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        pool_size=3,
        norm_layer=GroupNorm,
        act_layer=nn.GELU,
        num_classes=1000,
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        **kwargs,
    ):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.num_features = embed_dims[-1]

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0],
        )

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`.

                    The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = (
                nn.Linear(embed_dims[-1], num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


@register_model
def poolformer_s12(**kwargs):
    """PoolFormer-S12 model, Params: 12M.

    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers,
        num_classes=0,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["poolformer_s"]
    return model


@register_model
def poolformer_s24(**kwargs):
    """PoolFormer-S24 model, Params: 21M."""
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers,
        num_classes=0,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["poolformer_s"]
    return model


@register_model
def poolformer_s36(**kwargs):
    """PoolFormer-S36 model, Params: 31M."""
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers,
        num_classes=0,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["poolformer_s"]
    return model


@register_model
def poolformer_m36(**kwargs):
    """PoolFormer-M36 model, Params: 56M."""
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers,
        num_classes=0,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["poolformer_m"]
    return model


@register_model
def poolformer_m48(**kwargs):
    """PoolFormer-M48 model, Params: 73M."""
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers,
        num_classes=0,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["poolformer_m"]
    return model


@register_model
def resnet10(**kwargs):
    model_args = dict(**kwargs)
    model = _create_resnet10(num_classes=2, in_channels=1, **model_args)
    return model


@register_model
def resnet10_feature_extractor():
    return resnet_feature_extractor(resnet10())


@register_model
def resnet18(**kwargs):
    model_args = dict(**kwargs)
    model = _create_resnet18(num_classes=2, in_channels=1, **model_args)
    return model


@register_model
def resnet18_feature_extractor():
    return resnet_feature_extractor(resnet18())


@register_model
def resnet18_imagenet():
    resnet_18 = _create_resnet18(pretrained=True)

    new_linear_layer = torch.nn.Linear(resnet_18.fc.in_features, 2)
    resnet_18.fc = new_linear_layer

    # resnet 18 takes 3 channels but our data is 1 channel
    class ChannelRepeater(torch.nn.Module):
        def forward(self, X: torch.Tensor):
            return X.repeat(1, 3, 1, 1)

    return torch.nn.Sequential(ChannelRepeater(), resnet_18)


@register_model
def resnet18_imagenet_feature_extractor():
    return resnet_feature_extractor(resnet18_imagenet())


@register_model
def resnet50(**kwargs):
    model_args = dict(**kwargs)
    model = _create_resnet50(num_classes=2, in_channels=1, **model_args)
    return model


@register_model
def resnet50_imagenet_feature_extractor():
    return resnet_feature_extractor(resnet50())


@register_model
def resnet10_feat_dim_256():
    from ..resnets import resnet10_custom

    return resnet10_custom(
        in_channels=1, n_classes=2, layer_channels=[32, 64, 128, 256], drop_rate="none"
    )


@register_model
def resnet10_feat_dim_128():
    from ..resnets import resnet10_custom

    return resnet10_custom(
        in_channels=1, n_classes=2, layer_channels=[16, 32, 64, 128], drop_rate="none"
    )


@register_model
def resnet10_feat_dim_64():
    from ..resnets import resnet10_custom

    resnet = resnet10_custom(
        in_channels=1, n_classes=2, layer_channels=[8, 16, 32, 64], drop_rate="none"
    )

    return resnet_feature_extractor(resnet)


def resnet10_compressed(compress_to: int = 64):
    resnet_base = resnet10_feature_extractor()
    model = torch.nn.Sequential(
        resnet_base, torch.nn.Linear(resnet_base.features_dim, compress_to)
    )

    model.get_features = model.__call__
    model.features_dim = compress_to

    return model


@register_model
def resnet10_tiny_compressed_to_3dim():

    model = resnet10_feat_dim_64()
    model.fc = torch.nn.Identity()
    model = torch.nn.Sequential(model, torch.nn.Linear(64, 3), torch.nn.ReLU())
    model.num_features = 3

    return model


@register_model
def resnet10_compressed_to_ndim(**kwargs):
    n = 128
    model = resnet10(**kwargs)
    model.fc = torch.nn.Identity()
    model = torch.nn.Sequential(
        model,
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, n),
        torch.nn.ReLU(),
    )
    model.num_features = n

    return model


@register_model
def resnet10_compressed_to_64dim():
    n = 64
    model = resnet10()
    model.fc = torch.nn.Identity()
    model = torch.nn.Sequential(
        model,
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, n),
        torch.nn.ReLU(),
    )
    model.num_features = n

    return model


@register_model
def attention_classifier(token_dim, num_classes, **kwargs):
    qk_dim = 64
    v_dim = 128
    num_heads = 5
    attention_model = MultiheadAttention(
        token_dim, qk_dim, v_dim, num_heads, num_classes, **kwargs
    )

    # model = torch.nn.Sequential(
    #     attention_model,
    #     torch.nn.Linear(v_dim * num_heads, v_dim), torch.nn.ReLU(),
    #     torch.nn.Linear(v_dim, num_classes), torch.nn.ReLU()
    # )
    attention_model.out_dim = v_dim * num_heads

    return attention_model


@register_model
def attention_MIL(token_dim, num_classes):
    attentionMIL_model = AttentionMIL(token_dim, num_classes)
    return attentionMIL_model


@register_model
def linear_aggregation(token_dim, num_classes):
    dense_model = SimpleAggregNet(token_dim=token_dim, num_classes=num_classes)
    return dense_model


def register_configs():
    for model in list_models():

        @dataclass
        class ModelConfig:
            model_name: str = model
            _target_: str = "src.modeling.create_model"

        from hydra.core.config_store import ConfigStore

        ConfigStore.instance().store(model, ModelConfig, "model_registry")


@register_model
def vicreg_resnet_10_crops_split_seed_2():

    # TRAIN_VAL_SPLIT_SEED = 2

    from src.utils.checkpoints import get_named_checkpoint

    # resnet 18 is a typo
    ckpt_path = get_named_checkpoint("resnet_18_vicreg_crops_ssl_0")

    # Hacky workaround
    kwargs = {
        "backbone_name": "resnet10_feature_extractor",
        "batch_size": 1,
        "proj_output_dim": 512,
        "proj_hidden_dim": 512,
        "sim_loss_weight": 25.0,
        "var_loss_weight": 25.0,
        "cov_loss_weight": 1.0,
        "num_epochs": 200,
        "opt_cfg": {
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "nesterov": False,
            "gamma": 0.1,
            "optim_algo": "Adam",
            "extra_opt_args": {},
            "lars_options": None,
            "scheduler_options": {
                "warmup_epochs": 10,
                "warmup_start_lr": 0.0,
                "min_lr": 0.0,
                "scheduler_interval": "step",
                "final_lr": 0.0,
                "decay_epochs": [60, 80],
                "scheduler_type": "warmup_cosine",
                "max_epochs": None,
            },
        },
    }

    from src.lightning.lightning_modules.self_supervised.vicreg import VICReg

    model = VICReg(**kwargs)

    from torch.nn import Linear
    import torch

    model.linear_layer = Linear(512, 2)

    model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    return model


@register_model
def vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop(split_seed):

    from src.utils.checkpoints import get_named_checkpoint

    ckpt_name = f"ssl-pretrain_all-centers_no-prst_ndl_crop_{split_seed}"
    ckpt_path = get_named_checkpoint(ckpt_name)

    from src.lightning.lightning_modules.self_supervised.vicreg import VICReg

    model = VICReg.load_from_checkpoint(ckpt_path)

    return model


def _seeds_to_core_clf_checkpoint_name(seed, split_seed):
    return f"core_clf_ssd_{split_seed}_gsd_{seed}"


@register_model
def sl_all_centers(seed, split_seed):

    from src.utils.checkpoints import get_named_checkpoint

    ckpt_name = f"sl_all-centers_ssd_{split_seed}_gsd_{seed}"
    ckpt_path = get_named_checkpoint(ckpt_name)

    import torch

    checkpoint = torch.load(ckpt_path)

    class _Model(torch.nn.Module):
        def __init__(self, backbone_name):
            super().__init__()
            self.backbone = create_model(backbone_name)

        def forward(self, x):
            return self.backbone.forward(x)

    model = _Model(checkpoint["hyper_parameters"]["backbone_name"])
    model.load_state_dict(checkpoint["state_dict"])

    return model


@register_model
def finetune_linear_all_centers(seed, split_seed):

    from src.utils.checkpoints import get_named_checkpoint

    ckpt_name = (
        f"finetune_all-centrs_ssl-pretrain_all-centrs_ssd_{split_seed}_gsd_{seed}"
    )
    ckpt_path = get_named_checkpoint(ckpt_name)

    from src.lightning.lightning_modules.self_supervised.finetune import Finetuner
    from omegaconf import OmegaConf
    from dataclasses import dataclass

    @dataclass
    class Backbone:
        _target_: str = "src.modeling.registry.create_model"
        model_name: str = "vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop"
        split_seed: int = 0

    backbone_conf = OmegaConf.structured(Backbone(split_seed=split_seed))
    model = Finetuner.load_from_checkpoint(ckpt_path, backbone=backbone_conf)

    return model


@register_model
def finetune_semisup_all_centers(seed, split_seed):

    from src.utils.checkpoints import get_named_checkpoint

    ckpt_name = f"finetune-semi-sup_all-centrs_ssl-pretrain_all-centrs_ssd_{split_seed}_gsd_{seed}"
    ckpt_path = get_named_checkpoint(ckpt_name)

    from src.lightning.lightning_modules.self_supervised.finetune import Finetuner
    from omegaconf import OmegaConf
    from dataclasses import dataclass

    @dataclass
    class Backbone:
        _target_: str = "src.modeling.registry.create_model"
        model_name: str = "vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop"
        split_seed: int = 0

    backbone_conf = OmegaConf.structured(Backbone(split_seed=split_seed))
    model = Finetuner.load_from_checkpoint(ckpt_path, backbone=backbone_conf)

    return model


@register_model
def grade_linear_all_centers():
    from src.utils.checkpoints import get_named_checkpoint

    ckpt_name = f"grading_Multattn-sl_all-centrs_ssl-pretrain_all-centrs"
    ckpt_path = get_named_checkpoint(ckpt_name)

    from src.lightning.lightning_modules.self_supervised.finetune import (
        CoreGradeFinetuner,
    )
    from omegaconf import OmegaConf
    from dataclasses import dataclass

    @dataclass
    class Backbone:
        _target_: str = "src.modeling.registry.create_model"
        model_name: str = "vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop"
        split_seed: int = 1

    backbone_conf = OmegaConf.structured(Backbone)
    model = CoreGradeFinetuner.load_from_checkpoint(ckpt_path, backbone=backbone_conf)

    return model


def core_classifier(seed, split_seed):
    from src.utils.checkpoints import get_named_checkpoint

    path = get_named_checkpoint(_seeds_to_core_clf_checkpoint_name(seed, split_seed))
    sd = torch.load(path)

    from modeling.registry.registry import create_model

    feat_extractor = create_model("resnet10_feature_extractor")

    feat_sd = {
        k.removeprefix("backbone."): v
        for k, v in sd["feature_extractor"].items()
        if "backbone" in k
    }
    feat_extractor.load_state_dict(feat_sd)

    seq_model_config = {
        "pool_mode": "mean",
        "in_feats": 512,
        "feature_reduction": None,
        "hidden_size": 512,
        "num_layers": 12,
        "num_attn_heads": 8,
        "intermediate_size": 768,
        "patch_dropout": 0.2,
        "inner_dropout": 0.2,
        "use_pos_embeddings": True,
        "grid_shape": (28, 46),
    }

    from src.modeling.seq_models import ExactSeqModel

    seq_model = ExactSeqModel(**seq_model_config)
    seq_model.load_state_dict(sd["sequence_model"])

    linear_layer = torch.nn.Linear(512, 2)
    linear_layer.load_state_dict(sd["linear_layer"])

    class ExactSeqClassifier(torch.nn.Module):
        def __init__(self, feat_extractor, seq_model, linear_layer):
            super().__init__()
            self.feat_extractor = feat_extractor
            self.seq_model = seq_model
            self.linear_layer = linear_layer

        def forward(self, patches, X, Y):
            n_patches, n_chans, h, w = patches.shape

            features = self.feat_extractor(patches)
            out = self.seq_model(features, X, Y, output_attentions=True)
            out["logits"] = self.linear_layer(out["pool_output"])
            return out

    model = ExactSeqClassifier(feat_extractor, seq_model, linear_layer)

    model.eval()

    return model


def core_classifier_no_pos_emb():

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    seq_model_config = dict(
        pool_mode="cls",
        in_feats=512,
        feature_reduction="linear",
        hidden_size=64,
        num_layers=12,
        num_attn_heads=8,
        intermediate_size=64,
        patch_dropout=0.2,
        inner_dropout=0.2,
        use_pos_embeddings=False,
        grid_shape=[28, 46],
    )

    from src.utils.checkpoints import get_named_checkpoint

    path = get_named_checkpoint("core_clf_ssd_2_gsd_0_nopos_emb_core_mixing")
    sd = torch.load(path)

    from modeling.registry.registry import create_model

    feat_extractor = create_model("resnet10_feature_extractor")

    feat_sd = {
        k.removeprefix("backbone."): v
        for k, v in sd["feature_extractor"].items()
        if "backbone" in k
    }
    feat_extractor.load_state_dict(feat_sd)

    from src.modeling.seq_models import ExactSeqModel

    seq_model = ExactSeqModel(**seq_model_config)
    seq_model.load_state_dict(sd["sequence_model"])

    linear_layer = torch.nn.Linear(64, 2)
    linear_layer.load_state_dict(sd["linear_layer"])

    class ExactSeqClassifier(torch.nn.Module):
        def __init__(self, feat_extractor, seq_model, linear_layer):
            super().__init__()
            self.feat_extractor = feat_extractor
            self.seq_model = seq_model
            self.linear_layer = linear_layer

        def forward(self, patches, X, Y):
            n_patches, n_chans, h, w = patches.shape

            features = self.feat_extractor(patches)
            out = self.seq_model(features, X, Y, output_attentions=True)
            out["logits"] = self.linear_layer(out["pool_output"])
            return out

    model = ExactSeqClassifier(feat_extractor, seq_model, linear_layer)

    model.eval()

    return model
