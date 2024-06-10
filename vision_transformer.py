import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

# from ..ops.misc import Conv2dNormActivation, MLP
from .misc import Conv2dNormActivation, MLP
# from ..transforms._presets import ImageClassification, InterpolationMode
from ._presets import ImageClassification, InterpolationMode
# from ..utils import _log_api_usage_once
from .utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

"""
vision transformer网络框架：Linear Projection (Patch + Position 的Embedding层)、Transformer Encoder和MLP Head（分类层）三大部分组成
"""
class MLPBlock(MLP):
    """Transformer MLP block.
     in_dim（输入特征的维度）、mlp_dim（隐藏层的维度）和dropout（dropout概率，用于防止过拟合）
     """

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)
        # 遍历检查模块是否为全连接层，若是则对它进行初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   #权重初始化方法Xavier,保持每层激活的方差在反向传播时大致相同，有助于避免梯度消失或爆炸的问题
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  #偏置的值将从均值为0、标准差为1e-6的正态分布中随机抽取。

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        version = local_metadata.get("version", None)

        if version is None or version < 2:  #保持模型的兼容性
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053

            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)
        """
        正确加载状态字典，它处理了参数的匹配、缺失和多余的键，以及在严格模式下的错误检查
        """
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """
    Transformer encoder block.
    初始化EncoderBlock类的实例。

    参数:
    - num_heads: int, 多头自注意力机制中头的数量。
    - hidden_dim: int, 隐藏层的维度。
    - mlp_dim: int, MLP块中的中间维度。
    - dropout: float, 在自注意力和MLP块之后应用的dropout比率。
    - attention_dropout: float, 仅在自注意力块之后应用的dropout比率。
    - norm_layer: Callable, 创建层规范化（LayerNorm）的函数，默认使用nn.LayerNorm，epsilon设为1e-6。
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block 注意力模块
        self.ln_1 = norm_layer(hidden_dim)  # 层规范化层1，用于self-attention块之前的输入
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)  # 多头自注意力模块
        self.dropout = nn.Dropout(dropout)  # dropout层，用于减少过拟合

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)  # 层规范化层2，用于MLP块之前的输入
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)  # 多层感知机块


    def forward(self, input: torch.Tensor):
        # 输入应为三维tensor   (batch_size, seq_length, hidden_dim)
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # Attention block
        x = self.ln_1(input)  # 通过层归一化层1处理输入
        x, _ = self.self_attention(x, x, x, need_weights=False)  # 多头自注意力，不输出权重
        x = self.dropout(x)  # Dropout 正则化自注意力输出
        x = x + input  # 实现残差连接
        # MLP block
        y = self.ln_2(x)  # 再次通过层归一化处理
        y = self.mlp(y)  # 通过MLP进行非线性变换
        return x + y  # 残差连接的Self_Attention输出与MLP输出相加

class Encoder(nn.Module):
    """用于序列到序列翻译任务的 Transformer 模型encoder。"""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        # 默认的归一化层函数，使用 LayerNorm 并设置 epsilon=1e-6
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()  # 调用父类 nn.Module 的构造函数
        # 这里创建了一个形状为 (1, seq_length, hidden_dim) 的张量，
        # 并用均值为 0，标准差为 0.02 的正态分布进行初始化，然后将其包装为模型的参数（张量变成可以学习的模型参数）

        # nn.Parameter()函数为类型转换函数，将一个不可训练的类型tensor 转化为可以训练的parameter 并绑定到module里面(net.parameter()中
        # 除了这个positional embedding 还有下面的class token
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        # 初始化 Dropout 层，用于在训练过程中随机丢弃一些网络连接，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 使用有序字典存储layer，
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        # 循环创建指定数量的编码器层，并添加到有序字典中，叠加Encoderblock块
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,  # 将归一化层函数传递给编码器块
            )
        # 将有序字典转换为 Sequential 模块，顺序执行每一层
        self.layers = nn.Sequential(layers)
        # 创建最终的归一化层，用于在所有编码器层之后进行输出的归一化处理
        self.ln = norm_layer(hidden_dim)

    # 定义前向传播函数，用于执行encoder的逻辑
    def forward(self, input: torch.Tensor):
        # 确保输入张量的维度正确为三维张量 (batch_size, seq_length, hidden_dim)
        assert input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        # 将位置编码加到输入张量中
        input = input + self.pos_embedding
        # 先应用 Dropout，然后通过所有layers，最后进行最终的归一化处理
        return self.ln(self.layers(self.dropout(input)))

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    # 初始化函数，定义模型参数和层结构
    def __init__(
        self,
        image_size: int,               # 输入图像的大小
        patch_size: int,               # 图像块的大小
        num_layers: int,               # Transformer编码器中的层数
        num_heads: int,                # 多头注意力的头数
        hidden_dim: int,               # 隐藏层的维度
        mlp_dim: int,                  # MLP中间层的维度
        dropout: float = 0.0,          # Dropout比例
        attention_dropout: float = 0.0,# Attention的Dropout比例
        num_classes: int = 1000,       # 输出分类数
        # representation_size: 中间表示的尺寸，如果设置为None，则直接使用隐藏层的维度进行分类。
        # 如果提供一个整数值，则会在隐藏层和分类头之间添加一个线性层和Tanh激活函数。
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                                        # 归一化层
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
                                        # 卷积层的配置
    ):

        super().__init__()

        _log_api_usage_once(self)
        # 断言  输入图像尺寸能够被patch size整除
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        # 存储模型配置参数
        self.image_size = image_size  # 输入图像的尺寸
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim  # Transformer的隐藏层维度
        self.mlp_dim = mlp_dim  # MLP中间层的维度
        self.attention_dropout = attention_dropout  # 注意力机制的Dropout比例
        self.dropout = dropout  # Dropout比例
        self.num_classes = num_classes  # 分类的类别数
        self.representation_size = representation_size  # 中间表示的尺寸
        self.norm_layer = norm_layer  # 归一化层

        #定义(ViT) 模型的初始卷积层
        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3  # 初始通道数为3，对应于RGB图像的三通道
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                # 遍历卷积层配置列表，为每个配置创建一个模块
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,  # 当前层的输入通道数
                        out_channels=conv_stem_layer_config.out_channels,  # 当前层的输出通道数
                        kernel_size=conv_stem_layer_config.kernel_size,  # 卷积核的大小
                        stride=conv_stem_layer_config.stride,  # 卷积的步长
                        norm_layer=conv_stem_layer_config.norm_layer,  # 归一化层
                        activation_layer=conv_stem_layer_config.activation_layer,  # 激活层
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2 #序列长度 =高方向上patch的个数 * 宽上patch个数

        # 也将class token转化为可以学习的parameter
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()

        # 根据是否指定了 representation_size 决定分类头的构建方式，是否需要一个额外的中间表示层
        if representation_size is None:
            # 如果没有提供 representation_size，分类头只包含一个线性层
            # 这个线性层负责将特征从 Transformer 的隐藏层维度映射到类别数维度
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)

        else:
            # 如果提供了 representation_size，分类头会包含两个线性层和一个激活函数层
            # 第一个线性层将特征从隐藏层维度映射到中间表示的维度
            # "pre_logits"  中间特征表示
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()

            # 第二个线性层将中间表示的特征映射到最终的类别数维度
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # 模型初始化过程中权重初始化的部分
        # 确保模型中卷积层和线性层，在训练开始之前具有合适的初始权重

        # 检查conv_proj是否是一个标准的nn.Conv2d，如果是，则初始化其权重
        if isinstance(self.conv_proj, nn.Conv2d):
            # fan_in输入通道数乘以卷积核的大小
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            # 使用截断的正态分布初始化权重，其标准差为1/sqrt(fan_in)
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            # 如果存在偏置，则初始化为零
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        # 检查conv_proj是否包含一个最后的1x1卷积层，如果是，则初始化其权重
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # 使用标准正态分布初始化权重，其均值为0，标准差为sqrt(2/out_channels)
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            # 如果存在偏置，则初始化为零
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        # 如果heads模块包含名为pre_logits的线性层，初始化这个线性层的权重
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            # 计算pre_logits线性层的fan_in 输入特征数
            fan_in = self.heads.pre_logits.in_features
            # 使用截断的正态分布初始化权重，其标准差为1/sqrt(fan_in)
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            # 初始化偏置为零
            nn.init.zeros_(self.heads.pre_logits.bias)

        # 检查heads对象是否包含一个分类头线性层，如果是，则初始化其权重
        if isinstance(self.heads.head, nn.Linear):
            # 初始化分类头线性层的权重为零
            nn.init.zeros_(self.heads.head.weight)
            # 初始化偏置为零
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理输入图像转换为适合Transformer编码器处理的格式
        param x: Input image tensor with shape (batch_size, channels, height, width).
        return:  处理后的输入张量，形状为 (batch_size, hidden_dim, sequence_length).
        """
        n, c, h, w = x.shape #n数量 c 通道数 h高 w宽
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p #高方向的patch size
        n_w = w // p #宽方向的patch size
        # 对单个图像的处理
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # 重塑为二维序列张量
        x = x.reshape(n, self.hidden_dim, n_h * n_w) # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)

        x = x.permute(0, 2, 1)
        # 调整序列张量的顺序，使其符合Transformer自注意力层期望的输入格式
        # 在这里，通过permute方法将序列张量的顺序调整为 (n, (n_h * n_w), hidden_dim)
        # 自注意力层期望的输入格式为 (N, S, E) 其中N 是批量大小，S 是图像补丁的数量，E 是嵌入维度
        return x

    def forward(self, x: torch.Tensor):

        x = self._process_input(x)
        # 获取批量大小，即输入图像的数量
        n = x.shape[0]

        # 将类 token 扩展到与批量中所有图像相同的大小，并添加到每个图像序列的最前面
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # 提取最开头的CLS“标记”（类 token）作为最终的特征表示
        x = x[:, 0]
        # 通过分类头 获得最终的分类结果
        x = self.heads(x)

        return x


def _vision_transformer(
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> VisionTransformer:
    """
    创建并配置一个 Vision Transformer 模型的函数。
    当需要使用预训练权重时，快速创建和配置模型实例

    参数:
    - patch_size (int): 输入图像分割成小块（patches）时每块的大小。
    - num_layers (int): 模型中的 Transformer 层数。
    - num_heads (int): 每个 Transformer 层中使用的注意力头数。
    - hidden_dim (int): 模型隐藏层的维度。
    - mlp_dim (int): 多层感知机（MLP）中的维度。
    - weights (WeightsEnum): 可选的权重枚举，用于加载预训练的权重。
    - progress (bool): 是否在加载权重时显示进度条。
    - kwargs (Any): 其他关键字参数，用于传递给模型构造函数。

    返回:
    - VisionTransformer: 配置好的 Vision Transformer 模型实例。
    """
    # 如果提供了权重，根据权重的元数据更新模型参数
    if weights is not None:
        # 更新 kwargs 中的 num_classes 参数，使其与权重元数据中的类别数匹配
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        # 确保权重元数据中的最小尺寸是正方形的
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        # 更新 kwargs 中的 image_size 参数，使其与权重元数据中的尺寸匹配
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])

    # 设置默认的图像尺寸，如果 kwargs 中没有指定 image_size
    image_size = kwargs.pop("image_size", 224)

    # 创建 VisionTransformer 模型实例
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    # 加载预训练的权重到模型中
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))


    return model

_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}

"""
ViT-B (base size)模型适用于需要平衡精度和计算资源的场景。
ViT-L (large size)模型提供更高的精度，但需要更多的计算资源。
ViT-H (Huge size)模型则在有大量计算资源可用时提供最好的性能
16指输入图像被分割成 16x16 像素的小块 
"""


class ViT_B_16_Weights(WeightsEnum):
    """
    类定义了 ViT B/16 模型的不同预训练权重配置。
    提供了三种不同的预训练权重配置,每种配置提供了模型的元数据、性能指标和下载链接
    """
    IMAGENET1K_V1 = Weights(
        # 指定模型输入图像的预处理方式
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,# 模型参数的数量
            "min_size": (224, 224), # 模型接受的最小图像尺寸
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                # 模型在 ImageNet-1K 数据集上的性能指标
                "ImageNet-1K": {
                    "acc@1": 81.072, # top-1 准确率
                    "acc@5": 95.318, # top-5 准确率
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        } #元数据
    )
    # 权重通过端到端微调SWAG模型得到
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_ops": 55.484,
            "_file_size": 331.398,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    # 使用SWAG方法训练得到，在此基础上学习了一个线性分类器。
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224, # 输入图像的裁剪大小
            resize_size=224, # 输入图像的调整大小。
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),# 模型接受的最小图像尺寸
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1  # 默认使用的权重配置


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88224232,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                }
            },
            "_ops": 4.409,
            "_file_size": 336.604,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=242),
        meta={
            **_COMMON_META,
            "num_params": 304326632,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        transforms=partial(
            ImageClassification,
            crop_size=512,
            resize_size=512,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 305174504,
            "min_size": (512, 512),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                }
            },
            "_ops": 361.986,
            "_file_size": 1164.258,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 304326632,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_32-c7638314.pth",

        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 306535400,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                }
            },
            "_ops": 15.378,
            "_file_size": 1169.449,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        # transforms 定义了图像预处理的方式，包括将图像裁剪和调整至 518x518 像素，使用双三次插值
        transforms=partial(
            ImageClassification,
            crop_size=518,
            resize_size=518,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 633470440,
            "min_size": (518, 518),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                }
            },
            "_ops": 1016.717,
            "_file_size": 2416.643,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 632045800,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                }
            },
            "_ops": 167.295,
            "_file_size": 2411.209,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_SWAG_E2E_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
# 指定模型预训练权重的名称
@handle_legacy_interface(weights=("pretrained", ViT_B_32_Weights.IMAGENET1K_V1))
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    基类 用于构建Vision Transformer模型
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    #verify方法来验证传入的weights参数，用于确保传入的权重是有效的，并且符合模型的预期格式
    weights = ViT_B_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=12,# Transformer模型的层数
        num_heads=12,
        hidden_dim=768,# 隐藏层的维度
        mlp_dim=3072,# MLP（多层感知机）的维度
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_16_Weights.IMAGENET1K_V1))
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    weights = ViT_L_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_32_Weights.IMAGENET1K_V1))
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    weights = ViT_L_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    weights = ViT_H_14_Weights.verify(weights)

    return _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """
    interpolate_embeddings 函数的主要作用是在图像处理任务中帮助将预训练的 Transformer 模型适配到不同分辨率的图像上
    用于调整预训练模型的位置编码（positional embeddings），以适应不同分辨率的图像
    在用高分辨率（high resolution）图像做微调时，
    论文里说：保持patch size不变，直接把position embedding向量进行插值处理
    不会影响 Transformer Encoder学习的矩阵
    This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.
    参数:
    image_size: 新模型的图像尺寸。
    patch_size: 新模型的图像块尺寸。
    model_state: 预训练模型的状态字典，包含模型的所有参数。
    interpolation_mode: 用于上采样的算法，默认为 bicubic（双三次插值）。
    reset_heads: 是否重置分类头的状态，默认为 False


    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # 从模型状态字典中获取位置编码
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    # 检查位置编码的形状是否正确
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")
    # 计算新模型的序列长度
    new_seq_length = (image_size // patch_size) ** 2 + 1

    # 序列长度有变化
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        # 分离类 token 编码和图像特征编码
        seq_length -= 1  # 减去类 token
        new_seq_length -= 1  # 新序列长度也减去1
        pos_embedding_token = pos_embedding[:, :1, :]  # 类 token 编码
        pos_embedding_img = pos_embedding[:, 1:, :]  # 图像特征编码
        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        # 调整图像特征编码的形状，进行插值
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        # 检查序列长度是否为完全平方数
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        # 将图像特征编码重塑为二维网格形状
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)

        # 计算新序列长度的平方根
        new_seq_length_1d = image_size // patch_size

        # 对位置编码进行插值
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )
        # 将插值后的编码reshape回一维序列形状
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
        # 调整插值编码的形状，与类 token 编码拼接
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)

        # 将类 token 编码和插值后的图像特征编码拼接起来
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        # 在模型状态字典中更新位置编码
        model_state["encoder.pos_embedding"] = new_pos_embedding

        # 如果 reset_heads 为 True，则不复制分类头的状态
        if reset_heads:
            model_state_copy = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

        # 返回更新后的状态字典
    return model_state
