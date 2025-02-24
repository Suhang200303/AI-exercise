import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.optim as optim

# 定义图像分块嵌入类，将输入图像分块并进行嵌入
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # 使用卷积操作将输入图像分块并嵌入到低维空间
        x = self.proj(x)
        # 展平每个图像分块的嵌入
        x = x.flatten(2)
        # 将通道维移到倒数第二个维度
        x = x.transpose(1, 2)
        return x

# 定义多头自注意力机制类
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 初始化全连接层，用于计算查询、键和值
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        # 将输入嵌入映射成查询、键和值
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力权重
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        
        # 计算注意力输出
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        return attn_output

# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        # 定义层归一化和多头自注意力机制
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 定义MLP前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 执行自注意力机制和MLP前馈网络
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 定义视觉Transformer模型
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6, num_classes=100, dropout=0.1):
        super(VisionTransformer, self).__init__()
        # 图像分块嵌入层
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # 类别Token，用于图像分类任务
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)])
        
        # 最终的线性分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        # 图像分块嵌入
        x = self.patch_embed(x)
        
        # 在序列维度上连接类别Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码并应用Dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer编码器层
        for layer in self.encoder:
            x = layer(x)
        
        # 应用层归一化并提取类别Token的输出
        x = self.norm(x)
        cls_token_final = x[:, 0]
        # 使用线性层进行分类
        x = self.head(cls_token_final)
        return x
