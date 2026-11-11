import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

class Spacial_Attention_Block(nn.Module):
    def __init__(self, channel):
        super(Spacial_Attention_Block, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1)

        self.avg1 = nn.Conv2d(channel, channel * 2, 1)
        self.avg2 = nn.Conv2d(channel, channel * 2, 1)
        self.max1 = nn.Conv2d(channel, channel * 2, 1)
        self.max2 = nn.Conv2d(channel, channel * 2, 1)

        self.avg11 = nn.Conv2d(channel * 2, channel, 1)
        self.avg22 = nn.Conv2d(channel * 2, channel, 1)
        self.max11 = nn.Conv2d(channel * 2, channel, 1)
        self.max22 = nn.Conv2d(channel * 2, channel, 1)

        self.proj_out = nn.Conv2d(channel, channel, 1)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()
        # 保存原始四维特征，后续应用空间注意力时使用
        f1_orig = f1
        f2_orig = f2

        # 展平用于计算通道注意力
        f1_flat = f1.reshape(b, c, -1)   # [B, C, H*W]
        f2_flat = f2.reshape(b, c, -1)   # [B, C, H*W]

        # ----- 通道注意力权重 -----
        avg_1 = torch.mean(f1_flat, dim=-1, keepdim=True).unsqueeze(-1)  # [B, C, 1, 1]
        max_1, _ = torch.max(f1_flat, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)                                      # [B, C, 1, 1]

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)   # [B, C, 1]
        max_1 = self.max11(max_1).squeeze(-1)   # [B, C, 1]
        channel_attn1 = avg_1 + max_1           # [B, C, 1]

        avg_2 = torch.mean(f2_flat, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2_flat, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        channel_attn2 = avg_2 + max_2           # [B, C, 1]

        # ----- 交叉注意力加权 -----
        cross = torch.matmul(channel_attn1, channel_attn2.transpose(1, 2))  # [B, C, C]
        f1_cross = torch.matmul(F.softmax(cross, dim=-1), f1_flat)          # [B, C, H*W]
        f2_cross = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2_flat)  # [B, C, H*W]

        # 恢复为四维特征图
        f1_cross = f1_cross.reshape(b, c, h, w)   # [B, C, H, W]
        f2_cross = f2_cross.reshape(b, c, h, w)   # [B, C, H, W]

        # ----- 空间注意力图 -----
        # 对 f1_cross 计算空间注意力
        avg_out1 = torch.mean(f1_cross, dim=1, keepdim=True)      # [B, 1, H, W]
        max_out1, _ = torch.max(f1_cross, dim=1, keepdim=True)    # [B, 1, H, W]
        spatial_input1 = torch.cat([avg_out1, max_out1], dim=1)   # [B, 2, H, W]
        spatial_attn1 = F.relu(self.conv1_spatial(spatial_input1))
        spatial_attn1 = torch.sigmoid(self.conv2_spatial(spatial_attn1))  # [B, 1, H, W]

        # 对 f2_cross 计算空间注意力
        avg_out2 = torch.mean(f2_cross, dim=1, keepdim=True)
        max_out2, _ = torch.max(f2_cross, dim=1, keepdim=True)
        spatial_input2 = torch.cat([avg_out2, max_out2], dim=1)
        spatial_attn2 = F.relu(self.conv1_spatial(spatial_input2))
        spatial_attn2 = torch.sigmoid(self.conv2_spatial(spatial_attn2))  # [B, 1, H, W]

        # ----- 应用空间注意力（残差连接）到原始特征上 -----
        f1_enhanced = f1_orig * spatial_attn1 + f1_orig   # [B, C, H, W]
        f2_enhanced = f2_orig * spatial_attn2 + f2_orig   # [B, C, H, W]

        # ----- 最终融合输出 -----
        out = self.proj_out(F.gelu(f1_enhanced) * f2_enhanced)  # [B, C, H, W]
        return out

if __name__ == '__main__':
    channel = 128
    model = Spacial_Attention_Block(channel=channel)
    model.eval()
    x = torch.randn(4, channel, 64, 64)
    y = torch.randn(4, channel, 64, 64)

    with torch.no_grad():
        start = timeit.default_timer()
        out = model(x, y)
        end = timeit.default_timer()

    elapsed = end - start
    print(f"输出 out 的形状：{out.shape}")
    print(f"前向传播运行时间：{elapsed:.4f} 秒")