import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.DWM_Deformable_Transformer import DWM_MSDeformableNeighborhoodAttention


# ---------- 辅助函数 ----------
class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = LayerNorm2d(dim, eps=1e-5)

    def forward(self, x):
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super(FeedForward, self).__init__()

        self.project_in = nn.Conv2d(dim, dim*4, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, padding=1, groups=dim*4, bias=False)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = DWM_MSDeformableNeighborhoodAttention(
            dim,
            num_heads,
            kernel_sizes=[3, 3, 3, 3],
            dilations=[1, 1, 1, 1],
            branch_configs=[
                {'id': 0, 'window_size': (4, 4), 'transpose': False},
                {'id': 1, 'window_size': (4, 4), 'transpose': True},
                {'id': 2, 'window_size': (8, 8), 'transpose': False},
                {'id': 3, 'window_size': (8, 8), 'transpose': True},
            ]
        )
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class RDB_Block(nn.Module):
    def __init__(self, dim, num_heads, num_layers=3):
        super(RDB_Block, self).__init__()
        self.num_layers = num_layers
        self.trans_block = TransformerBlock(dim, num_heads)

        self.conv = nn.ModuleList()
        for i in range(num_layers):
            self.conv.append(
                nn.Conv2d(
                    in_channels=dim * (i + 1),
                    out_channels=dim,
                    kernel_size=3,
                    padding='same',
                    bias=True
                )
            )

        self.proj = nn.Conv2d(
            in_channels=dim * (num_layers + 1),
            out_channels=dim,
            kernel_size=1,
            padding='same',
            bias=True
        )

    def forward(self, x):
        stored_outputs = [x]
        for i in range(self.num_layers):
            local_concat = torch.cat(stored_outputs, dim=1)
            out = self.conv[i](local_concat)
            out = self.trans_block(out)
            stored_outputs.append(out)

        final_concat = torch.cat(stored_outputs, dim=1)
        final_out = self.proj(final_concat)
        final_out = self.trans_block(final_out)

        res_out = x + final_out
        return res_out

if __name__ == '__main__':
    model = RDB_Block(32, 2, 3)
    start = timeit.default_timer()
    model.eval()
    x = torch.randn(4, 32, 64, 64)

    with torch.no_grad():
        start = timeit.default_timer()
        out = model(x)
        end = timeit.default_timer()

    elapsed = end - start
    print(f"输出 out 的形状：{out.shape}")
    print(f"前向传播运行时间：{elapsed:.4f} 秒")