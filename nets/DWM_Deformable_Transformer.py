import math
import warnings

import einops
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
# NATTEN 兼容: 论文实验环境 (Linux, natten==0.17.5) 提供 na2d_qk/na2d_av (支持 rpb);
# natten>=0.20 移除了分离 API, 其融合算子既不支持 rpb, 又缺少 head_dim=4 的后端
# (部分环境还会回退到未经 torch.compile 的 flex_attention, 边界结果不正确)。
# 因此仅在旧 API 可用时使用 NATTEN 加速; 否则自动使用下方数学等价的
# 纯 PyTorch 参考实现 (含 rpb, 与论文训练语义一致)。natten 为可选依赖。
try:
    from natten.functional import na2d_qk, na2d_av
    _NATTEN_OLD_API = True
except ImportError:
    _NATTEN_OLD_API = False


def _na2d_reference(q, k, v, kernel_size, dilation, rpb=None, is_causal=False):
    """natten.na2d 的纯 PyTorch 参考实现 (与 natten<=0.17 的
    na2d_qk/softmax/na2d_av 语义一致, 支持 rpb)。
    输入/输出: (B, heads, H, W, D), is_causal=False, 奇数 kernel_size。"""
    assert kernel_size % 2 == 1, 'reference NA 仅支持奇数 kernel_size'
    assert not is_causal, 'reference NA 不支持 is_causal'
    B, heads, H, W, D = q.shape
    ks, d = kernel_size, dilation
    pad = (ks // 2) * d

    # (B, heads, H*W, D)
    q_flat = q.reshape(B, heads, H * W, D)
    # (B, heads, H, W, D) -> (B, heads*D, H, W) 需先换轴 (D 轴与 heads 轴不相邻)
    kv = lambda t: t.permute(0, 1, 4, 2, 3).reshape(B, heads * D, H, W)
    # unfold 邻域: (B, heads*D, ks*ks, H*W) -> (B, heads, D, H*W, ks*ks)
    k_unf = F.unfold(kv(k), kernel_size=ks,
                     dilation=d, padding=pad).reshape(B, heads, D, ks * ks, H * W).permute(0, 1, 2, 4, 3)
    v_unf = F.unfold(kv(v), kernel_size=ks,
                     dilation=d, padding=pad).reshape(B, heads, D, ks * ks, H * W).permute(0, 1, 2, 4, 3)

    attn = torch.einsum('bhqd,bhdql->bhql', q_flat, k_unf)

    # 遮蔽越界邻居 (zero padding 处), softmax 仅在有效邻域内
    ones = q.new_ones(1, 1, H, W)
    valid = F.unfold(ones, kernel_size=ks, dilation=d, padding=pad)  # (1, ks*ks, H*W)
    attn = attn.masked_fill(valid.permute(0, 2, 1).reshape(1, 1, H * W, ks * ks) == 0, float('-inf'))

    if rpb is not None:
        # rpb: (heads, 2ks-1, 2ks-1)。索引约定逐行复刻 natten==0.17.5 的
        # rel_pos_bias_2d.hpp / natten_cpu_commons.h: index = get_pb_start(pos) + ki,
        # 即相对距离 + ns, 且在图像边界处按 NATTEN 的裁剪规则平移。
        ns = ks // 2
        idx = torch.arange(H, device=q.device)
        pi = torch.full((H,), ns, dtype=torch.long, device=q.device)
        pi[:ns] += ns - torch.arange(ns, device=q.device)
        pi[H - ns:] += (H - 1 - ns) - torch.arange(H - ns, H, device=q.device)
        if d > 1:
            pi = torch.where(idx - ns * d < 0,
                             ks - 1 - torch.div(idx, d, rounding_mode='floor'),
                             torch.where(idx + ns * d >= H,
                                         torch.div(H - 1 - idx, d, rounding_mode='floor'),
                                         pi))
        idy = torch.arange(W, device=q.device)
        pj = torch.full((W,), ns, dtype=torch.long, device=q.device)
        pj[:ns] += ns - torch.arange(ns, device=q.device)
        pj[W - ns:] += (W - 1 - ns) - torch.arange(W - ns, W, device=q.device)
        if d > 1:
            pj = torch.where(idy - ns * d < 0,
                             ks - 1 - torch.div(idy, d, rounding_mode='floor'),
                             torch.where(idy + ns * d >= W,
                                         torch.div(W - 1 - idy, d, rounding_mode='floor'),
                                         pj))
        row = pi.view(H, 1, 1, 1) + torch.arange(ks, device=q.device).view(1, 1, ks, 1)  # (H,1,ks,1)
        col = pj.view(1, W, 1, 1) + torch.arange(ks, device=q.device).view(1, 1, 1, ks)  # (1,W,1,ks)
        rpb_pos = rpb[:, row, col]  # (heads, H, W, ks, ks), rpb_pos[i,j,ki,kj] = rpb[:, pi[i]+ki, pj[j]+kj]
        attn = attn + rpb_pos.reshape(1, heads, H * W, ks * ks)

    attn = attn.softmax(dim=-1)
    out = torch.einsum('bhql,bhdql->bhqd', attn, v_unf)
    return out.reshape(B, heads, H, W, D)


def na2d_compat(q, k, v, kernel_size, dilation, is_causal, rpb):
    """跨版本 NA 前向, 输入/输出 (B, heads, H, W, D)。
    旧版 NATTEN (<=0.17, 提供 na2d_qk/na2d_av): 论文原始路径, 优先使用;
    否则使用 _na2d_reference 纯 PyTorch 实现 (含 rpb, 数学等价)。"""
    if not _NATTEN_OLD_API:
        return _na2d_reference(q.contiguous(), k.contiguous(), v.contiguous(),
                               kernel_size, dilation, rpb=rpb, is_causal=is_causal)
    attn = na2d_qk(q.contiguous(), k.contiguous(),
                   kernel_size=(kernel_size, kernel_size),
                   dilation=dilation, is_causal=is_causal, rpb=rpb)
    attn = attn.softmax(dim=-1)
    attn = nn.Dropout(0)(attn)
    return na2d_av(attn, v.contiguous(),
                   kernel_size=(kernel_size, kernel_size),
                   dilation=dilation, is_causal=is_causal)


# ---------- 辅助函数 ----------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

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


# ---------- 可变形邻域注意力（四分支窗口 + 四分支邻域）----------
class DWM_MSDeformableNeighborhoodAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kernel_sizes: list = [3, 3, 5, 5],
        dilations: list = [1, 2, 1, 2],
        branch_configs: list = [
            {'id': 0, 'window_size': (4, 4), 'transpose': False},
            {'id': 1, 'window_size': (4, 4), 'transpose': True},
            {'id': 2, 'window_size': (8, 8), 'transpose': False},
            {'id': 3, 'window_size': (8, 8), 'transpose': True},
        ],
        offset_range_factor: float = 1.0,
        stride: int = 1,
        no_off: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5

        self.num_dim = dim
        self.num_heads = heads
        self.num_dim_heads = self.num_dim // self.num_heads

        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.stride = stride
        self.is_causal = is_causal
        self.branch_configs = branch_configs

        # ---------- Offset 生成网络 ----------
        self.proj_q_offset = nn.Conv2d(self.num_dim, self.num_dim, kernel_size=1, stride=1, padding=0)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.num_dim_heads,
                self.num_dim_heads,
                kernel_size=kernel_sizes[0],
                stride=self.stride,
                padding=kernel_sizes[0] // 2 if kernel_sizes[0] != stride else 0,
                groups=self.num_dim_heads
            ),
            LayerNorm2d(self.num_dim_heads),
            nn.GELU(),
            nn.Conv2d(self.num_dim_heads, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # ---------- 投影层（先投影再拆分，避免维度不匹配）----------
        # 对采样后的特征做 Q、K、V 投影，用于窗口注意力
        self.proj_q_win = nn.Linear(self.num_dim, self.num_dim, bias=False)
        self.proj_k_win = nn.Linear(self.num_dim, self.num_dim, bias=False)
        self.proj_v_win = nn.Linear(self.num_dim, self.num_dim, bias=False)

        # 邻域注意力的投影（与之前相同）
        self.proj_q = nn.Conv2d(self.num_dim, self.num_dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.num_dim, self.num_dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.num_dim, self.num_dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.num_dim, self.num_dim, kernel_size=1, stride=1, padding=0)

        # ---------- 相对位置偏置管理 ----------
        self.window_rpbs = nn.ParameterDict()   # 窗口分支 RPE
        self.rpbs = nn.ParameterList()          # 邻域分支 RPE
        for ks in kernel_sizes:
            rpb = nn.Parameter(torch.zeros(self.num_heads, (2 * ks - 1), (2 * ks - 1)))
            trunc_normal_(rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
            self.rpbs.append(rpb)

        # LPE 残差
        self.rpe_table = nn.Conv2d(self.num_dim, self.num_dim, kernel_size=3, stride=1, padding=1, groups=self.num_dim)

        self.proj_drop = nn.Dropout(0, inplace=True)
        self.attn_drop = nn.Dropout(0, inplace=True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # window_rpbs 是 forward 时才创建的惰性参数: load_networks (strict=False)
        # 发生在首次 forward 之前, checkpoint 中对应的键会被静默丢弃, 导致推理
        # 用随机初始化的偏置。这里先把未匹配的 window_rpbs 缓存下来,
        # 待 _get_window_rpb 创建同名参数时拷入训练值。
        self._pending_window_rpbs = {}
        for k, v in state_dict.items():
            name = prefix + 'window_rpbs.'
            if k.startswith(name):
                self._pending_window_rpbs[k[len(name):]] = v
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_window_rpb(self, branch_id, seq_len_kv, heads, device):
        """获取形状为 (1, 1, heads, seq_len_kv, seq_len_kv) 的相对位置偏置"""
        key = f"branch_{branch_id}_len_{seq_len_kv}"
        if key not in self.window_rpbs:
            rpb = nn.Parameter(torch.zeros(1, 1, heads, seq_len_kv, seq_len_kv, device=device))
            pending = getattr(self, '_pending_window_rpbs', {})
            if key in pending:
                with torch.no_grad():
                    rpb.copy_(pending[key].to(device))
            else:
                trunc_normal_(rpb)
            self.window_rpbs[key] = rpb
        return self.window_rpbs[key]

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_heads, -1, -1, -1)
        return ref

    def _window_attention_branch(self, q, k, v, H, W, branch_cfg):
        """
        仿照 DWM_MSA 单个分支：输入 q/k/v 形状 (B, H, W, C)，执行窗口注意力。
        返回形状 (B, H, W, C)
        """
        wh, ww = branch_cfg['window_size']
        transpose = branch_cfg['transpose']
        branch_id = branch_cfg['id']

        assert H % wh == 0 and W % ww == 0

        # 1. 划分为窗口
        q = einops.rearrange(q, 'b (h wh) (w ww) c -> b (h w) (wh ww) c', wh=wh, ww=ww)
        k = einops.rearrange(k, 'b (h wh) (w ww) c -> b (h w) (wh ww) c', wh=wh, ww=ww)
        v = einops.rearrange(v, 'b (h wh) (w ww) c -> b (h w) (wh ww) c', wh=wh, ww=ww)

        # 2. local / non‑local 切换，并记录 key/value 序列长度用于 RPE
        if transpose:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            seq_len_kv = q.shape[2]   # 窗口数量
        else:
            seq_len_kv = q.shape[2]   # 窗口内序列长度

        # 3. 拆分为多头
        q = einops.rearrange(q, 'b n l (h d) -> b n h l d', h=self.num_heads)
        k = einops.rearrange(k, 'b n l (h d) -> b n h l d', h=self.num_heads)
        v = einops.rearrange(v, 'b n l (h d) -> b n h l d', h=self.num_heads)

        # 4. 获取相对位置偏置（形状需匹配 sim 的最后两维）
        rpb = self._get_window_rpb(branch_id, seq_len_kv, self.num_heads, q.device)

        # 5. 注意力计算
        q = q * self.scale
        sim = torch.einsum('b n h i d, b n h j d -> b n h i j', q, k)
        sim = sim + rpb
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b n h i j, b n h j d -> b n h i d', attn, v)

        # 6. 恢复形状
        out = einops.rearrange(out, 'b n h l d -> b n l (h d)')
        if transpose:
            out = out.permute(0, 2, 1, 3)

        # 7. 合并窗口回到空间维度
        out = einops.rearrange(
            out,
            'b (h w) (wh ww) c -> b (h wh) (w ww) c',
            h=H // wh, w=W // ww, wh=wh, ww=ww
        )
        return out

    def _attention_compute(self, q, k, v, kernel_size, dilation, is_causal, rpb):
        out = na2d_compat(q.contiguous(), k.contiguous(), v.contiguous(),
                          kernel_size=kernel_size, dilation=dilation,
                          is_causal=is_causal, rpb=rpb)
        return out

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # ---------- 1. 生成 offset 和采样特征(Deformable) ----------
        q_offset = self.proj_q_offset(x)
        q_off = einops.rearrange(q_offset, 'b (g c) h w -> (b g) c h w',
                                 g=self.num_heads, c=self.num_dim_heads)
        offset = self.conv_offset(q_off).contiguous()
        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.num_heads, self.num_dim_heads, H, W),
                grid=pos[..., (1, 0)],
                mode='bilinear',
                align_corners=True
            )

        x_sampled = x_sampled.reshape(B, C, H, W)

        # ---------- 2. 对采样特征进行投影，然后拆分为四个窗口分支(DWM) ----------
        # 转为 (B, H, W, C) 以便线性投影
        x_sampled_hwc = x_sampled.permute(0, 2, 3, 1).contiguous()
        q_win = self.proj_q_win(x_sampled_hwc)
        k_win = self.proj_k_win(x_sampled_hwc)
        v_win = self.proj_v_win(x_sampled_hwc)

        # 按通道均分为 4 份，每份独立进行窗口注意力
        q_chunks_win = torch.chunk(q_win, 4, dim=-1)
        k_chunks_win = torch.chunk(k_win, 4, dim=-1)
        v_chunks_win = torch.chunk(v_win, 4, dim=-1)

        win_outputs = []
        for i in range(4):
            out_win = self._window_attention_branch(
                q_chunks_win[i], k_chunks_win[i], v_chunks_win[i],
                H, W, self.branch_configs[i]
            )
            win_outputs.append(out_win)

        # 拼接四个分支的输出，得到增强后的采样特征
        x_sampled_enhanced_hwc = torch.cat(win_outputs, dim=-1)   # (B, H, W, C)
        x_sampled_enhanced = x_sampled_enhanced_hwc.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # ---------- 3. 邻域注意力阶段 ----------
        q_all = self.proj_q(x)                      # Q 来自原始特征
        k_all = self.proj_k(x_sampled_enhanced)     # K、V 来自增强后的特征
        v_all = self.proj_v(x_sampled_enhanced)

        residual_lepe = self.rpe_table(q_all)

        # 拆分为四份，每份独立计算邻域注意力
        q_chunks = torch.chunk(q_all, 4, dim=1)
        k_chunks = torch.chunk(k_all, 4, dim=1)
        v_chunks = torch.chunk(v_all, 4, dim=1)

        out_branches = []
        for i in range(4):
            qi = q_chunks[i]
            ki = k_chunks[i]
            vi = v_chunks[i]

            head_dim_i = (C // 4) // self.num_heads
            qi = einops.rearrange(qi, 'b (g c) h w -> b g h w c', g=self.num_heads, c=head_dim_i)
            ki = einops.rearrange(ki, 'b (g c) h w -> b g h w c', g=self.num_heads, c=head_dim_i)
            vi = einops.rearrange(vi, 'b (g c) h w -> b g h w c', g=self.num_heads, c=head_dim_i)

            out_i = self._attention_compute(
                qi * self.scale, ki, vi,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
                rpb=self.rpbs[i]
            )
            out_i = einops.rearrange(out_i, 'b g h w c -> b (g c) h w')
            out_branches.append(out_i)

        output = torch.cat(out_branches, dim=1)
        output = self.proj_out(output) + residual_lepe
        return output


if __name__ == '__main__':
    model = DWM_MSDeformableNeighborhoodAttention(
        dim=32,
        heads=2,
        kernel_sizes=[3, 3, 3, 3],
        dilations=[1, 1, 1, 1],
        branch_configs=[
            {'id': 0, 'window_size': (4, 4), 'transpose': False},
            {'id': 1, 'window_size': (4, 4), 'transpose': True},
            {'id': 2, 'window_size': (8, 8), 'transpose': False},
            {'id': 3, 'window_size': (8, 8), 'transpose': True},
        ]
    )
    model.eval()
    x = torch.randn(4, 32, 64, 64)

    with torch.no_grad():
        start = timeit.default_timer()
        out = model(x)
        end = timeit.default_timer()

    elapsed = end - start
    print(f"输出 out 的形状：{out.shape}")
    print(f"前向传播运行时间：{elapsed:.4f} 秒")