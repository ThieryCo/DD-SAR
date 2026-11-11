# DD-SAR

**DD-SAR: Deformable Dual-window Attention Network for SAR Image Despeckling**

DD-SAR 是一个用于 SAR 图像相干斑抑制的去噪网络。它以 U 形编解码结构为骨架，核心是
**DWM（Dual-Window Multi-branch）可变形邻域注意力**：先由可偏移采样（deformable）生成
偏移并对特征重采样，再对采样特征做四分支窗口注意力（local / non-local 两种划分方式 ×
两种窗口尺寸），最后以四分支邻域注意力（neighborhood attention）融合原始特征与增强特征。
编解码跳跃连接处使用**空间交叉注意力融合块（SAB）**聚合浅层细节与深层语义。

## 网络结构

```
输入 (1ch)
 ├─ input_proj (3×3 conv) → dim=32
 ├─ SAB_1 / SAB_2                     (空间交叉注意力融合块, 用于解码端)
 ├─ Encoder:  TransformerBlock×7 → Downsample → TransformerBlock×7 → Downsample
 ├─ Middle:   TransformerBlock×6
 ├─ Decoder:  Upsample → [SAB 融合 + 残差] → TransformerBlock×7
 │            Upsample → [SAB 融合 + 残差] → TransformerBlock×7
 └─ 输出: concat(enc_in_1) → 1×1 conv → 去噪结果 (1ch)
```

每个 `TransformerBlock` = LayerNorm → **DWM_MSDeformableNeighborhoodAttention** →
LayerNorm → Gated-FFN（DWConv + 门控 GELU）。

`DWM_MSDeformableNeighborhoodAttention` 分三个阶段：

1. **Deformable 采样**：逐头生成偏移量，`grid_sample` 重采样得到对齐特征；
2. **双窗口分支注意力**：对采样特征按通道四等分，分别做
   `(4,4)`/`(8,8)` × 正常/转置 共 4 种划分的窗口注意力后拼接；
3. **分支邻域注意力**：Q 来自原始特征、K/V 来自增强特征，按通道四等分，
   以不同 `kernel_size`/`dilation` 的邻域注意力计算并融合（含相对位置偏置 rpb 与
   局部位置编码 LPE 残差）。

## 仓库结构

```
├── train.py / test.py             # 训练 / 推理入口
├── nets/
│   ├── __init__.py                # 网络注册 (define / init / scheduler)
│   ├── DD_SAR.py                  # DD-SAR 主网络 (DD_SAR 类)
│   ├── Transformer_Block.py       # TransformerBlock (LayerNorm/FFN)
│   ├── DWM_Deformable_Transformer.py  # 核心: 可变形 + 双窗口 + 邻域注意力
│   └── Spatial_Attention_Block.py     # SAB 跳跃连接融合块
├── models/DD_SAR_model.py         # 训练/推理封装 (AdamW, MIX3 loss)
├── loss/loss.py                   # MIX3 = Charbonnier + FFT-L1 (weight 0.01)
├── data/aligned_dataset.py        # 噪声/干净成对数据 (随机 1/4 尺寸裁剪 + 翻转)
├── options/                       # argparse 配置
├── util/                          # 日志 / 可视化 / 图像保存
└── checkpoints/DD-SAR/4/latest_net_DD_SAR.pth   # 预训练权重 (200 epochs, ~23MB)
```

## 运行环境

论文实验环境（`nets/DWM_Deformable_Transformer.py` 中的邻域注意力
NATTEN 算子按此环境配置）：

| 项目 | 版本 |
|------|------|
| 操作系统 | Linux |
| 邻域注意力算子 | [NATTEN](https://www.shi-labs.com/natten/) `natten==0.17.5` |
| PyTorch | 2.x（与 natten 0.17.5 官方 wheel 兼容的版本，如 torch≤2.5） |
| GPU | NVIDIA CUDA |

本仓库亦可在无 NATTEN 的环境（含 Windows / 新版 PyTorch）直接运行：未检测到
`natten==0.17.x` 时自动使用仓库内置的纯 PyTorch 邻域注意力实现（数学等价、含 rpb）。
已验证环境：Windows 10 / Python 3.12 / PyTorch 2.14.1+cu126 / 无 NATTEN。

安装依赖：

```bash
pip install -r requirements.txt
# 论文原始加速路径 (可选, Linux):
pip install natten==0.17.5
```

## 数据准备

合成训练数据为成对的噪声/干净图像（同名文件按 `sorted` 顺序一一对应）：

```
datasets/synthetic/AID/Look1/train/
├── noisy_train/     # 噪声图像
└── ground_truth/    # 干净图像
```

测试阶段只需噪声图像目录（真实 SAR 数据）：

```
datasets/real/SAR_intensity/   # 单通道强度图, 任意尺寸
```

> 训练时从图像中随机裁剪 `H/4 × H/4` 的 patch（官方 256×256 数据 → 64×64 输入）。

## 训练

```bash
python train.py \
    --data_root_noise ./datasets/synthetic/AID/Look1/train/noisy_train \
    --data_root_clean ./datasets/synthetic/AID/Look1/train/ground_truth
```

主要超参（`options/train_options.py`）：AdamW，lr `2e-4`，weight decay `1e-4`，
batch size 2，共 `niter + niter_decay = 200` epochs（前 100 恒定 lr，后 100 线性衰减），
损失为 MIX3（Charbonnier + FFT-L1×0.01）。checkpoint 保存在
`./checkpoints/DD-SAR/4/`（可用 `--look` 更改子目录名）。

## 推理

```bash
# 使用仓库自带预训练权重 (./checkpoints/DD-SAR/4/latest_net_DD_SAR.pth)
python test.py --data_root_noise <噪声图像目录> --num_test 100

# 结果保存在 ./results/DD-SAR/test_latest/images/
```

也可以直接单独构建网络做前向：

```python
import torch, nets
net = nets.define('normal', 0.02, [], type('O', (), {'net': 'DD_SAR'})())
x = torch.randn(1, 1, 256, 256)
y = net(x)   # (1, 1, 256, 256)
```

## 预训练权重

`checkpoints/DD-SAR/4/latest_net_DD_SAR.pth` 为训练 200 epochs 后的 DD-SAR 权重
（约 3.0M 参数）。其中窗口相对位置偏置（window RPE）按输入分辨率懒创建，
加载 checkpoint 后首次前向会自动恢复训练值。

## 引用

如果你在研究中使用了本代码，请引用：

```bibtex
@article{DD-SAR,
  title   = {DD-SAR: Deformable Dual-window Attention Network for SAR Image Despeckling},
  author  = {},
  journal = {},
  year    = {}
}
```

## License

MIT License，详见 [LICENSE](LICENSE)。
