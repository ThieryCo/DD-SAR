import timeit
import torch
import torch.nn as nn

from nets.Transformer_Block import TransformerBlock, RDB_Block
from nets.Spatial_Attention_Block import Spacial_Attention_Block


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DD_SAR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=32):
        super(DD_SAR, self).__init__()
        self.heads = [2, 4, 8]
        self.blocks = [7, 7, 6, 7, 7]
        self.input_proj = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.SAB_1 = Spacial_Attention_Block(channel=dim)
        self.SAB_2 = Spacial_Attention_Block(channel=dim*2)

        self.encoder_Swin_1 = nn.Sequential(*[
            TransformerBlock(dim=dim*1, num_heads=self.heads[0]) for _ in range(self.blocks[0])
        ])
        self.down_1to2 = Downsample(dim)
        self.encoder_Swin_2 = nn.Sequential(*[
            TransformerBlock(dim=dim*2, num_heads=self.heads[1]) for _ in range(self.blocks[1])
        ])
        self.down_2to3 = Downsample(dim*2)

        self.middle = nn.Sequential(*[
            TransformerBlock(dim=dim*4, num_heads=self.heads[2]) for _ in range(self.blocks[2])
        ])

        self.up_3to2 = Upsample(dim * 4)
        self.reduce_channel_3to2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=False)
        self.decoder_Swin_2 = nn.Sequential(*[
            TransformerBlock(dim=dim*2, num_heads=self.heads[1]) for _ in range(self.blocks[3])
        ])
        self.up_2to1 = Upsample(dim*2)
        self.reduce_channel_2to1 = nn.Conv2d(int(dim * 2), int(dim * 1), kernel_size=1, bias=False)
        self.decoder_Swin_1 = nn.Sequential(*[
            TransformerBlock(dim=dim*1, num_heads=self.heads[0]) for _ in range(self.blocks[4])
        ])

        self.output_proj = nn.Conv2d(dim * 2, out_channels, kernel_size=1, bias=False)


    def forward(self, input):
        DSwinIR_enc_in_1 = self.input_proj(input)
        DSwinIR_enc_out_1 = self.encoder_Swin_1(DSwinIR_enc_in_1)
        DSwinIR_enc_in_2 = self.down_1to2(DSwinIR_enc_out_1)
        DSwinIR_enc_out_2 = self.encoder_Swin_2(DSwinIR_enc_in_2)
        DSwinIR_enc_in_3 = self.down_2to3(DSwinIR_enc_out_2)

        DSwinIR_middle = self.middle(DSwinIR_enc_in_3)

        DSwinIR_dec_in_3 = self.up_3to2(DSwinIR_middle)
        SAB_1 = self.SAB_2(DSwinIR_dec_in_3,DSwinIR_enc_in_2)
        DSwinIR_dec_in_3 = DSwinIR_dec_in_3 + DSwinIR_enc_in_2 + SAB_1

        DSwinIR_dec_out_3 = self.decoder_Swin_2(DSwinIR_dec_in_3)
        DSwinIR_dec_in_2 = self.up_2to1(DSwinIR_dec_out_3)
        SAB_2 = self.SAB_1(DSwinIR_dec_in_2, DSwinIR_enc_in_1)
        DSwinIR_dec_in_2 = DSwinIR_dec_in_2 + DSwinIR_enc_in_1 + SAB_2

        DSwinIR_dec_out_2 = self.decoder_Swin_1(DSwinIR_dec_in_2)

        output = torch.cat([DSwinIR_dec_out_2, DSwinIR_enc_in_1], dim=1)
        output = self.output_proj(output)

        return output


if __name__ == '__main__':
    model = DD_SAR()
    model.eval()
    x = torch.randn(4, 1, 64, 64)

    with torch.no_grad():
        start = timeit.default_timer()
        out = model(x)
        end = timeit.default_timer()

    elapsed = end - start
    print(f"输出 out 的形状：{out.shape}")
    print(f"前向传播运行时间：{elapsed:.4f} 秒")