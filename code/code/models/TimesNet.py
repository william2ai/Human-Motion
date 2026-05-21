import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

import pdb


def FFT_for_Period(x, k=2):
    # [B, T, C]
    # pdb.set_trace()
    # X(f) is the Fourier transform of the signal 
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    # 此行计算批次和通道维度中每个频率分量的平均幅度。xf 的绝对值给出频率分量的幅度，.mean(0) 和 .mean(-1) 通过计算平均值降低维度
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    # top_list is the indices for the top frequencies
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    # "period" refers to the length of the cycle of a repeating pattern or signal. 
    period = x.shape[1] // top_list
    # abs(xf).mean(-1)[:, top_list] extract and average the magnitudes of the most significant frequency components from the FFT results
    return period, abs(xf).mean(-1)[:, top_list]



class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # pdb.set_trace()
        # B: 16; T:96, N:16
        B, T, N = x.size()
        # print(B,T,N)
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding 不能整除的情况下需要用0进行padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # pdb.set_trace()
            # reshape -> [B, N, length // period, period] N是feature的维度
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'imputation' or self.task_name == 'imputation_graph':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        
        # ADD
        # self.histroy_proj = nn.Linear(self.seq_len, self.pred_len)
        # self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        # self.time_enc = nn.Sequential(
        #                               nn.Linear(self.time_dim, args.c_out//args.rda), 
        #                               nn.LayerNorm(args.c_out//args.rda),
        #                               nn.ReLU(),
        #                               nn.Linear(args.c_out//args.rda, args.c_out//args.rdb), 
        #                               nn.LayerNorm(args.c_out//args.rdb),
        #                               nn.ReLU(),
        #                               nn.Conv1d(in_channels=self.seq_len, 
        #                                         out_channels=self.seq_len, 
        #                                         kernel_size=args.ksize, 
        #                                         padding='same'),
        #                               nn.Linear(args.c_out//args.rdb, args.c_out),
        #                               )
        
        # self.beta = args.beta

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        
        # Normalization from Non-stationary Transformer
        # 这个板块：subtracted by the mean and divided by the standard deviation
        # pdb.set_trace()
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        # reshape -> [B, 1, C]
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)


        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        # valid_values = x_enc * block_mask

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out
    

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation' or self.task_name=="imputation_graph":
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
