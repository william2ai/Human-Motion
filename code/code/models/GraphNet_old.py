import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter_softmax
# import torch_scatter
from torch_geometric.utils import add_self_loops, degree
import pdb
import numpy as np
import math

# TCN: mse:0.22716310620307922, mae:0.13739536702632904

def FFT_for_Period(x, k=3):
    # [B, T, C]
    # pdb.set_trace()
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, dropout=0.1):
        super(TCNBlock, self).__init__()
        # 修正padding计算保证序列长度不变
        padding = dilation * (kernel_size - 1) // 2  # 关键修改点
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(
            input_dim, 
            output_dim, 
            kernel_size,
            padding=padding,  # 使用修正后的padding
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(output_dim)
        
        # 第二层卷积（保持相同padding）
        self.conv2 = nn.Conv1d(
            output_dim, 
            output_dim, 
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # 下采样层（当输入输出通道数不一致时调整）
        self.downsample = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        # 保存原始序列长度用于最后裁切
        original_length = x.size(-1)
        
        # 第一个卷积+BN+激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # 第二个卷积+BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接前的维度匹配
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # 确保序列长度一致（针对奇数长度场景的额外保护）
        residual = residual[..., :original_length]  # 裁切到原始长度
        out = out[..., :original_length]
        
        # 残差连接
        out += residual
        
        return F.gelu(out)



class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=4, dropout=0.1):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        
        # 逐层构建TCN块
        for i in range(num_layers):
            dilation = 2 ** i  # 指数级增长的膨胀率
            in_channels = input_dim if i == 0 else output_dim
            self.layers.append(
                TCNBlock(
                    in_channels, 
                    output_dim,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
        
    def forward(self, x):
        # 调整维度: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        
        # 逐层处理
        for layer in self.layers:
            x = layer(x)
        
        # 恢复维度: (B, C, L) -> (B, L, C)
        return x.permute(0, 2, 1)





class FrequencyGraphModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.num_sensors = configs.num_sensors
        
        # Core components
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.graph_convs = nn.ModuleDict()
        self.dropout = nn.Dropout(0.2)

    def add_graph_conv_layer(self, period, edge_index):
        """Dynamically add and register graph convolution layers"""
        layer_key = f"period_{period}"
        if layer_key not in self.graph_convs:
            self.graph_convs[layer_key] = AttentionGraphConvLayer(
                in_channels=period,
                out_channels=self.seq_len,
                num_edges=edge_index.shape[1]
            )

    def forward(self, x, edge_index):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        results = []

        for i in range(self.k):
            period = period_list[i]
            length = self._calculate_padded_length(period, x.device)
            out = self._pad_sequence(x, length, x.device)
            
            # Reshape and process through graph convolution
            processed = self._process_period(out, period, edge_index, B, N)
            results.append(processed[:, :(self.seq_len + self.pred_len), :])

        return self._aggregate_results(results, period_weight)

    def _calculate_padded_length(self, period, device):
        """Calculate required padding length for the sequence"""
        base_length = self.seq_len + self.pred_len
        if base_length % period != 0:
            return ((base_length // period) + 1) * period
        return base_length

    def _pad_sequence(self, x, target_length, device):
        """Pad sequence to target length"""
        current_length = x.size(1)
        if current_length < target_length:
            padding = torch.zeros(x.size(0), target_length - current_length, x.size(2), device=device)
            return torch.cat([x, padding], dim=1)
        return x

    def _process_period(self, x, period, edge_index, batch_size, num_nodes):
        """Process data through graph convolution for a specific period"""
        # Reshape tensor for period-based processing
        B, L, N = x.size()
        x_reshaped = x.view(B, L // period, period, N).permute(0, 3, 1, 2).contiguous()
        x_reshaped = x_reshaped.view(B * N, L // period, -1)

        # Dynamically add graph conv layer if needed
        self.add_graph_conv_layer(period, edge_index)
        
        # Process through graph convolution
        layer_key = f"period_{period}"
        out = self.graph_convs[layer_key](x_reshaped, edge_index)
        out = self.dropout(out)

        # Reshape back to original dimensions
        return out.view(B, N, -1, out.shape[-1]).permute(0, 2, 1, 3).reshape(B, -1, N)

    def _aggregate_results(self, results, weights):
        """Aggregate results from different periods"""
        stacked = torch.stack(results, dim=-1)
        weights = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        return torch.sum(stacked * weights, dim=-1)


class AttentionGraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edges):
        super().__init__(aggr='add')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lin = nn.Linear(in_channels, out_channels).to(device)
        self.attention = nn.Parameter(torch.Tensor(1, out_channels)).to(device)
        self.edge_weights = nn.Parameter(torch.ones(num_edges)).to(device)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.attention)
        nn.init.normal_(self.lin.weight, mean=0, std=0.1)

    def forward(self, x, edge_index):
        x = self.lin(x.mean(dim=1))  # Temporal aggregation
        edge_weights = F.softmax(self.edge_weights[edge_index[0]] * self.edge_weights[edge_index[1]], dim=0)
        return self.propagate(edge_index, x=x, edge_weights=edge_weights)

    def message(self, x_j, edge_weights):
        return edge_weights.view(-1, 1) * x_j


def create_graph(configs, x_enc, top_k=3):
    """Vectorized graph creation with batch processing"""
    B, T, N = x_enc.shape
    x_reshaped = x_enc.permute(0, 2, 1)
    
    # Vectorized cosine similarity calculation
    norm = torch.norm(x_reshaped, dim=2, keepdim=True)
    similarity_matrix = torch.matmul(x_reshaped, x_reshaped.transpose(1,2)) / (norm * norm.transpose(1,2))
    
    # Top-k selection with exclusion of self-connections
    top_k_indices = torch.topk(similarity_matrix, k=top_k+1, dim=2).indices[:, :, 1:]
    
    # Generate edge indices using vectorized operations
    sensors = torch.arange(N, device=x_enc.device).view(1, N, 1).expand(B, N, top_k)
    edges = torch.stack([
        sensors.flatten(),
        top_k_indices.flatten()
    ], dim=0)
    
    # Remove duplicate edges and maintain batch information
    unique_edges = torch.unique(edges, dim=1, sorted=False)
    return unique_edges.contiguous()


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                         configs.freq, configs.dropout)
        
        # 使用优化后的频率图模型
        self.frequency_graph_model = FrequencyGraphModel(configs)
        
        # 时间卷积网络增强
        self.tcn = nn.Sequential(
            TCN(
                input_dim=configs.d_model,
                output_dim=configs.d_model*2,
                kernel_size=3,
                num_layers=4,
                dropout=configs.dropout
            ),
            nn.GELU(),
            nn.Linear(configs.d_model*2, configs.d_model)
        )
        
        # 改进的归一化层
        self.tcn_norm = nn.LayerNorm(configs.d_model)
        self.graph_norm = nn.LayerNorm(configs.d_model)
        
        # 动态投影层
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model*2),
            nn.GELU(),
            nn.Linear(configs.d_model*2, configs.c_out)
        )
        
        # 图缓存相关参数
        self.graph_update_interval = 10  # 每10个step更新一次图结构
        self.register_buffer('cached_edge_index', None)
        self.step_counter = 0

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 数据标准化
        x_raw = x_enc.clone().detach()
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                         torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        # 嵌入层
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        
        # 动态图生成
        edge_index = create_graph(self.configs,x_enc)
        
        # 频率图处理
        # pdb.set_trace()
        x_enc = self.graph_norm(x_enc + self.frequency_graph_model(x_enc, edge_index))
        
        # 时序卷积处理
        x_enc = self.tcn_norm(x_enc + self.tcn(x_enc))
        
        # 投影输出
        dec_out = self.projection(x_enc)
        
        # 反标准化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pred_len, 1)
        dec_out += means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pred_len, 1)
        
        if self.training:
            self.step_counter += 1
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['imputation', 'imputation_graph']:
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return None

