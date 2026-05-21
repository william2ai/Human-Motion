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
import matplotlib.pyplot as plt
import networkx as nx

def plot_attention_heatmap(edge_index, weights, num_nodes):
    """绘制注意力权重热力图"""
    pdb.set_trace()
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj_matrix[src, dst] = weights[i]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    # plt.show()
    plt.savefig("1.png")

def plot_attention_graph(edge_index, weights, num_nodes):
    """绘制带权重的图结构"""
    G = nx.DiGraph()
    
    # 添加带权重的边
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        G.add_edge(src, dst, weight=weights[i])
    
    # 布局和可视化参数
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)
    weights = [w['weight'] for (u, v, w) in edges]
    
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, 
            with_labels=True,
            node_color='skyblue',
            node_size=500,
            edge_color=weights,
            edge_cmap=plt.cm.Blues,
            width=2.0,
            arrowsize=20)
    plt.title("Graph Attention Structure")
    # plt.show()
    plt.savefig("2.png")
    
# TCN: mse:0.22716310620307922, mae:0.13739536702632904


def visualize_attention(batch_idx, layer_data, num_nodes=9):
    """可视化指定batch样本的注意力图"""
    # 提取数据
    edge_index = layer_data["edge_index"][batch_idx]  # [2, 27]
    weights = layer_data["attention_weights"][batch_idx]  # [27]
    
    # 创建邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj_matrix[src, dst] = weights[i]
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap="viridis", interpolation='nearest')
    plt.colorbar()
    plt.title(f"Batch {batch_idx} Attention Heatmap")
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.xticks(range(num_nodes))
    plt.yticks(range(num_nodes))
    # plt.show()
    
    # 绘制网络图
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    for src, dst, w in zip(edge_index[0], edge_index[1], weights):
        G.add_edge(src, dst, weight=w)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, 
            node_color='skyblue', 
            edge_color=weights, 
            edge_cmap=plt.cm.Blues,
            width=2.0, 
            arrowsize=20)
    plt.title(f"Batch {batch_idx} Graph Attention")
    # plt.show()
    plt.savefig("tmp.png")
    
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
        self.attention_data = {} 
        
        # Core components
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.graph_convs = nn.ModuleDict()
        self.dropout = nn.Dropout(0.2)

    def add_graph_conv_layer(self, period, edge_index):
        """Dynamically add and register graph convolution layers"""
        layer_key = f"{period}"
        if layer_key not in self.graph_convs:
            self.graph_convs[layer_key] = AttentionGraphConvLayer(
                in_channels=period,
                out_channels=self.seq_len,
                num_edges=edge_index.shape[-1]
            )

    def forward(self, x, edge_index):
        B, T, N = x.size() # [64,96,64]
        period_list, period_weight = FFT_for_Period(x, self.k)
        results = []

        # pdb.set_trace()
        for i in range(self.k):
            period = period_list[i]
            length = self._calculate_padded_length(period, x.device)
            out = self._pad_sequence(x, length, x.device)
            
            # Reshape and process through graph convolution
            processed = self._process_period(out, period, edge_index, B, N)
            results.append(processed[:, :(self.seq_len + self.pred_len), :])

        return self._aggregate_results(results, period_weight), period_list

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
        x_reshaped = x_reshaped.view(B, N, L // period, -1)

        # pdb.set_trace()
        # Dynamically add graph conv layer if needed
        self.add_graph_conv_layer(period, edge_index)
        
        # Process through graph convolution
        layer_key = f"{period}"
        out = self.graph_convs[layer_key](x_reshaped, edge_index)
        out = self.dropout(out)
        
        
        # pdb.set_trace()
        self.attention_data[layer_key] = {
            "edge_index": edge_index.cpu().numpy(),
            "attention_weights": self.graph_convs[layer_key].batch_edge_data['attention_weights']
        }

        # pdb.set_trace()
        # Reshape back to original dimensions
        return out.reshape(B, -1, N)

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
        # self.edge_weights = nn.Parameter(torch.ones(num_edges)).to(device)\
        total_edges=1728
        self.edge_weights = nn.Parameter(torch.ones(total_edges))
        # self.attention_weights = None  # 新增属性
        
        
        
        nn.init.xavier_uniform_(self.attention)
        nn.init.normal_(self.lin.weight, mean=0, std=0.1)

    def forward(self, x, edge_index):
        # 输入形状:
        # x: [batch_size*num_nodes, seq_len, d_model] = [64*9=576, 96, 32]
        # edge_index: [64, 2, 27]
        B,N,T,_=x.shape
        x = self.lin(x.mean(dim=2)).reshape(B,N,-1)  # Temporal aggregation
        
        # 展平处理
        batch_size = edge_index.shape[0]
        edge_index_flat = edge_index.view(2, -1)  # [2, 1728]
        
        # 计算边权重
        # pdb.set_trace()
        src_nodes = edge_index_flat[0]  # [1728]
        dst_nodes = edge_index_flat[1]  # [1728]
        edge_weights = self.edge_weights[src_nodes] * self.edge_weights[dst_nodes]
        
        # 按目标节点分组归一化
        unique_dst = torch.unique(dst_nodes)
        softmax_weights = torch.zeros_like(edge_weights)
        for dst in unique_dst:
            mask = (dst_nodes == dst)
            softmax_weights[mask] = F.softmax(edge_weights[mask], dim=0)
        
        self.batch_edge_data = {
            "edge_index": edge_index.cpu().detach().numpy(),  # [64, 2, 27]
            "attention_weights": softmax_weights.view(batch_size, 27).cpu().detach().numpy()  # [64, 27]
        }
        
        out=self.propagate(edge_index_flat.to(x.device), x=x, edge_weights=softmax_weights.to(x.device))
        
        # pdb.set_trace()
        return out.permute(0,2,1)
    
    def message(self, x_j, edge_weights):
        return x_j * edge_weights.unsqueeze(1)  # 保持维度一致性


def create_graph(x_enc, top_k=3):
    # pdb.set_trace()
    """在batch内动态生成图结构"""
    B, T, N = x_enc.shape  # x_enc是当前batch的输入 [B, T, N]
    
    # 计算每个batch独立的相似度矩阵
    x_reshaped = x_enc.permute(0, 2, 1)  # [B, N, T]
    norm = torch.norm(x_reshaped, dim=2, keepdim=True)  # [B, N, 1]
    similarity_matrix = torch.matmul(x_reshaped, x_reshaped.transpose(1,2)) / (norm * norm.transpose(1,2))
    
    # 为每个样本生成独立的边索引
    batch_edge_indices = []
    for b in range(B):
        edge_pairs = []
        for src in range(N):
            # 获取top_k+1相似节点（排除自己）
            _, top_indices = torch.topk(similarity_matrix[b, src], k=top_k+1)
            for dst in top_indices[1:].tolist():
                edge_pairs.append([src, dst])
        # 去重并转换为tensor
        edge_tensor = torch.unique(torch.tensor(edge_pairs),dim=0).t()
        batch_edge_indices.append(edge_tensor)
    
    # pdb.set_trace()
    # 堆叠所有batch的边索引 [B, 2, num_edges]
    return torch.stack(batch_edge_indices, dim=0) 

class NodeAwareEmbedding(nn.Module):
    def __init__(self, num_nodes, d_model, dropout=0.1):
        super().__init__()
        # 每个节点独立嵌入
        self.node_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_nodes)
        ])
        # self.position_emb = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, num_nodes]
        node_embeddings = []
        for i in range(x.size(-1)):
            # 处理每个节点的时间序列
            node_feat = x[..., i].unsqueeze(-1)  # [B, L, 1]
            emb = self.node_embeddings[i](node_feat)  # [B, L, d_model]
            # emb += self.position_emb(node_feat)  # 加入位置编码
            node_embeddings.append(emb)
        x = torch.stack(node_embeddings, dim=2)  # [B, L, N, d_model]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        
        # Embedding
        # 节点感知嵌入
        self.embedding = NodeAwareEmbedding(
            num_nodes=configs.num_sensors,
            d_model=configs.d_model,
            dropout=configs.dropout
        )
        
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
        self.graph_norm = nn.LayerNorm(9)
        
        # 动态投影层
        self.projection = nn.Sequential(
            nn.Linear(9, configs.d_model*2),
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
        
        # pdb.set_trace()
        # 动态图生成
        
        
        
        pdb.set_trace()
        # 嵌入层
        # x_enc = self.enc_embedding(x_enc, x_mark_enc)
        x = self.embedding(x_enc)
        
        edge_index = create_graph(x_enc)
        
        # 频率图处理
        # pdb.set_trace()
        out,period_list=self.frequency_graph_model(x_enc, edge_index)
        x_enc = self.graph_norm(x_enc +out )
        
        # 获取注意力数据
        # attention_data = self.frequency_graph_model.attention_data
        # visualize_attention(batch_idx=0, layer_data=attention_data['96'])

 
        # 时序卷积处理
        # x_enc = self.tcn_norm(x_enc + self.tcn(x_enc))
        
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

