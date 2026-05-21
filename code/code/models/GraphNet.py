import torch
import torch.nn as nn
import torch.distributed as dist
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter_softmax
# import torch_scatter
from torch_geometric.utils import add_self_loops, degree, softmax as pyg_softmax
import pdb
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import dense_to_sparse

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
    
def FFT_for_Period(x, k=3, sync_distributed=False):
    # [B, T, C]
    # pdb.set_trace()
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    if sync_distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(frequency_list, op=dist.ReduceOp.SUM)
        frequency_list /= dist.get_world_size()
    frequency_list[0] = -float('inf')
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.clamp_min(1)
    period = (x.shape[1] // top_list).detach().cpu().tolist()
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





# class FrequencyGraphModel(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.k = configs.top_k
#         self.d_model = configs.d_model
#         self.num_sensors = configs.num_sensors
#         self.attention_data = {} 
        
#         # Core components
#         self.fc = nn.Linear(self.d_model, self.d_model)
#         self.graph_convs = nn.ModuleDict()
#         self.dropout = nn.Dropout(0.2)
        
#         self.x=None
#         # ...其他初始化...
#         self.k = configs.top_k
#         # 注意力权重生成层
#         self.period_attention = nn.Linear(self.k, self.k)

    # def add_graph_conv_layer(self, period, edge_index):
    #     """Dynamically add and register graph convolution layers"""
    #     layer_key = f"{period}"
    #     if layer_key not in self.graph_convs:
    #         self.graph_convs[layer_key] = AttentionGraphConvLayer(
    #             in_channels=period,
    #             out_channels=self.seq_len,
    #             num_edges=edge_index.shape[-1]
    #         )

class FrequencyGraphModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.num_sensors = configs.num_sensors
        self.num_edges = min(configs.num_edges, configs.num_sensors - 1)
        self.attention_data = {} 
        
        periods = self._possible_periods(self.seq_len)
        num_graph_edges = self.num_sensors * self.num_edges
        self.graph_convs = nn.ModuleDict({
            str(period): AttentionGraphConvLayer(
                in_channels=period,
                out_channels=self.seq_len,
                num_edges=num_graph_edges,
            )
            for period in periods
        })
        # 其他组件保持不变
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(0.2)
        self.period_attention = nn.Linear(self.k, self.k)

    @staticmethod
    def _possible_periods(seq_len):
        max_freq = max(1, seq_len // 2)
        return sorted({max(1, seq_len // freq) for freq in range(1, max_freq + 1)})
    
    def forward(self, x, edge_index, sync_distributed_periods=False):
        self.x=x
        B, T, N = x.size() # [64,96,64]
        period_list, period_weight = FFT_for_Period(
            x,
            self.k,
            sync_distributed=sync_distributed_periods,
        )
        results = []

        # pdb.set_trace()
        for i in range(self.k):
            period = period_list[i]
            length = self._calculate_padded_length(period, x.device)
            out = self._pad_sequence(x, length, x.device)
            
            # Reshape and process through graph convolution
            processed = self._process_period(out, period, edge_index, B, N)
            results.append(processed[:, :(self.seq_len + self.pred_len), :])

        # pdb.set_trace()
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

    # def _process_period(self, x, period, edge_index, batch_size, num_nodes):
    #     """Process data through graph convolution for a specific period"""
    #     # Reshape tensor for period-based processing
    #     B, L, N = x.size()
    #     x_reshaped = x.view(B, L // period, period, N).permute(0, 3, 1, 2).contiguous()
    #     x_reshaped = x_reshaped.view(B, N, L // period, -1)

    #     # pdb.set_trace()
    #     # Dynamically add graph conv layer if needed
    #     self.add_graph_conv_layer(period, edge_index)
        
    #     # Process through graph convolution
    #     layer_key = f"{period}"
    #     out = self.graph_convs[layer_key](x_reshaped, edge_index)
    #     out = self.dropout(out)
        
        
    #     # pdb.set_trace()
    #     self.attention_data[layer_key] = {
    #         "edge_index": edge_index.cpu().numpy(),
    #         "attention_weights": self.graph_convs[layer_key].batch_edge_data['attention_weights']
    #     }

    #     # pdb.set_trace()
    #     # Reshape back to original dimensions
    #     return out.reshape(B, -1, N)
    
    def _process_period(self, x, period, edge_index, batch_size, num_nodes):
            # 确保 edge_index 在正确的设备上
        device = x.device
        edge_index = edge_index.to(device)
        
        period = int(period)
        graph_conv_layer = self.graph_convs[str(period)]
        
        # 处理数据
        B, L, N = x.size()
        x_reshaped = x.view(B, L // period, period, N).permute(0, 3, 1, 2).contiguous()
        x_reshaped = x_reshaped.view(B, N, L // period, -1)
        
        # 通过图卷积层
        out = graph_conv_layer(x_reshaped, edge_index)
        out = self.dropout(out)
        
        if not self.training and graph_conv_layer.batch_edge_data is not None:
            layer_key = f"{period}"
            self.attention_data[layer_key] = {
                "edge_index": graph_conv_layer.batch_edge_data["edge_index"],
                "attention_weights": graph_conv_layer.batch_edge_data["attention_weights"]
            }
        
        return out.reshape(B, -1, N)
    
    def _aggregate_results(self, results, weights):
        """Aggregate results from different periods"""
        stacked = torch.stack(results, dim=-1)
        weights = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        return torch.sum(stacked * weights, dim=-1)


# class AttentionGraphConvLayer(MessagePassing):
#     def __init__(self, in_channels, out_channels, num_edges):
#         super().__init__(aggr='add')
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.lin = nn.Linear(in_channels, out_channels).to(device)
#         self.attention = nn.Parameter(torch.Tensor(1, out_channels)).to(device)
#         # self.edge_weights = nn.Parameter(torch.ones(num_edges)).to(device)\
#         total_edges=1728
#         self.edge_weights = nn.Parameter(torch.ones(total_edges))
#         # self.attention_weights = None  # 新增属性
    
        
#         nn.init.xavier_uniform_(self.attention)
#         nn.init.normal_(self.lin.weight, mean=0, std=0.1)

class AttentionGraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edges):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.attention = nn.Parameter(torch.Tensor(1, out_channels))
        self.batch_edge_data = None
        
        # 关键修改：使用实际边数而不是硬编码值
        self.edge_weights = nn.Parameter(torch.ones(num_edges))
        
        nn.init.xavier_uniform_(self.attention)
        nn.init.normal_(self.lin.weight, mean=0, std=0.1)
        
    # def forward(self, x, edge_index):
    #     # 输入形状:
    #     # x: [batch_size*num_nodes, seq_len, d_model] = [64*9=576, 96, 32]
    #     # edge_index: [64, 2, 27]
    #     B,N,T,_=x.shape
    #     x = self.lin(x.mean(dim=2)).reshape(B,N,-1)  # Temporal aggregation
        
    #     # 展平处理
    #     batch_size = edge_index.shape[0]
    #     edge_index_flat = edge_index.view(2, -1)  # [2, 1728]
        
    #     # 计算边权重
    #     # pdb.set_trace()
    #     src_nodes = edge_index_flat[0]  # [1728]
    #     dst_nodes = edge_index_flat[1]  # [1728]
    #     edge_weights = self.edge_weights[src_nodes] * self.edge_weights[dst_nodes]
        
    #     # 按目标节点分组归一化
    #     unique_dst = torch.unique(dst_nodes)
    #     softmax_weights = torch.zeros_like(edge_weights)
    #     for dst in unique_dst:
    #         mask = (dst_nodes == dst)
    #         softmax_weights[mask] = F.softmax(edge_weights[mask], dim=0)
        
    #     self.batch_edge_data = {
    #         "edge_index": edge_index.cpu().detach().numpy(),  # [64, 2, 27]
    #         "attention_weights": softmax_weights.view(batch_size, edge_index.size(-1)).cpu().detach().numpy()  # [64, 27]
    #     }
        
    #     out=self.propagate(edge_index_flat.to(x.device), x=x, edge_weights=softmax_weights.to(x.device))
        
    #     return out.permute(0,2,1)
    def forward(self, x, edge_index):
            # 确保输入在正确设备上
        device = x.device
        edge_index = edge_index.to(device)

        # 处理形状
        B, N, T, _ = x.shape
        x = self.lin(x.mean(dim=2)).reshape(B * N, -1)
        
        # Build a disjoint batched graph so each sample uses its own dynamic edges.
        batch_size = edge_index.shape[0]
        num_edges = edge_index.size(-1)
        offsets = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * N
        edge_index_flat = (edge_index + offsets).permute(1, 0, 2).contiguous().view(2, -1)
        
        edge_weights = self.edge_weights[:num_edges].repeat(batch_size)
        
        dst_nodes = edge_index_flat[1]
        softmax_weights = pyg_softmax(edge_weights, dst_nodes)
        
        if not self.training:
            self.batch_edge_data = {
                "edge_index": edge_index.cpu().detach().numpy(),
                "attention_weights": softmax_weights.view(batch_size, num_edges).cpu().detach().numpy()
            }
        else:
            self.batch_edge_data = None
        
        # 传播
        out = self.propagate(edge_index_flat, x=x, edge_weights=softmax_weights)
        
        return out.view(B, N, -1).permute(0, 2, 1)
    
    def message(self, x_j, edge_weights):
        activated_weights = F.leaky_relu(edge_weights, negative_slope=0.2)
        return x_j * activated_weights.unsqueeze(1)
        # return x_j * edge_weights.unsqueeze(1)  # 保持维度一致性



def create_graph(x_enc, top_k=3):
    # """在batch内动态生成图结构"""
    B, T, N = x_enc.shape
    top_k = max(1, min(int(top_k), N - 1))
    device = x_enc.device  # 获取输入数据的设备
    
    # 计算每个batch独立的相似度矩阵
    x_reshaped = x_enc.permute(0, 2, 1)  # [B, N, T]
    norm = torch.norm(x_reshaped, dim=2, keepdim=True)  # [B, N, 1]
    similarity_matrix = torch.matmul(x_reshaped, x_reshaped.transpose(1,2)) / (norm * norm.transpose(1,2)).clamp_min(1e-6)
    diag_idx = torch.arange(N, device=device)
    similarity_matrix[:, diag_idx, diag_idx] = -float('inf')
    
    top_indices = torch.topk(similarity_matrix, k=top_k, dim=-1).indices
    src_indices = torch.arange(N, device=device).view(1, N, 1).expand(B, N, top_k)
    return torch.stack((src_indices, top_indices), dim=1).reshape(B, 2, N * top_k).contiguous()

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
        
        self.tcn_enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                         configs.freq, configs.dropout)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.enc_in, configs.embed, 
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
        self.graph_norm = nn.LayerNorm(configs.c_out)
        self.norm=nn.LayerNorm(configs.c_out)
        
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
        
        self.lin = nn.Linear(in_features=configs.c_out, out_features=configs.d_model)
        # self.lin2=nn.Linear(128, 64)
        self.lin3=nn.Linear(configs.d_model, configs.c_out)
        self.sync_distributed_periods = True
        

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
        
        
        edge_index = create_graph(x_enc, self.configs.num_edges)
        
        x_enc_tcn = self.tcn_enc_embedding(x_enc, x_mark_enc)
        
        
        
        # 频率图处理
        # pdb.set_trace()
        sync_distributed_periods = (
            self.sync_distributed_periods
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        out,period_list=self.frequency_graph_model(
            x_enc,
            edge_index,
            sync_distributed_periods=sync_distributed_periods,
        )
        graph_out = self.graph_norm(x_enc +out)
        
        # 获取注意力数据
        # attention_data = self.frequency_graph_model.attention_data
        # visualize_attention(batch_idx=0, layer_data=attention_data['96'])


        # 时序卷积处理
        tcn_out = self.tcn_norm(x_enc_tcn + self.tcn(x_enc_tcn))
        
        
        ## 注意力机制
        # pdb.set_trace()
        graph_proj = self.lin(graph_out)  # Query
        
        
        # attn_out = F.scaled_dot_product_attention(graph_proj, tcn_out, tcn_out)  # (64, 96, 64)   
        # residual_out = attn_out + tcn_out 
        
        
        # Replace the attention with a linear transformation
        # linear_out = self.lin2(graph_proj)  # Apply a linear layer
        # Combine with tcn_out (residual connection)
        residual_out = graph_proj + tcn_out  # Perform residual addition
        
        
        dec_out = self.norm(self.lin3(residual_out))
        
        # 反标准化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pred_len, 1)
        dec_out += means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pred_len, 1)
        
        if self.training:
            self.step_counter += 1
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,labels=None):

        if self.task_name in ['imputation', 'imputation_graph']:
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            # pdb.set_trace()
            # labels.shape=torch.Size([64, 96])
            # x_enc.shape=torch.Size([64, 96,9])

            return dec_out
        return None
