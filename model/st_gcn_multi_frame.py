import torch
import torch.nn as nn

from model.Hypergraph import Graph
from model.Hypergraph_pool import Graph_pool
from model.time_transformer import  Transformer as time_trans
from einops import rearrange

inter_channels = [16, 32, 64, 128]

fc_out = inter_channels[-1]
fc_unit = 256
class GCN_Model(nn.Module):
    def __init__(self,hypergraph):
        super().__init__()

        # load graph
        self.momentum =  0.1
        self.in_channels = 2
        self.out_channels = 3
        self.layout = 'hm36_gt'
        self.strategy = 'spatial'
        self.cat = True
        self.inplace = True
        self.pad = 2
        self.sym_base = hypergraph
        self.time_length = (self.pad*2)+1

        self.graph = Graph(self.layout, self.strategy,self.sym_base, pad=self.pad )

        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda().unsqueeze(0) # K, T*V, T*V

        self.graph_pool = Graph_pool(self.layout, self.strategy, pad=self.pad )

        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda().unsqueeze(0)  #6 25 25

        kernel_size = 4

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.in_channels, inter_channels[0], kernel_size, residual=False),
            st_gcn(inter_channels[0], inter_channels[1], kernel_size),
        ))

        self.trans_networks = nn.ModuleList((
            time_trans(embed_dim=inter_channels[0], mlp_hidden_dim=2*inter_channels[0], length=self.time_length),
            time_trans(embed_dim=inter_channels[1], mlp_hidden_dim=2 * inter_channels[1], length=self.time_length),
            time_trans(embed_dim=inter_channels[2], mlp_hidden_dim=2 * inter_channels[2], length=self.time_length),
        ))

        self.st_gcn_pool = nn.ModuleList((
            st_gcn(inter_channels[1], inter_channels[2], kernel_size),
        ))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(inter_channels[2], inter_channels[3], kernel_size=(3, 1), padding = (1, 0)),
            nn.BatchNorm2d(inter_channels[3], momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels[3], fc_unit, kernel_size=(1, 1), padding = (0, 0)),
            nn.BatchNorm2d(fc_unit, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.1)
        )

        self.fcn = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.Conv2d(fc_unit, self.out_channels, kernel_size=1)
        )


    def graph_max_pool(self, x, p,stride=None):
        if max(p) > 1:
            if stride is None:
                x = nn.MaxPool2d(p)(x)
            else:
                x = nn.MaxPool2d(kernel_size=p,stride=stride)(x)
            return x
        else:
            return x


    def forward(self, x):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, 1, -1)


        x1,_ = self.st_gcn_networks[0](x,self.A)
        x1 =  x1.view(N*V,T,-1)
        x1 = self.trans_networks[0](x1)
        x1 = rearrange(x1, "(b j) t c -> b c (t j)",j=V).unsqueeze(2)

        x2, _ = self.st_gcn_networks[1](x1, self.A)

        x2 = x2.view(N, -1, T, V)
        x_avg_pool1 = self.avg_pool(x2.view(N,(T*V),-1))

        # Pooling
        for i in range(len(self.graph.part)):
            num_node= len(self.graph.part[i])
            x_i = x2[:, :, :, self.graph.part[i]]

            x_i = self.graph_max_pool(x_i, (1, num_node))

            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i

        x_sub_1 = x_sub1.view(N * len(self.graph.part), T, -1)
        x_sub_1 = self.trans_networks[1](x_sub_1)
        x_sub_1 = rearrange(x_sub_1, "(b j) t c -> b c t j", j=len(self.graph.part))

        x_sub1, _ = self.st_gcn_pool[0](x_sub_1.reshape(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())
        x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part))
        x_avg_pool2 = self.avg_pool(x_sub1.view(N, (T * len(self.graph.part)), -1))

        x_pool_1 = self.graph_max_pool(x_sub1, (1, len(self.graph.part)))

        x_sub_2 = x_pool_1.view(N * M, T, -1)
        x_sub_2 = self.trans_networks[2](x_sub_2)
        x_sub_2 = rearrange(x_sub_2, "(b j) t c -> b c t j", b=N)

        x_pool_1 = self.conv4(x_sub_2)

        x_up_sub = x_pool_1.repeat(1, 1, 1, len(self.graph.part))
        x_up_sub = torch.matmul(x_avg_pool2.view(N,1,T,len(self.graph.part)),x_up_sub)

        x_up_sub = self.conv2(x_up_sub)

        x_up = torch.zeros((N * M, fc_unit, T, V)).cuda()
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_up[:, :, :, self.graph.part[i]] = x_up_sub[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)


        x = torch.matmul(x_avg_pool1.view(N, 1, T, V), x_up.permute(0, 1, 3, 2).contiguous())
        x_out = torch.zeros((N * M, fc_unit, T, V)).cuda()
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_out[:, :, :, self.graph.part[i]] = x[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)

        x = self.fcn(x_out)
        x = x.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1).contiguous()
        return x

class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.05,
                 residual=True):

        super().__init__()
        self.inplace = True

        self.momentum = 0.1
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)

        self.tcn = nn.Sequential(

            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.05),
            nn.Conv2d(
                out_channels,
                out_channels,
                (1, 1),
                (stride, 1),
                padding = 0,
            ),
            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.Dropout(dropout, inplace=self.inplace),


        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
            )

        self.relu = nn.ReLU(inplace=self.inplace)

    def forward(self, x, A):

        res = self.residual(x)

        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        return self.relu(x), A
