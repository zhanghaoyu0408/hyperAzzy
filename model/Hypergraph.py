import numpy as np
import torch
from torch.nn import init
from torch import nn

def adj_from_Hedges(node_list,hedge_list):
    nodes = len(node_list)
    hedges = len(hedge_list)

    h_adj = np.zeros(shape=(nodes,hedges))
    for center_id,center_hedge in enumerate(hedge_list):
        for i in center_hedge:
            h_adj[i][center_id] = 1.0

    return h_adj

class HChebConv(nn.Module):
    def __init__(self, in_c, out_c, bias=True):
        super(HChebConv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(1, in_c, out_c))
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)


    def forward(self, inputs, graph):

        L = HChebConv.get_laplacian(graph)

        result = torch.matmul(L, inputs)

        result = torch.matmul(result, self.weight) + self.bias

        return result

    @staticmethod
    def get_laplacian(graph):
        H = np.array(graph)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T
        G = DV2 * H * W * invDE * HT * DV2
        return torch.tensor(G,dtype=torch.float32).cuda()

class Graph():
    def __init__(self,
                 layout,
                 strategy,
                 sym_base,
                 pad=0,
                 max_hop=1,
                 dilation=1):
        self.sym_base = sym_base
        self.max_hop = max_hop
        self.dilation = dilation
        self.seqlen = 2*pad+1    #3
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)

        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_distance_to_center(self,layout):
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        return dist_center

    def graph_link_between_frames(self,base):
        return [(front + i*self.num_node_each, back + i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]


    def basic_layout(self,neighbour_base, sym_base):
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link

    def get_edge(self, layout):
        if layout == 'hm36_gt':
            self.num_node_each = 17


            neighbour_base = [(0, 1), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5),
                              (7, 0), (8, 7), (9, 8), (10, 9), (11, 8),
                              (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)
                              ]

            self_link, time_link = self.basic_layout(neighbour_base, self.sym_base)

            self.la, self.ra =[11, 12, 13], [14, 15, 16]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            self.cb = [0, 7, 8, 9, 10]
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]
            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link
            self.center = 7

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        self.A = normalize_adjacency

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD