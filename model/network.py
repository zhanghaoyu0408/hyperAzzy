import torch
import torch.nn as nn
from model.st_gcn_multi_frame import GCN_Model
from einops import rearrange
from model.trans_hypothesis import Transformer as Transformer_hypothesis

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hypergraph1 = [(8,10),(9,7),(15,9),(7,12),(8,0),(3,0),(0,6)]
        self.gcn_model1 = GCN_Model(self.hypergraph1)

        self.hypergraph2 = [(0,9),(7,10),(15,12),(16,13),(2,5),(3,6)]
        self.gcn_model2 = GCN_Model(self.hypergraph2)

        self.hypergraph3 = [(0,10),(15,2),(16,3),(12,5),(13,6)]
        self.gcn_model3 = GCN_Model(self.hypergraph3)

        self.embedding_1 = nn.Sequential(
            nn.Conv1d(3 * 17, 512, kernel_size=1),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.embedding_2 = nn.Sequential(
            nn.Conv1d(3 * 17, 512, kernel_size=1),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.embedding_3 = nn.Sequential(
            nn.Conv1d(3 * 17, 512, kernel_size=1),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.Transformer_hypothesis = Transformer_hypothesis(length=5)

        self.regression = nn.Sequential(
            nn.BatchNorm1d(512*3, momentum=0.1),
            nn.Conv1d(512*3, 3*17, kernel_size=1)
        )

    def transformer_shape(self,x):
        B = x.shape[0]
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1,17,3)
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()
        return x

    def forward(self,x):

        x_1 = self.gcn_model1(x)
        x_1 = self.transformer_shape(x_1)
        x_2 = self.gcn_model1(x)
        x_2 = self.transformer_shape(x_2)
        x_3 = self.gcn_model1(x)
        x_3 = self.transformer_shape(x_3)

        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        x = self.Transformer_hypothesis(x_1, x_2, x_3)

        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=17).contiguous()
        return x

if __name__ == '__main__':
    x = torch.ones(size=(128,2,5,17,1)).cuda()
    model = Model().cuda()
    output = model(x)
    print(output.shape)








