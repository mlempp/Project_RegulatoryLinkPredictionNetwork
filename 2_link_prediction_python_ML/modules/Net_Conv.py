import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj

kwarg = MessagePassing(flow = 'source_to_target', aggr = 'add')

class Net_Conv(torch.nn.Module):
    def __init__(self,kwarg, p=0, input_size=1, output_size=2):
        super(Net_Conv,self).__init__()
        self.p          = p
        self.drop1      = torch.nn.Dropout(p = p)
        self.conv1      = GCNConv(input_size,5,kwarg, cached = True)
        self.linear1    = torch.nn.Linear(int(input_size)*2,4,bias = True)
        self.linear2    = torch.nn.Linear(10,2, bias = True)
        
    def forward(self, data, flag):
        x, edge_index, edge_orig  = data.x, data.edge_index, data.edge_index_orig
        if flag == 'Training':
           edge_index1, _ = dropout_adj(edge_orig,p=self.p)   
           edge_index2, _ = dropout_adj(edge_orig,p=self.p)        
        else:
           edge_index1, _ = dropout_adj(edge_orig,p=0)
           edge_index2, _ = dropout_adj(edge_orig,p=0)

        x = self.conv1(x, edge_index1)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        
        z = x[edge_index[0,:],:]
        y = x[edge_index[1,:],:]
        
        x = torch.cat((y,z), 1)
    
        
        x2 = self.linear2(x)
        x = x2
        return x, x2