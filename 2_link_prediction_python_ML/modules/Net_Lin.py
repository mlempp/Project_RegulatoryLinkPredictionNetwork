import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj

kwarg = MessagePassing(flow = 'source_to_target', aggr = 'add')

class Net_Lin(torch.nn.Module):
    def __init__(self,kwarg, p=0, input_size=1, output_size=2):
        super(Net_Lin,self).__init__()
        self.p          = p
        self.drop1      = torch.nn.Dropout(p = p)
        self.linear1    = torch.nn.Linear(input_size,5,bias = True)
        self.linear2    = torch.nn.Linear(10,2, bias = True)
        
        
    def forward(self, data, flag):
        x, edge_index, edge_orig  = data.x, data.edge_index, data.edge_index_orig
        if flag == 'Training':
           edge_index1, _ = dropout_adj(edge_orig,p=self.p)   
           edge_index2, _ = dropout_adj(edge_orig,p=self.p)        
        else:
           edge_index1, _ = dropout_adj(edge_orig,p=0)
           edge_index2, _ = dropout_adj(edge_orig,p=0)

        x = self.linear1(x)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        
        z = x[edge_index[0,:],:]
        y = x[edge_index[1,:],:]
        
        x = torch.cat((y,z), 1)
        
        
        x2 = self.linear2(x)
        x = x2
        return x, x2
