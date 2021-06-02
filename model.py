import torch as th
import torch.nn as nn

import math
import dgl
import dgl.function as fn


class GTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_relations, num_nodes, num_rel_types):
        super(GTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.num_rel_types = num_rel_types
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        # self.k_linears = nn.ModuleList()
        # self.q_linears = nn.ModuleList()
        # self.v_linears = nn.ModuleList()
        #
        # for t in range(self.num_relations):
        #     self.k_linears.append(nn.Linear(in_dim, out_dim))
        #     self.q_linears.append(nn.Linear(in_dim, out_dim))
        #     self.v_linears.append(nn.Linear(in_dim, out_dim))
        self.k_linears = []
        self.q_linears = []
        self.v_linears = []
        for i in range(self.num_heads):
            temp_k = []
            temp_q = []
            temp_v = []
            for j in range(self.num_heads):
                temp_k.append(nn.Linear(in_dim, out_dim))
                temp_q.append(nn.Linear(in_dim, out_dim))
                temp_v.append(nn.Linear(in_dim, out_dim))
            self.k_linears.append(temp_k)
            self.q_linears.append(temp_q)
            self.v_linears.append(temp_v)

        self.w_trans = nn.Linear(num_heads * out_dim, out_dim)

    def subgraph(self, g, r_type_idx):
        g.to(th.device('cpu'))
        sub_graph = dgl.edge_subgraph(g, r_type_idx).to(th.device('cuda:0'))
        g.to(th.device('cuda:0'))
        return sub_graph

    def forward(self, g):
        with g.local_scope():
            for relation_type in range(self.num_rel_types):
                r_type_idx = th.nonzero(g.edata['etype']==relation_type, as_tuple=False).squeeze()
                # print(g.edata['etype'])
                # print('第 %d 种关系的子图：'%relation_type)
                # print(r_type_idx)
                sub_graph = self.subgraph(g, r_type_idx)
                # print(sub_graph)
                # print(sub_graph.edata['etype'])


