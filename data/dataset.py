import torch as th
import torch.nn as nn
import dgl
from dgl.data import FB15k237Dataset


def main():
    dataset = FB15k237Dataset()
    graph = dataset[0]
    device = th.device('cuda:0')
    graph = graph.to(device)
    if th.cuda.is_available():
        print("avaliable")
    else:
        print("NOT!")
    train_mask = graph.edata['train_mask']
    # train_idxt = th.nonzero(train_mask, as_tuple=True)
    # train_idxf = th.nonzero(train_mask, as_tuple=False)
    train_idf = th.nonzero(train_mask, as_tuple=False).squeeze()
    train_set = th.arange(graph.number_of_edges())[train_mask]
    print(train_idf)

    # train_id = [train_idf, train_idxf, train_idxt]
    # for i in train_id:
    #     print(i)

    # print(graph)
    # print(graph.edges())
    # print(graph.canonical_etypes)
    # print(graph.srcdata['ntype'].size())
    # print(graph.dstdata)
    # print(graph.nodes())
    # print(graph.edges())
    # print(graph.ndata['ntype'])
    # print(graph.ndata['ntype'].size())
    # print(graph.edata['etype'].size())
    # print(graph.edata['etype'])
    # print(graph.etypes)
    # print(graph.ntypes)

    # num_nodes = graph.nodes().size()[0]
    # emb = nn.Embedding(num_nodes, 3)
    # graph.ndata['embedding'] = emb(th.tensor([i for i in range(graph.nodes().size()[0])])).cuda()
    # print(graph.ndata['embedding'])

    etype = graph.edata['etype']


if __name__ == '__main__':
    main()
