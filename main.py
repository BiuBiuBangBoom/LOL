from model import GTransformerLayer
import torch as th
import torch.nn as nn
import dgl
from dgl.data import FB15k237Dataset

num_rel_types = 237 * 2


def main():
    dataset = FB15k237Dataset()
    graph = dataset[0]
    device = th.device('cuda:0')
    # graph = graph.to(device)
    model = GTransformerLayer(100, 100, 8, graph.num_nodes(), graph.num_edges(), num_rel_types)
    model.to(device)
    model(graph)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
