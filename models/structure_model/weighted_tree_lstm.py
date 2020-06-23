#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class WeightedHierarchicalTreeLSTMEndtoEnd(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix, out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root=None,
                 hierarchical_label_dict=None,
                 label_trees=None):
        """
        TreeLSTM variant for Hierarchy Structure
        :param num_nodes: int, N
        :param in_matrix: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_matrix: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param layers: int, the number of layers <- config.structure_encoder.num_layer
        :param time_step: int, the number of time steps <- config.structure_encoder.time_step
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        :param root: Tree object of the root node
        :param hierarchical_label_dict: Dict{parent_id: child_id}
        :param label_trees: List[Tree]
        """
        super(WeightedHierarchicalTreeLSTMEndtoEnd, self).__init__()
        self.root = root
        mem_dim = in_dim // 2
        self.hierarchical_label_dict = hierarchical_label_dict
        self.label_trees = label_trees
        # child-sum
        self.bottom_up_lstm = WeightedChildSumTreeLSTMEndtoEnd(in_dim, mem_dim, num_nodes, in_matrix, device)
        # parent2child
        self.top_down_lstm = WeightedTopDownTreeLSTMEndtoEnd(in_dim, mem_dim, num_nodes, out_matrix, device)
        self.tree_projection_layer = torch.nn.Linear(2 * mem_dim, mem_dim)
        self.node_dropout = torch.nn.Dropout(dropout)
        self.num_nodes = num_nodes
        self.mem_dim = mem_dim

    def forward(self, inputs):
        """
        forward pass
        :param inputs: torch.FloatTensor, (batch, N, in_dim)
        :return: label_features -> torch.FloatTensor, (batch, N, in_dim)
        """
        inputs = inputs.transpose(0, 1)  # N, batch_size, dim
        for i in self.hierarchical_label_dict[self.root.idx]:
            self.bottom_up_lstm(self.label_trees[i + 1], inputs)
            self.top_down_lstm(self.label_trees[i + 1], inputs)

        tree_label_feature = []
        nodes_keys = list(self.label_trees.keys())
        nodes_keys.sort()
        for i in nodes_keys:
            if i == 0:  # should be root.idx
                continue
            tree_label_feature.append(
                torch.cat((self.node_dropout(self.label_trees[i].bottom_up_state[1].view(inputs.shape[1], 1, self.mem_dim)),
                           self.node_dropout(self.label_trees[i].top_down_state[1].view(inputs.shape[1], 1, self.mem_dim))),
                          2))
        label_feature = torch.cat(tree_label_feature, 1)  # label_feature: batch_size, num_nodes, 2 * node_dimension

        return label_feature


class WeightedChildSumTreeLSTMEndtoEnd(nn.Module):
    def __init__(self, in_dim, mem_dim,
                 num_nodes=-1, prob=None,
                 device=torch.device('cpu')):
        """
        Child-Sum variant for hierarchy-structure
        Child-Sum treelstm paper:Tai, K. S., Socher, R., & Manning, C. D. (2015).
            Improved semantic representations from tree-structured long short-term memory networks.
             arXiv preprint arXiv:1503.00075.
        :param in_dim: int, config.structure_encoder.dimension
        :param mem_dim: int, in_dim // 2
        :param num_nodes: int, the number of nodes in the hierarchy taxonomy
        :param prob: numpy.array, the prior probability of the hierarchical relation
        :param if_prob_train: Boolean, True for updating the prob
        :param device: torch.device  <- config.train.device_setting.device
        """
        super(WeightedChildSumTreeLSTMEndtoEnd, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.node_transformation = torch.nn.ModuleList()
        self.node_transformation_decompostion = torch.nn.ModuleList()
        self.prob = torch.Tensor(prob).to(device)
        self.prob = Parameter(self.prob)

    def node_forward(self, inputs, child_c, child_h):
        """
        forward pass of each node
        :param inputs: original state
        :param child_c: the current state of the child nodes
        :param child_h: the hidden state of the child nodes
        :return: c ( current state ) -> torch.FloatTensor (1, mem_dim),
                 h ( hidden state ) -> torch.FloatTensor (1, mem_dim)
        """
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(2) // 3, dim=2)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1, 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        """
        forward pass of the overall child-sum module
        :param tree: Tree object
        :param inputs: torch.FloatTensor, (N, batch, in_dim)
        :return: bottom_up_state -> torch.FloatTensor, (N, batch, mem_dim)
        """
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
            child_h = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
        else:
            child_c, child_h = zip(
                *map(lambda x: (self.prob[tree.idx][x.idx] * y for y in x.bottom_up_state), tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.bottom_up_state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.bottom_up_state


class WeightedTopDownTreeLSTMEndtoEnd(nn.Module):
    def __init__(self, in_dim, mem_dim,
                 num_nodes=-1, prob=None,
                 device=torch.device('cpu')):
        """
        Top-Down variant for hierarchy-structure
        Top-Down TreeLSTM paper: Zhang, X., Lu, L., & Lapata, M. (2015). Top-down tree long short-term memory networks.
            arXiv preprint arXiv:1511.00060.
        :param in_dim: int, config.structure_encoder.dimension
        :param mem_dim: int, in_dim // 2
        :param num_nodes: int, the number of nodes in the hierarchy taxonomy
        :param prob: numpy.array, the prior probability of the hierarchical relation
        :param if_prob_train: Boolean, True for updating the prob
        :param device: torch.device  <- config.train.device_setting.device
        """
        super(WeightedTopDownTreeLSTMEndtoEnd, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.node_transformation = torch.nn.ModuleList()
        self.node_transformation_decompostion = torch.nn.ModuleList()
        self.prob = torch.Tensor(prob).to(device)
        self.prob = Parameter(self.prob)

    def node_forward(self, inputs, parent_c, parent_h):
        """
        forward pass for each node
        forward pass of each node
        :param inputs: original state
        :param parent_c: the current state of the child nodes
        :param parent_h: the hidden state of the child nodes
        :return: c ( current state ) -> torch.FloatTensor (1, mem_dim),
                 h ( hidden state ) -> torch.FloatTensor (1, mem_dim)
        """
        iou = self.ioux(inputs) + self.iouh(parent_h)
        i, o, u = torch.split(iou, iou.size(2) // 3, dim=2)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(self.fh(parent_h) + self.fx(inputs).repeat(len(parent_h), 1, 1))
        fc = torch.mul(f, parent_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs, state=None, parent=None):
        """
        forward pass of the overall child-sum module
        :param tree: Tree object
        :param inputs: torch.FloatTensor, (N, batch, in_dim)
        :return: top_down_state -> torch.FloatTensor, (N, batch,  mem_dim)
        """
        if state is None:
            parent_c = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                    1)
            parent_h = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                    1)
        else:
            parent_c = self.prob[parent.idx][tree.idx] * state[0]
            parent_h = self.prob[parent.idx][tree.idx] * state[1]

        tree.top_down_state = self.node_forward(inputs[tree.idx], parent_c, parent_h)
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, tree.top_down_state, tree)
        return tree.top_down_state
