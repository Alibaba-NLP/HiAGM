#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
from models.structure_model.graphcnn import HierarchyGCN
from models.structure_model.tree import Tree
import json
import os
import numpy as np
from helper.utils import get_hierarchy_relations
from models.structure_model.weighted_tree_lstm import WeightedHierarchicalTreeLSTMEndtoEnd

MODEL_MODULE = {
    'TreeLSTM': WeightedHierarchicalTreeLSTMEndtoEnd,
    'GCN': HierarchyGCN
}


class StructureEncoder(nn.Module):
    def __init__(self,
                 config,
                 label_map,
                 device,
                 graph_model_type):
        """
        Structure Encoder module
        :param config: helper.configure, Configure Object
        :param label_map: data_modules.vocab.v2i['label']
        :param device: torch.device, config.train.device_setting.device
        :param graph_model_type: Str, model_type, ['TreeLSTM', 'GCN']
        """
        super(StructureEncoder, self).__init__()

        self.label_map = label_map
        self.root = Tree(-1)

        self.hierarchical_label_dict, self.label_trees = get_hierarchy_relations(os.path.join(config.data.data_dir,
                                                                                              config.data.hierarchy),
                                                                                 self.label_map,
                                                                                 root=self.root,
                                                                                 fortree=True)
        hierarchy_prob_file = os.path.join(config.data.data_dir, config.data.prob_json)
        f = open(hierarchy_prob_file, 'r')
        hierarchy_prob_str = f.readlines()
        f.close()
        self.hierarchy_prob = json.loads(hierarchy_prob_str[0])
        self.node_prob_from_parent = np.zeros((len(self.label_map), len(self.label_map)))
        self.node_prob_from_child = np.zeros((len(self.label_map), len(self.label_map)))

        for p in self.hierarchy_prob.keys():
            if p == 'Root':
                continue
            for c in self.hierarchy_prob[p].keys():
                # self.hierarchy_id_prob[self.label_map[p]][self.label_map[c]] = self.hierarchy_prob[p][c]
                self.node_prob_from_child[int(self.label_map[p])][int(self.label_map[c])] = 1.0
                self.node_prob_from_parent[int(self.label_map[c])][int(self.label_map[p])] = self.hierarchy_prob[p][c]
        #  node_prob_from_parent: row means parent, col refers to children

        self.model = MODEL_MODULE[graph_model_type](num_nodes=len(self.label_map),
                                                    in_matrix=self.node_prob_from_child,
                                                    out_matrix=self.node_prob_from_parent,
                                                    in_dim=config.structure_encoder.node.dimension,
                                                    dropout=config.structure_encoder.node.dropout,
                                                    device=device,
                                                    root=self.root,
                                                    hierarchical_label_dict=self.hierarchical_label_dict,
                                                    label_trees=self.label_trees)

    def forward(self, inputs):
        return self.model(inputs)
