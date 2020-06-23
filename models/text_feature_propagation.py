#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn


class HiAGMTP(nn.Module):
    def __init__(self, config, label_map, graph_model, device):
        """
        Hierarchy-Aware Global Model : (Serial) Text Propagation Variant
         :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        :param graph_model: computational graph for graph model
        :param device: torch.device, config.train.device_setting.device
        """
        super(HiAGMTP, self).__init__()

        self.config = config
        self.device = device
        self.label_map = label_map

        self.graph_model = graph_model

        # linear transform
        self.transformation = nn.Linear(config.model.linear_transformation.text_dimension,
                                        len(self.label_map) * config.model.linear_transformation.node_dimension)

        # classifier
        self.linear = nn.Linear(len(self.label_map) * config.embedding.label.dimension,
                                len(self.label_map))

        # dropout
        self.transformation_dropout = nn.Dropout(p=config.model.linear_transformation.dropout)
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)

    def forward(self, text_feature):
        """
        forward pass of text feature propagation
        :param text_feature ->  torch.FloatTensor, (batch_size, K0, text_dim)
        :return: logits ->  torch.FloatTensor, (batch, N)
        """
        text_feature = torch.cat(text_feature, 1)
        text_feature = text_feature.view(text_feature.shape[0], -1)

        text_feature = self.transformation_dropout(self.transformation(text_feature))
        text_feature = text_feature.view(text_feature.shape[0],
                                         len(self.label_map),
                                         self.config.model.linear_transformation.node_dimension)

        label_wise_text_feature = self.graph_model(text_feature)

        logits = self.dropout(self.linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1)))
        return logits
