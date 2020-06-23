#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, config, vocab, device):
        """
        origin class of the classification model
        :param config: helper.configure, Configure object
        :param vocab: data_modules.vocab, Vocab object
        :param device: torch.device, config.train.device_setting.device
        """
        super(Classifier, self).__init__()

        self.config = config
        self.device = device
        self.linear = nn.Linear(len(config.text_encoder.CNN.kernel_size) * config.text_encoder.CNN.num_kernel,
                                   len(vocab.v2i['label'].keys()))
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)

    def forward(self, inputs):
        """
        forward pass
        :param inputs: torch.FloatTensor, (batch, len(CNNs) * top_k, num_kernels)
        :return: logits, torch.FloatTensor (batch, N)
        """
        token_output = torch.cat(inputs, 1)
        token_output = token_output.view(token_output.shape[0], -1)
        logits = self.dropout(self.linear(token_output))
        return logits

