#!/usr/bin/env python
# coding: utf-8

import torch


class Collator(object):
    def __init__(self, config, vocab):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(Collator, self).__init__()
        self.device = config.train.device_setting.device
        self.label_size = len(vocab.v2i['label'].keys())

    def _multi_hot(self, batch_labels):
        """
        :param batch_labels: label idx list of one batch, List[List[int]], e.g.  [[1,2],[0,1,3,4]]
        :return: multi-hot value for classification -> List[List[int]], e.g. [[0,1,1,0,0],[1,1,0,1,1]
        """
        batch_size = len(batch_labels)
        max_length = max([len(sample) for sample in batch_labels])
        aligned_batch_labels = []
        for sample_label in batch_labels:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
        batch_labels_multi_hot = torch.zeros(batch_size, self.label_size).scatter_(1, aligned_batch_labels, 1)
        return batch_labels_multi_hot

    def __call__(self, batch):
        """
        transform data for training
        :param batch: Dict{'token': List[List[int]],
                           'label': List[List[int]],
                            'token_len': List[int]}
        :return: batch -> Dict{'token': torch.FloatTensor,
                               'label': torch.FloatTensor,
                               'token_len': torch.FloatTensor,
                               'label_list': List[List[int]]}
        """
        batch_token = []
        batch_label = []
        batch_doc_len = []
        for sample in batch:
            batch_token.append(sample['token'])
            batch_label.append(sample['label'])
            batch_doc_len.append(sample['token_len'])

        batch_token = torch.tensor(batch_token)
        batch_multi_hot_label = self._multi_hot(batch_label)
        batch_doc_len = torch.FloatTensor(batch_doc_len)
        return {
            'token': batch_token,
            'label': batch_multi_hot_label,
            'token_len': batch_doc_len,
            'label_list': batch_label
        }
