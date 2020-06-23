#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.dataset import Dataset
import helper.logger as logger
import json
import os


def get_sample_position(corpus_filename, on_memory, corpus_lines, stage):
    """
    position of each sample in the original corpus File or on-memory List
    :param corpus_filename: Str, directory of the corpus file
    :param on_memory: Boolean, True or False
    :param corpus_lines: List[Str] or None, on-memory Data
    :param mode: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
    :return: sample_position -> List[int]
    """
    sample_position = [0]
    if not on_memory:
        print('Loading files for ' + stage + ' Dataset...')
        with open(corpus_filename, 'r') as f_in:
            sample_str = f_in.readline()
            while sample_str:
                sample_position.append(f_in.tell())
                sample_str = f_in.readline()
            sample_position.pop()
    else:
        assert corpus_lines
        sample_position = range(len(corpus_lines))
    return sample_position


class ClassificationDataset(Dataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=True, corpus_lines=None, mode="TRAIN"):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param stage: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
        :param on_memory: Boolean, True or False
        :param corpus_lines: List[Str] or None, on-memory Data
        :param mode: TRAIN / PREDICT, for loading empty label
        """
        super(ClassificationDataset, self).__init__()
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file)}
        self.config = config
        self.vocab = vocab
        self.on_memory = on_memory
        self.data = corpus_lines
        self.max_input_length = self.config.text_encoder.max_length
        self.corpus_file = self.corpus_files[stage]
        self.sample_position = get_sample_position(self.corpus_file, self.on_memory, corpus_lines, stage)
        self.corpus_size = len(self.sample_position)
        self.mode = mode

    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        return self.corpus_size

    def __getitem__(self, index):
        """
        sample from the overall corpus
        :param index: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
        if index >= self.__len__():
            raise IndexError
        if not self.on_memory:
            position = self.sample_position[index]
            with open(self.corpus_file) as f_in:
                f_in.seek(position)
                sample_str = f_in.readline()
        else:
            sample_str = self.data[index]
        return self._preprocess_sample(sample_str)

    def _preprocess_sample(self, sample_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
        """
        raw_sample = json.loads(sample_str)
        sample = {'token': [], 'label': []}
        for k in raw_sample.keys():
            if k == 'token':
                sample[k] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]
            else:
                sample[k] = []
                for v in raw_sample[k]:
                    if v not in self.vocab.v2i[k].keys():
                        logger.warning('Vocab not in ' + k + ' ' + v)
                    else:
                        sample[k].append(self.vocab.v2i[k][v])
        if not sample['token']:
            sample['token'].append(self.vocab.padding_index)
        if self.mode == 'TRAIN':
            assert sample['label'], 'Label is empty'
        else:
            sample['label'] = [0]
        sample['token_len'] = min(len(sample['token']), self.max_input_length)
        padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
        sample['token'] += padding
        sample['token'] = sample['token'][:self.max_input_length]
        return sample
