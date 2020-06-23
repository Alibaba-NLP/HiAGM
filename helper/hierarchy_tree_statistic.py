#!/usr/bin/env python
# coding:utf-8

import os
from models.structure_model.tree import Tree
import json
import copy
from collections import defaultdict
from helper.configure import Configure
import sys

ROOT_LABEL = 'Root'


class DatasetStatistic(object):
    def __init__(self, config):
        """
        class for prior probability
        :param config: helper.configure, Configure object
        """
        super(DatasetStatistic, self).__init__()
        self.config = config
        self.root = Tree('Root')
        # label_tree => root
        self.label_trees = dict()
        self.label_trees[ROOT_LABEL] = self.root
        self.hierarchical_label_dict, self.label_vocab = self.get_hierar_relations_with_name(
            os.path.join(config.data.data_dir, config.data.hierarchy))
        self.level = 0
        self.level_dict = dict()
        self.init_prior_prob_dict = dict()

        # build tree structure for treelstm
        for parent in list(self.hierarchical_label_dict.keys()):
            print(self.label_trees.keys())
            print(parent)
            assert parent in self.label_trees.keys()
            parent_tree = self.label_trees[parent]
            self.init_prior_prob_dict[parent] = dict()

            for child in self.hierarchical_label_dict[parent]:
                assert child not in self.label_trees.keys()
                self.init_prior_prob_dict[parent][child] = 0
                child_tree = Tree(child)
                parent_tree.add_child(child_tree)
                self.label_trees[child] = child_tree
        self.prior_prob_dict = copy.deepcopy(self.init_prior_prob_dict)
        self.total_train_prob_dict = copy.deepcopy(self.init_prior_prob_dict)

        for label in self.hierarchical_label_dict.keys():
            print(label, len(self.hierarchical_label_dict[label]))
            print(self.hierarchical_label_dict[label])
        for label in self.label_vocab:
            label_depth = self.label_trees[label].depth()
            if label_depth not in self.level_dict.keys():
                self.level_dict[label_depth] = [label]
            else:
                self.level_dict[label_depth].append(label)

        print(self.level)
        print(self.level_dict)
        for i in self.level_dict.keys():
            print(i, len(set(self.level_dict[i])))

    def get_taxonomy_file(self):
        """

        :return:
        """
        file = open(os.path.join(self.config.data.data_dir, self.config.data.hierarchy), 'r')
        data = file.readlines()
        file.close()
        hierarcy_dict = dict()
        for line in data:
            line = line.rstrip('\n')
            p = line.split('parent: ')[1].split(' ')[0]
            c = line.split('child: ')[1].split(' ')[0]
            if p not in hierarcy_dict.keys():
                hierarcy_dict[p] = [c]
            else:
                hierarcy_dict[p].append(c)
        print(hierarcy_dict)
        known_label = ['Root']
        output_lines = []
        while len(known_label):
            output_lines.append([known_label[0]] + hierarcy_dict[known_label[0]])
            for i in hierarcy_dict[known_label[0]]:
                if i in hierarcy_dict.keys():
                    known_label.append(i)
            known_label = known_label[1:]
        print(output_lines)
        file = open(os.path.join(self.config.data.data_dir, self.config.data.hierarchy), 'w')
        for i in output_lines:
            file.write('\t'.join(i) + '\n')
        file.close()

    @staticmethod
    def get_hierar_relations_with_name(taxo_file_dir):
        parent_child_dict = dict()
        label_vocab = []
        f = open(taxo_file_dir, 'r')
        relation_data = f.readlines()
        f.close()
        for relation in relation_data:
            # relation_list = relation.split()
            relation_list = relation.rstrip('\n').split('\t')
            parent, children = relation_list[0], relation_list[1:]
            assert parent not in parent_child_dict.keys()
            parent_child_dict[parent] = children
            label_vocab.extend(children)
            label_vocab.append(parent)
        print(parent_child_dict)
        return parent_child_dict, set(label_vocab)

    def get_data_statistic(self, file_name):
        all_label_num = 0
        label_num_dict = dict()
        f = open(file_name, 'r')
        data = f.readlines()
        f.close()
        count_data = len(data)
        sample_count_not_to_end = 0
        path_count_not_to_end = 0
        doc_length_all = 0
        label_doc_len_dict = dict()
        level_num_dict = defaultdict(int)
        for i in range(self.level + 1):
            level_num_dict[i] = 0
        prob_dict = copy.deepcopy(self.init_prior_prob_dict)

        for sample in data:
            sample_flag = False
            sample = json.loads(sample)
            sample_label = sample['label']
            all_label_num += len(sample_label)
            doc_length_all += len(sample['token'])
            # sample label : list of labels
            for label in sample_label:
                path_flag = False
                assert label in self.label_vocab
                level_num_dict[self.label_trees[label]._depth] += 1
                if label in self.init_prior_prob_dict.keys():
                    # TODO the children of Root node, need to be changed according to different corpus
                    if label in ["CCAT", "ECAT", "GCAT", "MCAT"]:
                        prob_dict[ROOT_LABEL][label] += 1
                        self.prior_prob_dict[ROOT_LABEL][label] += 1
                        if 'train' in file_name or 'val' in file_name:
                            self.total_train_prob_dict[ROOT_LABEL][label] += 1

                    for c in self.init_prior_prob_dict[label].keys():
                        if c in sample_label:
                            prob_dict[label][c] += 1
                            self.prior_prob_dict[label][c] += 1
                            if 'train' in file_name or 'val' in file_name:
                                self.total_train_prob_dict[label][c] += 1

                if label not in label_num_dict:
                    label_num_dict[label] = 1
                    label_doc_len_dict[label] = len(sample['token'])
                else:
                    label_num_dict[label] += 1
                    label_doc_len_dict[label] += len(sample['token'])

                if self.label_trees[label].num_children > 0 and not (sample_flag and path_flag):
                    # flag = False
                    for child in self.label_trees[label].children:
                        if child.idx in sample_label:
                            sample_flag = True
                            path_flag = True

                    if not path_flag:
                        path_count_not_to_end += 1
                    if not sample_flag:
                        sample_count_not_to_end += 1
                        # print(sample)
        avg_label_num = float(all_label_num) / count_data
        avg_doc_len = float(doc_length_all) / count_data

        for label in self.label_vocab:
            if label not in label_doc_len_dict.keys():
                label_doc_len_dict[label] = 0.0
            else:
                label_doc_len_dict[label] = float(label_doc_len_dict[label]) / label_num_dict[label]

        return {
            'num_of_samples': count_data,
            'average_label_num_per_sample': avg_label_num,
            'average_doc_length_per_sample': avg_doc_len,
            'label_num_dict': label_num_dict,
            'average_doc_length_per_label': label_doc_len_dict,
            'sample_end_before_leaf_nodes': sample_count_not_to_end,
            'path_end_before_leaf_nodes': path_count_not_to_end,
            'level_sample_number': level_num_dict,
            'prob_dict': prob_dict
        }


def prior_probs(prob_dict):
    for p in prob_dict.keys():
        total_sum = 0
        for c in prob_dict[p]:
            total_sum += prob_dict[p][c]
        if total_sum:
            for c in prob_dict[p]:
                prob_dict[p][c] = float(prob_dict[p][c]) / total_sum
    return prob_dict


if __name__ == '__main__':
    configs = Configure(config_json_file=sys.argv[1])

    rcv1_dataset_statistic = DatasetStatistic(configs)
    # rcv1_dataset_statistic.get_taxonomy_file()
    train_statistics = rcv1_dataset_statistic.get_data_statistic(
        os.path.join(configs.data.data_dir, configs.data.train_file))
    val_statistics = rcv1_dataset_statistic.get_data_statistic(
        os.path.join(configs.data.data_dir, configs.data.val_file))
    test_statistics = rcv1_dataset_statistic.get_data_statistic(
        os.path.join(configs.data.data_dir, configs.data.test_file))

    print('*****TRAIN*****')
    print(train_statistics)
    print(prior_probs(train_statistics['prob_dict']))
    print('*****val*****')
    print(val_statistics)
    print(prior_probs(val_statistics['prob_dict']))
    print('*****TEST*****')
    print(test_statistics)
    print(prior_probs(test_statistics['prob_dict']))

    # check_rcv1_level_data()
    print('*****TOTAL*****')
    print(rcv1_dataset_statistic.prior_prob_dict)
    print(prior_probs(rcv1_dataset_statistic.prior_prob_dict))
    print('*****TOTAL TRAIN*****')
    print(rcv1_dataset_statistic.total_train_prob_dict)
    print(prior_probs(rcv1_dataset_statistic.total_train_prob_dict))
    train_probs = prior_probs(rcv1_dataset_statistic.total_train_prob_dict)
    with open(os.path.join(configs.data.data_dir, configs.data.prob_json), 'w') as json_file:
        json_str = json.dumps(train_probs)
        json_file.write(json_str)
