#!/usr/bin/env python
# coding: utf-8

import re
import tqdm
import helper.logger as logger
import json

english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                     'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                     "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                     "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                     'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                     'won', "won't", 'wouldn', "wouldn't", '\\.', '\\?', ',', '\\!', "'s", '']


def clean_stopwords(sample):
    """
    :param sample: List[Str], lower case
    :return:  List[Str]
    """
    return [token for token in sample if token not in english_stopwords]


def clean_str(string):
    """
    Original Source:  https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string: Str
    :return -> Str
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_line(sample):
    """
    :param sample: Str, "The sample would be tokenized and filtered according to the stopwords list"
    :return: token_list -> List[Str]
    """
    sample = clean_str(sample.lstrip().rstrip())
    token_list = clean_stopwords(sample.split(' '))
    return json.dumps({'token': token_list, 'label': []})


def preprocess_raw_file(file_path):
    """
    :param file_path: Str, file path of the raw file
    :return: List[Dict{'token': List[Str], 'label': []}]
    """
    corpus_data = list()
    raw_data = list()
    # sample_dict = {'token': [], 'label': []}
    logger.info('Loading and Preprocessing raw data in {}'.format(file_path))
    with open(file_path, 'r') as f:
        for line in tqdm.tqdm(f):
            sample_tokens = preprocess_line(line)
            raw_data.append({'token': line.rstrip(), 'label': []})
            corpus_data.append(json.dumps({'token': sample_tokens, 'label': []}))
    logger.info('The number of samples: {}'.format(len(corpus_data)))
    return raw_data, corpus_data


def load_processed_file(file_path):
    """
    :param file_path: Str, file path of the processed file
    :return: List[Dict{'token': List[Str], 'label': []}]
    """
    corpus_data = list()
    raw_data = list()
    # sample_dict = {'token': [], 'label': []}
    logger.info('Loading raw data in {}'.format(file_path))
    with open(file_path, 'r') as f:
        for line in tqdm.tqdm(f):
            raw_data.append(json.loads(line.rstrip()))
            corpus_data.append(line.rstrip())
    logger.info('The number of samples: {}'.format(len(corpus_data)))
    return raw_data, corpus_data
