#!/usr/bin/env python
# coding:utf-8

import xml.dom.minidom
from tqdm import tqdm
import json
import re

"""
NYTimes Reference: https://catalog.ldc.upenn.edu/LDC2008T19
"""

sample_ratio = 0.02
train_ratio = 0.7
min_per_node = 200

month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
year_list = [1987, 2007]
ROOT_DIR = 'Nytimes/'
label_f = 'nyt_label.vocab'
english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z(),!?\.\'\`]", " ", string)
    string = re.sub(r"/", " ", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", "  ", string)
    string = re.sub(r"!", "  ", string)
    string = re.sub(r"\(", "  ", string)
    string = re.sub(r"\)", "  ", string)
    string = re.sub(r"\?", "  ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 2003-07
def read_nyt(id_json):
    f = open(id_json, 'r')
    ids = f.readlines()
    f.close()
    print(ids[:2])
    corpus_list = []
    f = open(label_f, 'r')
    label_vocab_s = f.readlines()
    f.close()
    label_vocab = []
    for label in label_vocab_s:
        label = label.strip()
        label_vocab.append(label)
    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus = []

    for file_name in tqdm(ids):
        xml_path = file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                continue
            text = clean_str(text)
            text = [word.lower() for word in text.split() if word not in english_stopwords]
            sample['doc_token'] = text
            sample_label = []
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label == 'Top':
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
            sample['doc_label'] = sample_label
            sample['doc_topic'] = []
            sample['doc_keyword'] = []
            corpus.append(sample)
        except:
            print(xml_path)
            print('Something went wrong...')
            continue
    print(id_json, len(corpus))
    f = open('doc' + id_json, 'w')
    for line in corpus:
        doc = json.dumps(line) + '\n'
        f.write(doc)
    f.close()


if __name__ == '__main__':
    read_nyt('idnewnyt_train.json')
    read_nyt('idnewnyt_val.json')
    read_nyt('idnewnyt_test.json')
