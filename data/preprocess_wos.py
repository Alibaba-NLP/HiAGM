#!/usr/bin/env python
# coding:utf-8
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import re

"""
WoS Reference: https://github.com/kk7nc/HDLTex
"""

FILE_DIR = 'web-of-science-dataset/WebOfScience/Meta-data/Data.txt'
total_len = []
np.random.seed(7)

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
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

stats = {'Root': {'CS': 0, 'Medical': 0, 'Civil': 0, 'ECE': 0, 'biochemistry': 0, 'MAE': 0, 'Psychology': 0}, 'CS': {'Symbolic computation': 402, 'Computer vision': 432, 'Computer graphics': 412, 'Operating systems': 380, 'Machine learning': 398, 'Data structures': 392, 'network security': 445, 'Image processing': 415, 'Parallel computing': 443, 'Distributed computing': 403, 'Algorithm design': 379, 'Computer programming': 425, 'Relational databases': 377, 'Software engineering': 416, 'Bioinformatics': 365, 'Cryptography': 387, 'Structured Storage': 43}, 'Medical': {"Alzheimer's Disease": 368, "Parkinson's Disease": 298, 'Sprains and Strains': 142, 'Cancer': 359, 'Sports Injuries': 365, 'Senior Health': 118, 'Multiple Sclerosis': 253, 'Hepatitis C': 288, 'Weight Loss': 327, 'Low Testosterone': 305, 'Fungal Infection': 372, 'Diabetes': 353, 'Parenting': 343, 'Birth Control': 335, 'Heart Disease': 291, 'Allergies': 357, 'Menopause': 371, 'Emergency Contraception': 291, 'Skin Care': 339, 'Myelofibrosis': 198, 'Hypothyroidism': 315, 'Headache': 341, 'Overactive Bladder': 340, 'Irritable Bowel Syndrome': 336, 'Polycythemia Vera': 148, 'Atrial Fibrillation': 294, 'Smoking Cessation': 257, 'Lymphoma': 267, 'Asthma': 317, 'Bipolar Disorder': 260, "Crohn's Disease": 198, 'Idiopathic Pulmonary Fibrosis': 246, 'Mental Health': 222, 'Dementia': 237, 'Rheumatoid Arthritis': 188, 'Osteoporosis': 320, 'Medicare': 255, 'Psoriatic Arthritis': 202, 'Addiction': 309, 'Atopic Dermatitis': 262, 'Digestive Health': 95, 'Healthy Sleep': 129, 'Anxiety': 262, 'Psoriasis': 128, 'Ankylosing Spondylitis': 321, "Children's Health": 350, 'Stress Management': 361, 'HIV/AIDS': 358, 'Depression': 130, 'Migraine': 178, 'Osteoarthritis': 305, 'Hereditary Angioedema': 182, 'Kidney Health': 90, 'Autism': 309, 'Schizophrenia': 38, 'Outdoor Health': 2}, 'Civil': {'Green Building': 418, 'Water Pollution': 446, 'Smart Material': 363, 'Ambient Intelligence': 410, 'Construction Management': 412, 'Suspension Bridge': 395, 'Geotextile': 419, 'Stealth Technology': 148, 'Solar Energy': 384, 'Remote Sensing': 384, 'Rainwater Harvesting': 441, 'Transparent Concrete': 3, 'Highway Network System': 4, 'Nano Concrete': 7, 'Bamboo as a Building Material': 2, 'Underwater Windmill': 1}, 'ECE': {'Electric motor': 372, 'Satellite radio': 148, 'Digital control': 426, 'Microcontroller': 413, 'Electrical network': 392, 'Electrical generator': 240, 'Electricity': 447, 'Operational amplifier': 419, 'Analog signal processing': 407, 'State space representation': 344, 'Signal-flow graph': 274, 'Electrical circuits': 375, 'Lorentz force law': 44, 'System identification': 417, 'PID controller': 429, 'Voltage law': 54, 'Control engineering': 276, 'Single-phase electric power': 6}, 'biochemistry': {'Molecular biology': 746, 'Enzymology': 576, 'Southern blotting': 510, 'Northern blotting': 699, 'Human Metabolism': 622, 'Polymerase chain reaction': 750, 'Immunology': 652, 'Genetics': 566, 'Cell biology': 552, 'DNA/RNA sequencing': 14}, 'MAE': {'Fluid mechanics': 386, 'Hydraulics': 402, 'computer-aided design': 371, 'Manufacturing engineering': 346, 'Machine design': 420, 'Thermodynamics': 361, 'Materials Engineering': 289, 'Strength of materials': 335, 'Internal combustion engine': 387}, 'Psychology': {'Prenatal development': 389, 'Attention': 416, 'Eating disorders': 387, 'Borderline personality disorder': 376, 'Prosocial behavior': 388, 'False memories': 362, 'Problem-solving': 360, 'Prejudice': 389, 'Antisocial personality disorder': 368, 'Nonverbal communication': 394, 'Leadership': 350, 'Child abuse': 404, 'Gender roles': 395, 'Depression': 380, 'Social cognition': 397, 'Seasonal affective disorder': 365, 'Person perception': 391, 'Media violence': 296, 'Schizophrenia': 335}}
# 1-10 ['Electric motor', 'Satellite radio', 'Single-phase electric power']
# 4-4 ['Water Pollution', 'Bamboo as a Building Material', 'Underwater Windmill']
# 6-1 ['Cell biology', 'DNA/RNA sequencing']
# 4-10 ['Smart Material', 'Transparent Concrete', 'Nano Concrete']
# 1-9 ['Electrical generator', 'Analog signal processing']
# 4-1 ['Geotextile', 'Highway Network System']
# 5-7 ['Atrial Fibrillation', 'Depression']
# 5-10 ['Bipolar Disorder', 'Schizophrenia']
# 5-17 ['Digestive Health', 'Outdoor Health']

def get_data_from_meta():
    f = open(FILE_DIR, 'r')
    origin_txt = f.readlines()
    f.close()
    data = []
    label_check = {}
    for line in origin_txt[1:]:
        line = line.rstrip('\n')
        line = line.split('\t')
        assert len(line) == 7
        sample_label = [line[3].rstrip().lstrip(), line[4].rstrip().lstrip()]
        code = str(line[0]) + '-' + str(line[1])

        if code in label_check.keys():
            if sample_label[1] not in label_check[code]:
                label_check[code].append(sample_label[1])
        else:
            label_check[code] = [sample_label[1]]
        for i in label_check[code]:
            if stats[sample_label[0]][i] > stats[sample_label[0]][sample_label[1]]:
                sample_label[1] = i
                break
        # if sample_label[1] == 'Underwater Windmill':
        #     sample_label[1] = 'Water Pollution'
        # if sample_label[1] == 'Bamboo as a Building Material':
        #     sample_label[1] = 'Water Pollution'
        # if sample_label[1] == 'Nano Concrete':
        #     sample_label[1] = 'Smart Material'
        # if sample_label[1] == 'Highway Network System':
        #     sample_label[1] = 'Geotextile'
        # if sample_label[1] == 'Transparent Concrete':
        #     sample_label[1] = 'Smart Material'
        # if sample_label[1] == 'Outdoor Health':
        #     sample_label[1] = ''
        doc = line[6]
        doc = clean_str(doc)
        doc = [word.lower() for word in doc.split() if word not in english_stopwords and len(word) > 1]
        sample_text = doc
        total_len.append(len(sample_text))
        data.append({'doc_token': sample_text, 'doc_label': sample_label, 'doc_topic': [], 'doc_keyword': []})
    print(label_check)
    c = 0
    for i in label_check.keys():
        if len(label_check[i]) > 1:
            print(i, label_check[i])
            c += len(label_check[i]) - 1
    print(c)
    print(len(label_check.keys()))
    f = open('wos_total.json', 'w')
    for line in data:
        line = json.dumps(line)
        f.write(line + '\n')
    f.close()


def split_train_dev_test():
    f = open('wos_total.json', 'r')
    data = f.readlines()
    f.close()
    id = [i for i in range(46985)]
    np_data = np.array(data)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = list(train)
    val = list(val)
    test = list(test)
    f = open('wos_train.json', 'w')
    f.writelines(train)
    f.close()
    f = open('wos_test.json', 'w')
    f.writelines(test)
    f.close()
    f = open('wos_val.json', 'w')
    f.writelines(val)
    f.close()

    print(len(train), len(val), len(test))
    return


def get_hierarchy():
    f = open('wos_total.json', 'r')
    data = f.readlines()
    f.close()
    label_hierarchy = {}
    label_hierarchy['Root'] = []
    for line in data:
        line = line.rstrip('\n')
        line = json.loads(line)
        line = line['doc_label']
        if line[0] in label_hierarchy:
            if line[1] not in label_hierarchy[line[0]]:
                label_hierarchy[line[0]].append(line[1])
        else:
            label_hierarchy['Root'].append(line[0])
            label_hierarchy[line[0]] = [line[1]]
    f = open('wos.taxnomy', 'w')
    for i in label_hierarchy.keys():
        line = [i]
        line.extend(label_hierarchy[i])
        line = '\t'.join(line) + '\n'
        f.write(line)
    f.close()


if __name__ == '__main__':
    get_data_from_meta()
    get_hierarchy()
    split_train_dev_test()

