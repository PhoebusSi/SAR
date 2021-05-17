"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import zarr
import random
COUNTING_ONLY = False

class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot, image_dataroot, ratio, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'test']
        ans2label_path = os.path.join(dataroot, 'cache', 'train_test_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'train_test_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        print('loading image features and bounding boxes')
        # Load image features and bounding boxes
        self.features = zarr.open(os.path.join(image_dataroot, 'trainval.zarr'), mode='r')
        self.spatials = zarr.open(os.path.join(image_dataroot, 'trainval_boxes.zarr'), mode='r')
        
        
        
        self.v_dim = self.features[list(self.features.keys())[1]].shape[1]
        self.s_dim = self.spatials[list(self.spatials.keys())[1]].shape[1]
        print('loading image features and bounding boxes done!')

        self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            
    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                   
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None
    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features[entry['image']]))
            spatials = torch.from_numpy(np.array(self.spatials[entry['image']]))

        question = entry['q_token']
        question_id = entry['question_id']
        image_id = entry['image_id']
        answer = entry['answer']

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target, question_id, image_id
        else:
            return features, spatials, question, question_id, image_id

    def __len__(self):
        return len(self.entries)


