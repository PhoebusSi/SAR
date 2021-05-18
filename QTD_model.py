import torch
import _pickle as cPickle
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)




class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
        num_hid = 128
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(opt.dataroot + 'glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='GRU')
        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim = num_hid//2 , out_dim= 2,#opt.test_candi_ans_num,
                                           dropout=dropC, norm=norm, act=activation)
        self.normal = nn.BatchNorm1d(num_hid,affine=False)

    def forward(self, q):
        q = self.tokenize(q)
        q =  torch.from_numpy(np.array(q)) 
        w_emb = self.w_emb(q.cuda())
        q_emb = self.q_emb(w_emb) 
        q_repr = self.q_net(q_emb) 
        batch_size = q.size(0)
        logits_pos = self.classifier(q_repr)
        return logits_pos
    def tokenize(self, q_text, max_length=14):
        """Tokenizes the questions.
 
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        token_list = []
        for q_iter in q_text:
            tokens = self.dictionary.tokenize(q_iter, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            assert len(tokens) ==  max_length
            token_list.append(tokens)
        return token_list
