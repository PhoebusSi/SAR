from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
from transformers import LxmertTokenizer, LxmertModel
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import zarr
import random
import pickle
COUNTING_ONLY = False


def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
                    'amount of' in q.lower() or \
                    'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


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


def _create_entry(img, question, answer, ans4reranker, label2ans):

    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
        ans4reranker.pop('image_id')
        ans4reranker.pop('question_id')
        if len(answer['labels']):
            answer['label_text'] = label2ans[answer['labels'][answer['scores'].index(max(answer['scores']))]]
            answer['label_all_text'] = ", ".join([label2ans[i] for i in answer['labels']] )
        else:
            answer['label_text'] = None
            answer['label_all_text'] = None
        candi_ans = {}
        candi_ans['top20'] = ans4reranker['top20']
        candi_ans['top20_scores'] = ans4reranker['top20_scores']
        top20_text = [label2ans[i] for i in candi_ans['top20']]
        candi_ans['top20_text'] = top20_text
            

    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'question_type': answer['question_type'],
        'answer': answer,
        'candi_ans' : candi_ans
        }
    return entry


def _load_dataset(dataroot, name, label2ans,ratio=1.0):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'test'
    """
    question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % (name))
    questions = sorted(json.load(open(question_path)),
            key=lambda x: x['question_id'])


    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]
    

    ans4reranker_path = os.path.join(dataroot, '%s_top20_candidates.json'%name)
    #ans4reranker_path = os.path.join('data4VE/%s_dataset4VE_demo.json'%name)
    ans4reranker = sorted(json.load(open(ans4reranker_path)),
            key=lambda x: x['question_id'])
    
    ans_mean_len = 0
    ques_num = 0
    for i in answers:
        ans_mean_len = ans_mean_len + len(i['labels'])
        ques_num = ques_num + 1 
    utils.assert_eq(len(questions), len(answers))
    utils.assert_eq(len(ans4reranker), len(answers))

    if ratio < 1.0:
        index = random.sample(range(0,len(questions)), int(len(questions)*ratio))
        questions_new = [questions[i] for i in index]
        answers_new = [answers[i] for i in index]
        ans4reranker_new = [ans4reranker[i] for i in index]
    else:
        questions_new = questions
        answers_new = answers
        ans4reranker_new = ans4reranker
    entries = []
    tongji = {}
    tongji_ques = {}
    for question, answer, ans4reranker in zip(questions_new, answers_new, ans4reranker_new):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        utils.assert_eq(question['image_id'], ans4reranker['image_id'])
        utils.assert_eq(question['image_id'], ans4reranker['image_id'])
        img_id = question['image_id']

        if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
            new_entry = _create_entry(img_id, question, answer, ans4reranker, label2ans)
            ans_word = new_entry['answer']['label_text']
            if ans_word not in tongji.keys():
                tongji[ans_word] = 1
            else:
                tongji[ans_word] = tongji[ans_word] + 1
            entries.append(new_entry)
            que_word = " ".join(new_entry['question'].split()[:2])
            if que_word not in tongji_ques.keys():
                tongji_ques[que_word] = 1
            else:
                tongji_ques[que_word] = tongji_ques[que_word] + 1


    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot, image_dataroot, ratio, adaptive=False, opt=None):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'test']
        ans2label_path = os.path.join(dataroot, 'cache', 'train_test_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'train_test_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        
        if name == "train":
            self.candi_ans_num = opt.train_candi_ans_num
            self.num_ans_candidates = opt.train_candi_ans_num
        elif name == "test":
            self.candi_ans_num = opt.test_candi_ans_num
            self.num_ans_candidates = opt.test_candi_ans_num

        self.dictionary = dictionary
        self.adaptive = adaptive

        print('loading image features and bounding boxes')
        # Load image features and bounding boxes
        self.features = zarr.open(os.path.join(image_dataroot, 'trainval.zarr'), mode='r')
        self.spatials = zarr.open(os.path.join(image_dataroot, 'trainval_boxes.zarr'), mode='r')
        
        
        
        self.v_dim = self.features[list(self.features.keys())[1]].shape[1]
        self.s_dim = self.spatials[list(self.spatials.keys())[1]].shape[1]
        is_exist = os.path.exists('data4VE/R_'+name+'_top20_densecaption_tokenizer_ids.pkl')
        if not is_exist:
            self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)
            self.tokenize(max_length=15, candi_ans_num=self.candi_ans_num)
            self.tensorize(name)
        else:
            fp = open('data4VE/R_'+name+"_top20_densecaption_tokenizer_ids.pkl","rb+")
            self.entries = pickle.load(fp)
    def tokenize(self, max_length=15, candi_ans_num=5):
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        for entry in self.entries:
            q_a_text_top20 = []
            question_text = entry['question']
            question_type_text = entry['question_type']
            ans_text_list = entry['candi_ans']['top20_text']
            for ind, i in enumerate(ans_text_list):
                lower_question_text = question_text.lower()
                if question_type_text in lower_question_text :
                    dense_caption = lower_question_text.replace(question_type_text,i)[:-1]
                else:
                    dense_caption = i+" "+lower_question_text
                dense_caption_token_dict = tokenizer(dense_caption)
                qa_tokens = dense_caption_token_dict['input_ids']
                if len(qa_tokens) > max_length :
                    qa_tokens = qa_tokens[:max_length]
                else:
                    padding = [tokenizer('[PAD]')['input_ids'][1:-1][0]]*(max_length - len(qa_tokens))
                    qa_tokens = qa_tokens + padding
                assert len(qa_tokens) == max_length 
                q_a_tokens_tensor = torch.from_numpy(np.array([qa_tokens]))
                if ind == 0:
                    q_a_tokens_top_20 = q_a_tokens_tensor
                else:
                    q_a_tokens_top_20 = torch.cat([q_a_tokens_top_20, q_a_tokens_tensor])
            entry['candi_ans']["20_qa_text"] = q_a_tokens_top_20

               
    def tensorize(self, name):
        for entry in self.entries:
            answer = entry['answer']
            candi_ans = entry['candi_ans']
            top20 = torch.from_numpy(np.array(candi_ans['top20']))
            entry['candi_ans']['top20'] = top20
            top20_scores = torch.from_numpy(np.array(candi_ans['top20_scores']))
            entry['candi_ans']['top20_scores'] = top20_scores
        with open('data4VE/R_'+name+'_top20_densecaption_tokenizer_ids.pkl', 'wb') as f:
            pickle.dump(self.entries, f)

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features[entry['image']]))
            spatials = torch.from_numpy(np.array(self.spatials[entry['image']]))

        question_text = entry['question']
        question_id = entry['question_id']
        answer = entry['answer']
        candi_ans = entry['condi_ans']

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            ans_type = answer['answer_type'] 
            target = candi_ans['top20_scores'][:self.candi_ans_num]
            qa_text = candi_ans['20_qa_text'][:self.candi_ans_num]
            topN_id = candi_ans['top20'][:self.candi_ans_num]
            LMH_bias = entry["bias"][:self.candi_ans_num] 
            return features, spatials, target, question_id, qa_text, topN_id, ans_type, question_text, LMH_bias#entry["bias"] 
        else:
            return features, spatials, question_id

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    dataroot = './data/vqacp2/'
    img_root = './data/coco/'
    dictionary = Dictionary.load_from_file(dataroot + 'dictionary.pkl')
    print(dictionary)
    train_dset = VQAFeatureDataset('train', dictionary, dataroot, img_root, ratio=1.0, adaptive=False)

    loader = DataLoader(train_dset, 256, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    for v, b, q, a, qid in loader:
        print(a.shape)
