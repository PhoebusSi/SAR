import sys
from collections import defaultdict, Counter
from transformers import LxmertTokenizer, LxmertModel
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
#R
from SAR_replace_dataset_vqacp import Dictionary, VQAFeatureDataset
#C
#from SAR_concatenate_dataset_vqacp import Dictionary, VQAFeatureDataset

from LMH_lxmert_model import Model as LXM_Model
from lxmert_model import Model
import utils
import opts_SAR as opts
from SAR_train import train


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        random.seed(seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    if int(opt.lp) == 0:
        model = Model(opt)
    elif int(opt.lp) == 1:
        model = Model(opt)
    elif int(opt.lp) == 2:
        model = LXM_Model(opt)
    else:
        print("opt.lp has to be selected in [0,1,2]")
        assert 0 == 1
    model = model.cuda()
    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False,opt=opt)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False,opt=opt)
    answer_voc_size = opt.ans_dim#

    # Compute the bias:
    # The bias here is just the expected score for each answer/question type

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            condi_top20_prob_array = np.zeros(20, np.float32)
            for i in range(len(condi_top20_prob_array)):
                condi_top20_prob_array[i] = question_type_to_prob_array[q_type][ex['condi_ans']['top20'][i]]
            ex['bias'] = condi_top20_prob_array

    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=0)#1, collate_fn=utils.trim_collate)
    opt.use_all = 1
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=0)#1, collate_fn=utils.trim_collate)


    train(model, train_loader, eval_loader, opt)
