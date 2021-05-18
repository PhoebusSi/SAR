import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from collections import defaultdict, Counter
import opts_SAR as opts
#R
#from SAR_replace_dataset_vqacp import Dictionary, VQAFeatureDataset
#C
from SAR_concatenate_dataset_vqacp import Dictionary, VQAFeatureDataset

from LMH_lxmert_model import Model as LXM_Model
from lxmert_model import Model
from QTD_model import Model as Model2
import utils



def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader, topN_id, all_N):
    _m, idx = p[:all_N].max(0)
    idx = topN_id[idx]
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, model2, dataloader, opt):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    batch_topN_id = torch.IntTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    all_N = []
    bar = progressbar.ProgressBar(maxval=N or None).start()
    counting2stop = 0
    count_l = [0]*20
    for v, b, a, q_id,qa_text, topN_id, ans_type, ques_text,bias in iter(dataloader):
        batch_N = []
        counting2stop=counting2stop+1
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        qa_text = qa_text.cuda()
        topN_id = topN_id.cuda()
        if opt.lp == 0:
            logits = model(qa_text, v, b, 0, 'test')
        elif opt.lp == 1:
            logits = model(qa_text, v, b, 0, 'test')
        elif opt.lp == 2:
            logits,_ = model(qa_text, v, b, 0, 'test',bias,a)
        mask = model2(ques_text)
        for i in mask:
            l = i.tolist()
            ind = l.index(max(l))
            count_l[ind]=count_l[ind]+1
            if ind == 0:
                #N' for yes/no question
                batch_N.append(opt.QTD_N4yesno)
            elif ind == 1:
                #N' for non-yes/no question
                batch_N.append(opt.QTD_N4non_yesno)
            else:
                assert 1==0

        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(q_id)
        all_N = all_N + batch_N
        batch_topN_id[idx:idx+batch_size].copy_(topN_id.data)

        idx += batch_size
    bar.update(idx)
    return pred, qIds, batch_topN_id, all_N


def make_json(logits, qIds, dataloader, topN_id, all_N):
    utils.assert_eq(logits.size(0), len(qIds))
 
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader, topN_id[i], all_N[i])
        results.append(result)
    return results

if __name__ == '__main__':
    opt = opts.parse_opt()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False,opt=opt)
    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False,opt=opt) 


    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device


    if int(opt.lp) == 0:
        model = Model(opt)
    elif int(opt.lp) == 1:
        model = Model(opt)
    elif int(opt.lp) == 2:
        model = LXM_Model(opt)
    model = model.cuda()
    model2 = Model2(opt)
    model2 = model2.cuda()
    answer_voc_size = opt.ans_dim#train_dset.num_ans_candidates
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
                condi_top20_prob_array[i]=question_type_to_prob_array[q_type][ex['condi_ans']['top20'][i]]
            ex['bias'] = condi_top20_prob_array

 
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    def process(args, model, model2, eval_loader):

        model_data = torch.load(opt.checkpoint_path4test)
        model2_data = torch.load(opt.checkpoint_path4test_QTDmodel)

        model.load_state_dict(model_data.get('model_state', model_data))
        model2.load_state_dict(model2_data.get('model_state', model2_data))
        model = nn.DataParallel(model).cuda()
        model2 = nn.DataParallel(model2).cuda()
        opt.s_epoch = model_data['epoch'] + 1

        model.train(False)
        model2.train(False)

        logits, qIds, topN_id, all_N = get_logits(model, model2, eval_loader, opt)
        results = make_json(logits, qIds, eval_loader, topN_id, all_N)
        model_label = opt.label 
        
        if opt.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % opt.s_epoch)
        
        utils.create_dir(opt.output)
        if 0 <= opt.s_epoch:
            model_label += '_epoch%d' % opt.s_epoch

        if opt.lp == 0:
            test_type = "-SAR"
        elif opt.lp == 1:
            test_type = "-SAR+SSL"
        elif opt.lp == 2:
            test_type = "-SAR+LMH"
        with open(opt.output+'/top'+str(opt.QTD_N4yesno)+'_'+str(opt.QTD_N4non_yesno)+test_type+'_answers_test_%s.json' \
            % (model_label), 'w') as f:
            json.dump(results, f)

    process(opt, model, model2, eval_loader)
