import os
import time
import itertools
from torch.autograd import Variable
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
import json
from torch.optim.lr_scheduler import MultiStepLR

   
# standard cross-entropy loss
def instance_bce(logits, labels):
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
def compute_TopKscore_with_logits(logits, labels,n):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(logits, dim=-1), k=n, dim=-1, sorted=True)
    logits_ind = top_ans_ind
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits_ind.view(-1, n), 1)
    scores = (one_hots * labels)
    scores = torch.max(scores, 1)[0].data
    return scores
def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss




def train(model, train_loader, eval_loader, opt):
    utils.create_dir(opt.output)
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=opt.weight_decay)

    
    logger = utils.Logger(os.path.join(opt.output, 'log.txt'))

    utils.print_model(model, logger)
    for param_group in optim.param_groups:
        param_group['lr'] = opt.learning_rate

    scheduler = MultiStepLR(optim, milestones=[100], gamma=0.8)

    scheduler.last_epoch = opt.s_epoch

    

    best_eval_score = 0
    for epoch in range(opt.s_epoch, opt.num_epochs):
        total_loss = 0
        total_norm = 0
        count_norm = 0
        train_score = 0
        t = time.time()
        N = len(train_loader.dataset)
        scheduler.step()

        for i, (v, b, a, _, qa_text, _, _, q_t, bias) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            a = a.cuda()
            bias = bias.cuda()
            qa_text = qa_text.cuda()
            rand_index = random.sample(range(0, opt.train_condi_ans_num), opt.train_condi_ans_num)
            qa_text = qa_text[:,rand_index,:]
            a = a[:,rand_index]
            bias = bias[:,rand_index]

            if opt.lp == 0:
                logits = model(qa_text, v, b, epoch, 'train')
                loss = instance_bce_with_logits(logits, a, reduction='mean')
            elif opt.lp == 1:
                logits = model(qa_text, v, b, epoch, 'train')
                loss_pos = instance_bce_with_logits(logits, a, reduction='mean')
                index = random.sample(range(0, v.shape[0]), v.shape[0])
                v_neg = v[index]
                b_neg = b[index]
                logits_neg = model(qa_text, v_neg, b_neg, epoch, 'train')
                self_loss = compute_self_loss(logits_neg, a)
                loss = loss_pos + opt.self_loss_weight * self_loss
            elif opt.lp == 2:
                logits, loss = model(qa_text, v, b, epoch, 'train', bias, a)
            else:
                assert 1==2
           
            loss.backward()

            total_norm += nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            count_norm += 1

            optim.step()
            optim.zero_grad()

            score = compute_score_with_logits(logits, a.data).sum()
            train_score += score.item()
            total_loss += loss.item() * v.size(0)

            if i != 0 and i % 100 == 0:
                print(
                    'training: %d/%d, train_loss: %.6f, train_acc: %.6f' %
                    (i, len(train_loader), total_loss / (i * v.size(0)),
                     100 * train_score / (i * v.size(0))))
        total_loss /= N
        if None != eval_loader:
            model.train(False)
            eval_score, bound = evaluate(model, eval_loader, opt)
            model.train(True)

        logger.write('\nlr: %.7f' % optim.param_groups[0]['lr'])
        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write(
            '\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm / count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))


        if (eval_loader is not None and eval_score > best_eval_score):
            if opt.lp == 0:
                model_path = os.path.join(opt.output, 'SAR'+str(opt.train_condi_ans_num)+'_best_model.pth')
            elif opt.lp == 1:
                model_path = os.path.join(opt.output, 'SAR_SSL'+str(opt.train_condi_ans_num)+'_best_model.pth')
            elif opt.lp == 2:
                model_path = os.path.join(opt.output, 'SAR_LMH'+str(opt.train_condi_ans_num)+'_best_model.pth')
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score
@torch.no_grad()
def evaluate(model, dataloader, opt):
    score = 0

    score_ini_num_list=[]
    for num in range(opt.test_condi_ans_num):
        score_ini_num_list.append(0)
    upper_bound = 0
    num_data = 0
    entropy = 0
    for i, (v, b, a, q_id, qa_text, _, _, q_t, bias) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        bias = bias.cuda()
        a = a.cuda()
        q_id = q_id.cuda()
        qa_text = qa_text.cuda()
        if opt.lp == 0:
            logits = model(qa_text, v, b, 0, 'test')
        elif opt.lp == 1:
            logits = model(qa_text, v, b, 0, 'test')
        elif opt.lp == 2:
            logits, _ = model(qa_text, v, b, 0, 'test', bias, a)
        pred = logits
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        for num in range(opt.test_condi_ans_num):
            batch_score_num = compute_TopKscore_with_logits(pred, a.cuda(), num+1).sum()
            score_ini_num_list[num] += batch_score_num.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    score_num_list = []
    for score_num in score_ini_num_list:
        score_num = score_num / len(dataloader.dataset)
        score_num_list.append(score_num)
    upper_bound = upper_bound / len(dataloader.dataset)

    return score, upper_bound#, entropy


def calc_entropy(att):  # size(att) = [b x v x q]
    sizes = att.size()
    eps = 1e-8
    # att = att.unsqueeze(-1)
    p = att.view(-1, sizes[1] * sizes[2])
    return (-p * (p + eps).log()).sum(1).sum(0)  # g

