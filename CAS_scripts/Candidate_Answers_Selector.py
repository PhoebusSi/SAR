import torch
import json
import torch.nn.functional as F
def compute_TopKscore_with_logits(logits, labels, n):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(logits, dim=-1), k=n, dim=-1, sorted=True)
    logits_ind = top_ans_ind
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits_ind.view(-1, n), 1)
    scores = (one_hots * labels)
    scores = torch.max(scores, 1)[0].data
    topN_scores = []
    for i in range(len(labels)):
        topN_scores.append(labels[i][logits_ind[i]])
    return scores, top_ans_ind, topN_scores



@torch.no_grad()
def evaluate(model, dataloader):
    '''
    When setting dataloader == train_dataloader, we can get the training set of 
    the Datasets for Answer Re-ranking module. 
    So does it for the test set.
    '''
    score = 0
    score20 = 0
    upper_bound = 0
    num_data = 0
    entropy = 0
    topN_dict_list = []
    for i, (v, b, q, a, q_id, v_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()

        q_id = q_id.cuda()
        
        v_id = v_id.cuda()
         
        pred, att = model(q, v, False)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        batch_score20, top20, top20_scores = compute_TopKscore_with_logits(pred, a.cuda(), 20)
        batch_score20 = batch_score20.sum()
        score20 += batch_score20.item() 
        for i in range(len(q)):
            topN_dict = {}
            topN_dict['question_id'] = q_id[i].cpu().numpy().tolist()
            topN_dict['image_id'] = v_id[i].cpu().numpy().tolist()
            topN_dict['top20'] = top20[i].cpu().numpy().tolist()
            topN_dict['top20_scores'] = top20_scores[i].cpu().numpy().tolist()
            topN_dict_list.append(topN_dict)

        
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)

        entropy += calc_entropy(att.data)
    
    json_str = json.dumps(topN_dict_list, indent=4)
    ##################################################################
    ### To build the Dataset for the Answer Re-ranking module      ###
    ### based on Visual Entailment.                                ###
    ### (build training and test sets, respectively)               ###
    ##################################################################
    #with open('./TrainingSet_top20_condidates.json', 'w') as json_file:
    with open('./TestSet_top20_condidates.json', 'w') as json_file:
        json_file.write(json_str)

    score = score / len(dataloader.dataset)
    score20 = score20 / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    print("score:",score)
    print("score20",score20)
    return score, upper_bound, entropy, json_str



