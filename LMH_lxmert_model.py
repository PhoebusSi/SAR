import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import LxmertTokenizer, LxmertModel
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier

from torch.nn import functional as F
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random
from LMH_vqa_debias_loss_functions import LearnedMixin
class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', return_dict=True)
        self.model = nn.DataParallel(self.model)
        self.candi_ans_num = opt.train_candi_ans_num
        self.batchsize = opt.batch_size
        self.Linear_layer = nn.Linear(768, 1)
        norm = opt.norm                                                      
        activation = opt.activation                                  
        dropC = opt.dropC                                                         
        self.debias_loss_fn = LearnedMixin(0.36)
        self.classifier = SimpleClassifier(in_dim=768, hid_dim=2 * 768, out_dim=1,    
                                           dropout=dropC, norm=norm, act=activation)  

    def forward(self, qa_text, v, b, epo, name, bias, labels):
        """
        qa_text (btachsize, candi_ans_num, max_length)
        v (batchsize, obj_num, v_dim)
        b (batchsize, obj_num, b_dim)

        return: logits
        """
        qa_text = qa_text.cuda()
        v= v.cuda()
        b= b.cuda()
        bias = bias.cuda()

        if name == 'train':
            self.candi_ans_num = self.opt.train_candi_ans_num
        elif name == 'test':
            self.candi_ans_num = self.opt.test_candi_ans_num
        qa_text_reshape = qa_text.reshape(qa_text.shape[0] * self.candi_ans_num, -1)

        v_repeat = v.repeat(1, self.candi_ans_num, 1)
        v_reshape = v_repeat.reshape( v.shape[0] * self.candi_ans_num,v.shape[1], v.shape[2] )
        b_repeat = b.repeat(1, self.candi_ans_num , 1)
        b_reshape = b_repeat.reshape( b.shape[0] * self.candi_ans_num,b.shape[1], b.shape[2] )

        
        outputs = self.model(qa_text_reshape, v_reshape, b_reshape)
        pool_out = outputs.pooled_output
        
        logits = self.classifier(pool_out)
        logits_reshape = logits.reshape(-1, self.candi_ans_num)
        pool_out_reshape = pool_out.reshape(v.shape[0], self.candi_ans_num, -1)  
        
        if labels is not None:
            loss = self.debias_loss_fn(torch.mean(pool_out_reshape,dim=1,keepdim=False), logits_reshape,bias, labels)
        else:
            loss = None
        return logits_reshape, loss



