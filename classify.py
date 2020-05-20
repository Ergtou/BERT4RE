# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

#import tokenization
import models
import optim
import train
from data_loader import json_file_data_loader
import os

from utils import set_seeds, get_device, truncate_tokens_pair


class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels, word_vec_mat):
        super().__init__()
        self.transformer = models.Transformer(cfg, word_vec_mat)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)
        self.att = nn.Embedding(cfg.labels_number, cfg.dim)
        self.bias = nn.Parameter(torch.Tensor(n_labels))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.att.weight.data)
        nn.init.normal(self.bias)

    def __attention_train_logit__(self, x, query):
        current_relation = self.att(query)
        attention_logit=( current_relation * x ).sum(-1)
        return attention_logit

    def __attention_test_logit__(self, x):
        attention_logit = self.att.weight @ torch.transpose(x,0,1)
        attention_logit = torch.transpose(attention_logit,0,1)
        # [real_size, rel_numbers]
        return attention_logit

    def __logit__(self, x):
        attention_logit = self.att.weight @ torch.transpose(x,0,1)
        attention_logit = torch.transpose(attention_logit,0,1)
        #logit = attention_logit + bias
        return attention_logit + self.bias
        #return attention_logit

    def forward(self, input_ids, pos1, pos2, input_mask, query, scope, evaluate = False):
        h = self.transformer(input_ids, pos1, pos2, input_mask)
        # only use the first h in the sequence
        #pooled_h = self.activ(self.fc(h[:, 0]))
        pooled_h = self.fc(h[:, 0])
        
        if evaluate == False:
            attention_logit = self.__attention_train_logit__(pooled_h, query)
            bag_repre = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = pooled_h[scope[i][0]:scope[i][1]]
                attention_score = F.softmax(attention_logit[scope[i][0]:scope[i][1]], -1)
                number = attention_score.size()[0]
                bag_repre.append((attention_score.expand(1, number) @ bag_hidden_mat).squeeze() ) # (1, n') x (n', hidden_size) = (1, hidden_size) -> (hidden_size)
            bag_repre = torch.stack(bag_repre, dim = 0)
            logits = self.classifier(self.drop(bag_repre))
            return logits
        
        if evaluate == True:
            attention_logit = self.__attention_test_logit__(pooled_h) # (n, rel_tot)
            bag_repre = []
            bag_logit = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = pooled_h[scope[i][0]:scope[i][1]]
                attention_score = F.softmax(torch.transpose(attention_logit[scope[i][0]:scope[i][1], :],0,1), -1) # softmax of (rel_tot, n')
                bag_repre_for_each_rel = attention_score @ bag_hidden_mat # (rel_tot, n') \dot (n', hidden_size) = (rel_tot, hidden_size)
                bag_logit_for_each_rel = self.__logit__(bag_repre_for_each_rel) # -> (rel_tot, rel_tot)
                bag_repre.append(bag_repre_for_each_rel)
                bag_logit.append(F.softmax(bag_logit_for_each_rel, -1).diag()) # could be improved by sigmoid?
            bag_repre = torch.stack(bag_repre, dim = 0)
            bag_logit = torch.stack(bag_logit, dim = 0)
            return bag_logit

#pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
#pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',

def main(task='mrpc',
         train_cfg='config/train_mrpc.json',
         model_cfg='config/bert_base.json',
         data_file='../glue/MRPC/train.tsv',
         model_file=None,
         pretrain_file=None,
         data_parallel=True,
         vocab=None,
         save_dir='./ckpt/',
         max_len=128,
         mode='eval'):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    '''
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task) # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    '''
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.benchmark = True

    dataset_dir = os.path.join('./data', 'nyt')
    train_data_iter = json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                        os.path.join(dataset_dir, 'word_vec.json'),
                                        os.path.join(dataset_dir, 'rel2id.json'), shuffle=True, batch_size=cfg.batch_size)

    test_data_iter = json_file_data_loader(os.path.join(dataset_dir, 'test.json'), 
                                        os.path.join(dataset_dir, 'word_vec.json'),
                                        os.path.join(dataset_dir, 'rel2id.json'), shuffle=True, batch_size=cfg.batch_size)

    model = Classifier(model_cfg, cfg.labels_number, train_data_iter.word_vec_mat)
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg,
                            model,
                            train_data_iter,
                            test_data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    #if mode == 'train':
    def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
        input_ids, pos1, pos2, input_mask, label_id, query, scope = batch['word'], batch['pos1'], batch['pos2'], batch['mask'], batch['rel'], batch['ins_rel'], batch['scope']
        logits = model(input_ids, pos1, pos2, input_mask, query, scope, evaluate = False)
        loss = criterion(logits, label_id)
        return loss

    #elif mode == 'eval':
    def evaluate(model, batch):
        input_ids, pos1, pos2, input_mask, label_id, query, scope = batch['word'], batch['pos1'], batch['pos2'], batch['mask'], batch['rel'], batch['ins_rel'], batch['scope']
        logits = model(input_ids, pos1, pos2, input_mask, query, scope, evaluate = True)
        _, label_pred = logits.max(1)
        result = (label_pred == label_id).float() #.cpu().numpy()
        accuracy = result.mean()
        return accuracy, label_pred, label_id, logits



    best_auc = 0
    best_epo = 0
    for cur_epoch in range(cfg.n_epochs):
        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

        auc = trainer.eval(evaluate, model_file, data_parallel)
        print("### AUC of {} epoch is {} ###".format(cur_epoch, auc))

        if auc > best_auc:
            best_auc = auc
            best_epo = cur_epoch
            trainer.save(cur_epoch)
    #total_accuracy = torch.cat(results).mean().item()
    print('Best AUC is : {} , epoch is {}'.format(best_auc, best_epo))


if __name__ == '__main__':
    fire.Fire(main)
