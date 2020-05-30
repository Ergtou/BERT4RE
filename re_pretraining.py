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
from re_pretraining_data_loader import json_file_data_loader
import os

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer

from utils import set_seeds, get_device, truncate_tokens_pair


class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels = 2):
        super().__init__()
        #self.transformer = models.Transformer(cfg, word_vec_mat)
        self.transformer = TinyBertForSequenceClassification.from_pretrained('./4L-312', num_labels= n_labels)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)
        self.bias = nn.Parameter(torch.Tensor(n_labels))

    def forward(self, input_ids, segment_ids, input_mask, evaluate = False):
        student_logits, student_atts, student_reps = self.transformer(input_ids, segment_ids, input_mask, is_student=True)
        # only use the first h in the sequence
        #pooled_h = self.activ(self.fc(h[:, 0]))
        pooled_h = self.activ(self.fc(student_reps[:, 0]))
        
        logits = self.classifier(self.drop(pooled_h))
        return logits

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
         save_dir='./pretrain/',
         max_len=512,
         mode='eval'):


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

    tokenizer = BertTokenizer.from_pretrained('./4L-312', do_lower_case=True)

    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.benchmark = True

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    dataset_dir = os.path.join('./data', 'nyt')
    train_data_iter = json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                        os.path.join(dataset_dir, 'word_vec.json'),
                                        os.path.join(dataset_dir, 'rel2id.json'), shuffle=True, batch_size= cfg.batch_size, tokenizer = tokenizer)

    test_data_iter = json_file_data_loader(os.path.join(dataset_dir, 'test.json'), 
                                        os.path.join(dataset_dir, 'word_vec.json'),
                                        os.path.join(dataset_dir, 'rel2id.json'), shuffle=True, batch_size= cfg.batch_size, test= True, tokenizer = tokenizer)

    right_steps = (train_data_iter.instance_tot // cfg.batch_size + 1) * cfg.n_epochs
    print("Total_ent_pair is {}, total_steps is {}".format(train_data_iter.instance_tot, right_steps))

    model = Classifier(model_cfg)
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg,
                            model,
                            train_data_iter,
                            test_data_iter,
                            optim.optim4GPU(cfg, model, right_steps),
                            save_dir, get_device())

    #if mode == 'train':
    def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
        input_ids, segment_ids, input_mask, label_id = batch['word'], batch['segment'], batch['mask'], batch['ins_rel']
        logits = model(input_ids, segment_ids, input_mask, evaluate = False)
        loss = criterion(logits, label_id)
        return loss

    def evaluate(model, batch):
        input_ids, segment_ids, input_mask, label_id = batch['word'], batch['segment'], batch['mask'], batch['ins_rel']
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
        result = (label_pred == label_id).float() #.cpu().numpy()
        accuracy = result.mean()
        return accuracy, result

    for cur_epoch in range(cfg.n_epochs):
        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

        results = trainer.pretrain_eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)
        #print("### AUC of {} epoch is {} ###".format(cur_epoch, auc))

        #if auc > best_auc:
        #    best_auc = auc
        #    best_epo = cur_epoch
        #    trainer.save(cur_epoch)
    #total_accuracy = torch.cat(results).mean().item()
    #print('Best AUC is : {} , epoch is {}'.format(best_auc, best_epo))


if __name__ == '__main__':
    fire.Fire(main)
