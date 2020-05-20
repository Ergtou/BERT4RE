# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import sys
import json
from typing import NamedTuple
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics
import checkpoint


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 30 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    labels_number: int = 54

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, train_data_iter, test_data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = train_data_iter # iterator to load data
        self.test_data_iter = test_data_iter
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(1):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                if batch==None:
                    break
                for k,v in batch.items():
                    if k == "entpair":
                        continue
                    batch[k]= torch.from_numpy(np.array(v)).long().to(self.device)
                self.optimizer.zero_grad()
                self.model.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()

                #iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                '''
                if global_step % self.cfg.save_steps == 0: # save
                    #self.save(global_step)
                    pass
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return
                '''

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        #self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        #results = [] # prediction results
        test_result = []
        pred_result = []
        tot_correct = 0
        tot_not_na_correct = 0
        tot = 0
        tot_not_na = 0
        entpair_tot = 0

        #iter_bar = tqdm(self.test_data_iter, desc='Iter (loss=X.XXX)')
        for i,batch in enumerate(self.test_data_iter):
            if batch==None:
                break
            for k,v in batch.items():
                if k == "entpair":
                    continue
                batch[k]= torch.from_numpy(np.array(v)).long().to(self.device)
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, label_pred, label_id, iter_logit= evaluate(model, batch) # accuracy to print
                
                iter_logit = iter_logit.cpu()
                iter_correct = (label_pred == label_id).sum()

                
                '''
                with open("./result.txt","w") as f:
                    for index in range(len(label_pred)):
                        f.write("{} {}\n".format(label_pred[index],label_id[index]))
                '''
                iter_not_na_correct = np.logical_and((label_pred == label_id).cpu(), (label_id != 0).cpu()).sum()
                #print(iter_correct.data, iter_not_na_correct.data)
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += label_id.shape[0]
                tot_not_na += (label_id != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                for idx in range(len(iter_logit)):
                    for rel in range(1, self.cfg.labels_number):
                        test_result.append({'score': iter_logit[idx][rel], 'flag': batch['multi_rel'][idx][rel]})
                        if batch['entpair'][idx] != "None#None":
                            pred_result.append({'score': float(iter_logit[idx][rel]), 'entpair': batch['entpair'][idx].encode('utf-8'), 'relation': rel})
                    entpair_tot += 1 

        #iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        print("test len is:{}".format(len(test_result)))
        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        print("sorted len is:{}".format(len(sorted_test_result)))
        prec = []
        recall = []
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.cfg.labels_number)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        return auc

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts


    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

