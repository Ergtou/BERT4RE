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

#import tokenization
import models
import optim
import train
from data_loader import json_file_data_loader
import os

from utils import set_seeds, get_device, truncate_tokens_pair

class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b


def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'mrpc': MRPC, 'mnli': MNLI}
    return table[task]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


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

    def __attention_train_logit__(self, x, query):
        current_relation = self.att(query.long())
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
        return attention_logit

    def forward(self, input_ids, pos1, pos2, input_mask, query, scope, evaluate = False):
        h = self.transformer(input_ids, pos1, pos2, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))
        
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
                print(bag_logit_for_each_rel.shape)
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
         save_dir='../exp/bert/mrpc',
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
    dataset_dir = os.path.join('./data', 'nyt')
    data_iter = json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                        os.path.join(dataset_dir, 'word_vec.json'),
                                        os.path.join(dataset_dir, 'rel2id.json'), shuffle=True)

    model = Classifier(model_cfg, cfg.labels_number, data_iter.word_vec_mat)
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, pos1, pos2, input_mask, label_id, query, scope = batch['word'], batch['pos1'], batch['pos2'], batch['mask'], batch['rel'], batch['ins_rel'], batch['scope']
            logits = model(input_ids, pos1, pos2, input_mask, query, scope, evaluate = False)
            loss = criterion(logits, label_id.long())
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):
            input_ids, pos1, pos2, input_mask, label_id, query, scope = batch['word'], batch['pos1'], batch['pos2'], batch['mask'], batch['rel'], batch['ins_rel'], batch['scope']
            logits = model(input_ids, pos1, pos2, input_mask, query, scope, evaluate = True)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
