# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Pretrain transformer with Masked LM and Sentence Classification """

from random import randint, shuffle
from random import random as rand
import fire
import json
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import models
import optim
import train

from utils import set_seeds, get_device, get_random_word, truncate_tokens_pair

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline() # throw away an incomplete sentence


class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, train_file, test_file, batch_size, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.train_data = json.load(open(train_file, "r"))
        self.test_data = json.load(open(test_file,"r"))
        self.sentences = [example['sentence'] for example in self.train_data]
        self.sentences.extend([example['sentence'] for example in self.test_data])
        self.one_epoch_step = len(self.sentences) // batch_size +1

        self.data_index = 0

        print("Number of seqs is {}, one_epoch_steps is {}".format(len(self.sentences),self.one_epoch_step))

        if not os.path.exists('LM_data.txt'):
            with open("./LM_data.txt","w",encoding="utf-8") as f:
                for seq in self.sentences:
                    f.write(seq+"\n")
        
        self.f_pos = open("./LM_data.txt","r", encoding="utf-8")

        self.vocabulary = []
        for seq in self.sentences:
            self.vocabulary.extend(seq.split())
        self.vocabulary = list(set(self.vocabulary))

        self.word_to_id = {}

        self.word_tot = 0
        self.word_to_id['[PAD]'] = self.word_tot
        self.word_tot += 1
        for i in range(len(self.vocabulary)):
            self.word_to_id[self.vocabulary[i]] = self.word_tot
            self.word_tot +=1
        
        self.word_to_id['[UNK]'] = self.word_tot
        self.word_to_id['[BLANK]'] = self.word_tot +1
        self.word_to_id['[MASK]'] = self.word_tot +2
        self.word_to_id['[SEP]'] = self.word_tot +3
        self.word_to_id['[CLS]'] = self.word_tot +4

        self.vocabulary.append('[UNK]')
        self.vocabulary.append('[BLANK]')
        self.vocabulary.append('[MASK]')
        self.vocabulary.append('[SEP]')
        self.vocabulary.append('[CLS]')
        self.word_tot += 5

        print("Total {} words".format(self.word_tot))
        print("UNK id is {}".format(self.word_to_id['[UNK]']))
        print("CLS id is {}".format(self.word_to_id['[CLS]']))



        #self.f_pos = open(file, "r", encoding='utf-8', errors='ignore') # for a positive sample
        #self.f_neg = open(file, "r", encoding='utf-8', errors='ignore') # for a negative (random) sample
        #self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size
    
    def indexer(self, x):
        x_ids = []
        for word in x:
            if word in self.vocabulary:
                x_ids.append(self.word_to_id[word])
            else:
                x_ids.append(self.word_to_id['[UNK]'])
        return x_ids

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            #tokens.extend(self.tokenize(line.strip()))
            tokens.extend(line.strip().split())
        return tokens

    def __iter__(self): # iterator to load data
        while True:
            batch = []
            for i in range(self.batch_size):
                # sampling length of each tokens_a and tokens_b
                # sometimes sample a short sentence to match between train and test sequences
                '''
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob \
                    else int(self.max_len / 2)
                '''
                len_tokens = self.max_len

                is_next = rand() < 0.5 # whether token_b is next to token_a or not

                #tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
                tokens_a = []
                while len(tokens_a) < self.max_len:
                    tokens_a.extend(self.sentences[self.data_index % len(self.sentences)].strip().split())
                    self.data_index += 1

                #seek_random_offset(self.f_neg)
                # = self.f_pos if is_next else self.f_neg
                #tokens_b = self.read_tokens(f_next, len_tokens, False)
 
                if tokens_a is None: # end of file
                    self.f_pos.seek(0, 0) # reset file pointer
                    return

                instance = (is_next, tokens_a)
                for proc in self.pipeline:
                    instance = proc(instance, self.vocabulary, self.indexer)

                batch.append(instance)

            # To Tensor
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            #batch_tensors = [ x for x in zip(*batch)]
            yield batch_tensors
            #yield batch


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, max_len=512):
        super().__init__()
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = None # vocabulary (sub)words
        self.indexer = None # function from token to token index
        self.max_len = max_len

    def __call__(self, instance, vocab, indexer):
        self.vocab_words = vocab
        self.indexer = indexer

        is_next, tokens_a = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        #truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a
        tokens = tokens[:self.max_len]
        #segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5: # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        #print(tokens)
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)
        #print(masked_ids)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        #print(input_ids)
        #segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        #return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)
        return (input_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg, word_tot):
        super().__init__()
        self.transformer = models.Transformer(cfg, word_tot)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.dim, 2)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, input_mask)
        pooled_h = self.fc(h[:, 0])
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        #logits_clsf = self.classifier(pooled_h)

        return logits_lm


def main(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='../tbc/books_large_all.txt',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='./LM',
         log_dir='./LM',
         max_len=512,
         max_pred=20,
         mask_prob=0.15):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    max_len = model_cfg.max_len

    set_seeds(cfg.seed)

    dataset_dir = os.path.join('./data', 'nyt')
    train_file = os.path.join(dataset_dir, 'train.json')
    test_file = os.path.join(dataset_dir, 'test.json')

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
    #                                list(tokenizer.vocab.keys()),
    #                                tokenizer.convert_tokens_to_ids,
                                    max_len)]

    print("Pipeline Over")

    data_iter = SentPairDataLoader(train_file,
                                    test_file,
                                   cfg.batch_size,
                                   max_len,
                                   pipeline=pipeline)

    print("Data_iter Over")

    model = BertModel4Pretrain(model_cfg,data_iter.word_tot)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim4GPU(cfg, model, data_iter.one_epoch_step * cfg.n_epochs)
    trainer = train.Trainer(cfg, model, data_iter, None, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        #input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch
        input_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        #logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        logits_lm = model(input_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        #loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            #'loss_clsf': loss_clsf.item(),
                            #'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           global_step)
        #return loss_lm + loss_clsf
        return loss_lm
    print("Start training")
    for cur_epoch in range(cfg.n_epochs):
        trainer.train(get_loss, model_file, None, data_parallel)
        print("{} epoch is Done".format(cur_epoch))


if __name__ == '__main__':
    fire.Fire(main)
