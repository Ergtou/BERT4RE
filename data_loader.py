from six import iteritems

import json
import os
import multiprocessing
import numpy as np
import random

import platform

class file_data_loader:
    def __next__(self):
        raise NotImplementedError
    
    def next(self):
        return self.__next__()

    def next_batch(self, batch_size):
        raise NotImplementedError


class json_file_data_loader(file_data_loader):
    MODE_INSTANCE = 0      # One batch contains batch_size instances.
    MODE_ENTPAIR_BAG = 1   # One batch contains batch_size bags, instances in which have the same entity pair (usually for testing).
    MODE_RELFACT_BAG = 2   # One batch contains batch size bags, instances in which have the same relation fact. (usually for training).

    def _load_preprocessed_file(self):
        sys_base = platform.system()

        if sys_base == "Windows":
            print("file name is {}".format(self.file_name))
            name_prefix = '.'.join(self.file_name.split('\\')[-1].split('.')[:-1])
            print("name_prefix is {}".format(name_prefix))
            word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('\\')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
        else:
            print("file name is {}".format(self.file_name))
            name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
            print("name_prefix is {}".format(name_prefix))
            word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            print("Not processed data dir")
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        segment_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_segment.npy')
        rel_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_rel.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        relfact2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_relfact2scope.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(rel_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(entpair2scope_file_name) or \
           not os.path.exists(segment_npy_file_name) or \
           not os.path.exists(relfact2scope_file_name):
            print("Missing some processed file")
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_segment = np.load(segment_npy_file_name)
        self.data_rel = np.load(rel_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.entpair2scope = json.load(open(entpair2scope_file_name))
        self.relfact2scope = json.load(open(relfact2scope_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, rel2id_file_name, shuffle=True, max_length=512, case_sensitive=False, reprocess=False, batch_size=160, test=False, tokenizer = None):
        '''
        file_name: Json file storing the data in the following format
            [
                {
                    'sentence': 'Bill Gates is the founder of Microsoft .',
                    'head': {'word': 'Bill Gates', ...(other information)},
                    'tail': {'word': 'Microsoft', ...(other information)},
                    'relation': 'founder'
                },
                ...
            ]
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        rel2id_file_name: Json file storing relation-to-id diction in the following format
            {
                'NA': 0
                'founder': 1
                ...
            }
            **IMPORTANT**: make sure the id of NA is 0!
        mode: Specify how to get a batch of data. See MODE_* constants for details.
        shuffle: Whether to shuffle the data, default as True. You should use shuffle when training.
        max_length: The length that all the sentences need to be extend to, default as 120.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        batch_size: The size of each batch, default as 160.
        '''

        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        #self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rel2id = json.load(open(rel2id_file_name))

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for i in range(len(self.ori_data)):
                    self.ori_data[i]['sentence'] = self.ori_data[i]['sentence'].lower()
                    self.ori_data[i]['head']['word'] = self.ori_data[i]['head']['word'].lower()
                    self.ori_data[i]['tail']['word'] = self.ori_data[i]['tail']['word'].lower()
                print("Finish eliminating")

            # Sort data by entities and relations
            print("Sort data...")
            self.ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
            print("Finish sorting")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = len(self.ori_data)
            self.entpair2scope = {} # (head, tail) -> scope
            self.relfact2scope = {} # (head, tail, relation) -> scope
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_segment = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_rel = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)

            last_entpair = ''
            last_entpair_pos = -1
            last_relfact = ''
            last_relfact_pos = -1
            for i in range(self.instance_tot):
                ins = self.ori_data[i]
                if ins['relation'] in self.rel2id:
                    self.data_rel[i] = self.rel2id[ins['relation']]
                else:
                    self.data_rel[i] = self.rel2id['NA']
                sentence = ' '.join(ins['sentence'].split()) # delete extra spaces
                head = ins['head']['word']
                tail = ins['tail']['word']

                ### Modified sentence ###
                #sentence = sentence.replace(head," SEP "+head+" SEP ")
                #sentence = sentence.replace(tail," SEP "+tail+" SEP ")
                #sentence = "CLS "+sentence
                seq_tokens = tokenizer.tokenize(sentence)
                #sentence = "CLS " + head + " SEP " + tail + " SEP " + sentence
                head_tokens = tokenizer.tokenize(head)
                tail_tokens = tokenizer.tokenize(tail)

                seq_tokens = ["[CLS]"] + head_tokens + ["[SEP]"] + tail_tokens + ["[SEP]"] + seq_tokens
                seq_tokens = seq_tokens[:max_length]

                input_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * len(input_ids)
                seq_length = len(input_ids)
                
                padding = [0] * (max_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                ###  ###

                cur_entpair = ins['head']['id'] + '#' + ins['tail']['id']
                cur_relfact = ins['head']['id'] + '#' + ins['tail']['id'] + '#' + ins['relation']
                if cur_entpair != last_entpair:
                    if last_entpair != '':
                        self.entpair2scope[last_entpair] = [last_entpair_pos, i] # left closed right open
                    last_entpair = cur_entpair
                    last_entpair_pos = i
                if cur_relfact != last_relfact:
                    if last_relfact != '':
                        self.relfact2scope[last_relfact] = [last_relfact_pos, i]
                    last_relfact = cur_relfact
                    last_relfact_pos = i
                
                self.data_segment[i] = np.array(segment_ids)
                self.data_word[i] = np.array(input_ids)
                self.data_length[i] = seq_length
                
                    
            if last_entpair != '':
                self.entpair2scope[last_entpair] = [last_entpair_pos, self.instance_tot] # left closed right open
            if last_relfact != '':
                self.relfact2scope[last_relfact] = [last_relfact_pos, self.instance_tot]

            print("Finish pre-processing")     

            print("Storing processed files...")
            name_prefix = '.'.join(os.path.split(file_name)[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_segment.npy'), self.data_segment)
            np.save(os.path.join(processed_data_dir, name_prefix + '_rel.npy'), self.data_rel)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.entpair2scope, open(os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json'), 'w'))
            json.dump(self.relfact2scope, open(os.path.join(processed_data_dir, name_prefix + '_relfact2scope.json'), 'w'))
            print("Finish storing")

        # Prepare for idx
        self.instance_tot = self.data_word.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = 0 # The number of relation facts, without NA.
        for key in self.relfact2scope:
            if key[-2:] != 'NA':
                self.relfact_tot += 1
        self.rel_tot = len(self.rel2id)


        ### Bag mode ###
        self.order = list(range(len(self.entpair2scope)))
        self.scope_name = []
        self.scope = []
        for key, value in iteritems(self.entpair2scope):
            self.scope_name.append(key)
            self.scope.append(value)
        
        '''
        elif self.mode == self.MODE_RELFACT_BAG:
            self.order = list(range(len(self.relfact2scope)))
            self.scope_name = []
            self.scope = []
            for key, value in iteritems(self.relfact2scope):
                self.scope_name.append(key)
                self.scope.append(value)
        else:
            raise Exception("[ERROR] Invalid mode")
        '''
        self.idx = 0

        if self.shuffle:
            random.shuffle(self.order) 

        print("Total relation fact: %d" % (self.relfact_tot))
        print("Total iins fact: %d" % (self.instance_tot))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, batch_size):
        if self.idx >= len(self.order):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.order) 
            #raise StopIteration
            return None

        batch_data = {}

        idx0 = self.idx
        idx1 = self.idx + batch_size
        if idx1 > len(self.order):
            idx1 = len(self.order)
        self.idx = idx1
        _word = []
        _mask = []
        _rel = []
        _ins_rel = []
        _multi_rel = []
        _entpair = []
        _length = []
        _scope = []
        _segment = []
        cur_pos = 0
        for i in range(idx0, idx1):
            if self.scope[self.order[i]][1]-self.scope[self.order[i]][0] > 5 * batch_size:
                tail = self.scope[self.order[i]][0] + 5 * batch_size
            else:
                tail = self.scope[self.order[i]][1]
            _word.append(self.data_word[self.scope[self.order[i]][0]:tail])
            _segment.append(self.data_segment[self.scope[self.order[i]][0]:tail])
            _mask.append(self.data_mask[self.scope[self.order[i]][0]:tail])
            _rel.append(self.data_rel[self.scope[self.order[i]][0]])
            _ins_rel.append(self.data_rel[self.scope[self.order[i]][0]:tail])
            _length.append(self.data_length[self.scope[self.order[i]][0]:tail])
            bag_size = tail - self.scope[self.order[i]][0]
            _scope.append([cur_pos, cur_pos + bag_size])
            cur_pos = cur_pos + bag_size

            #if self.mode == self.MODE_ENTPAIR_BAG:
            _one_multi_rel = np.zeros((self.rel_tot), dtype=np.int32)
            for j in range(self.scope[self.order[i]][0], tail):
                _one_multi_rel[self.data_rel[j]] = 1
            _multi_rel.append(_one_multi_rel)
            _entpair.append(self.scope_name[self.order[i]])
        '''
        for i in range(batch_size - (idx1 - idx0)):
            _word.append(np.zeros((1, self.data_word.shape[-1]), dtype=np.int32))
            _mask.append(np.zeros((1, self.data_mask.shape[-1]), dtype=np.int32))
            _rel.append(0)
            _ins_rel.append(np.zeros((1), dtype=np.int32))
            _length.append(np.zeros((1), dtype=np.int32))
            _scope.append([cur_pos, cur_pos + 1])
            cur_pos += 1

            #if self.mode == self.MODE_ENTPAIR_BAG:
            _multi_rel.append(np.zeros((self.rel_tot), dtype=np.int32))
            _entpair.append('None#None')
        '''
        batch_data['word'] = np.concatenate(_word)
        batch_data['segment'] = np.concatenate(_segment)
        batch_data['mask'] = np.concatenate(_mask)
        batch_data['rel'] = np.stack(_rel)
        batch_data['ins_rel'] = np.concatenate(_ins_rel)

        #if self.mode == self.MODE_ENTPAIR_BAG:
        batch_data['multi_rel'] = np.stack(_multi_rel)
        batch_data['entpair'] = _entpair

        batch_data['length'] = np.concatenate(_length)
        batch_data['scope'] = np.stack(_scope)

        return batch_data


if __name__=="__main__":
    dataset_dir = os.path.join('./data', 'nyt')
    data_loader = json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                        os.path.join(dataset_dir, 'word_vec.json'),
                                        os.path.join(dataset_dir, 'rel2id.json'), shuffle=True)
    
    batch_data = data_loader.next_batch(batch_size = 32)
    # One sentence likes    CLS this is a SEP head SEP and SEP tail SEP BLANK . #
    # and the example like [0,  10,  9, 4, 1, 2,   1,  20, 1,  33,  1,  3      ]#
    # Each pair SEP contains a head or tail #
    # Number is the index of each word (contains CLS, SEP ,BLANK)#

    print('Word shape is {}'.format(batch_data['word'].shape))  #[real_size, max_len]
    print('Pos1 shape is {}'.format(batch_data['pos1'].shape))  #[real_size, max_len]
    print('Pos2 shape is {}'.format(batch_data['pos1'].shape))  #[real_size, max_len]
    print('Rel shape is {}'.format(batch_data['rel'].shape))    #[batch_size,]
    print('Ins_rel shape is {}'.format(batch_data['ins_rel'].shape))    #[real_size,]
    print('length shape is {}'.format(batch_data['length'].shape))  #[real_size,]
    print("Mask shape is {}".format(batch_data['mask'].shape))  #[real_size, max_len]
    
    # For example:      example = batch_data['scope'][0] #
    #                   batch_data['word'][example[0]:example[1]] is the first bag #
    print('Scope is {}'.format(batch_data['scope'].shape))    #[batch_size, 2]

    print('Word is {}'.format(batch_data['word'][0]))  #[real_size, max_len]
    print('Pos1 is {}'.format(batch_data['pos1'][0]))  #[real_size, max_len]
    print('Pos2 is {}'.format(batch_data['pos1'][0]))  #[real_size, max_len]
    print('Rel is {}'.format(batch_data['rel'][0]))    #[batch_size,]
    print('Ins_rel is {}'.format(batch_data['ins_rel'][0]))    #[real_size,]
    print('length is {}'.format(batch_data['length'][0]))  #[real_size,]
    print("Mask is {}".format(batch_data['mask'][0]))  #[real_size, max_len]
    
    # For example:      example = batch_data['scope'][0] #
    #                   batch_data['word'][example[0]:example[1]] is the first bag #
    print('Scope is {}'.format(batch_data['scope'][0]))    #[batch_size, 2]

    for i, batch in enumerate(data_loader):
        print(i)
        print(batch)
        exit()
