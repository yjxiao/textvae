import os
from collections import Counter
import numpy as np
import torch


PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

EXTRA_VOCAB = ['_PAD', '_SOS', '_EOS', '_UNK']


class Corpus(object):
    def __init__(self, datadir, min_n=2, max_vocab_size=None, max_length=None, with_label=False):
        self.min_n = min_n
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.with_label = with_label
        filenames = ['train.txt', 'valid.txt', 'test.txt']
        self.datapaths = [os.path.join(datadir, x) for x in filenames]
        self._construct_vocab()
        self.num_classes = len(self.idx2label)
        self.train, self.valid, self.test = [
            Data(dp, (self.word2idx, self.label2idx), max_length, with_label)
            for dp in self.datapaths]

    def _construct_vocab(self):
        self._vocab = Counter()
        labels = []
        for datapath in self.datapaths:
            with open(datapath) as f:
                # parse data files to construct vocabulary            
                for line in f:
                    if self.with_label:
                        label, text = line.strip().split('\t')
                        if label not in labels:
                            labels.append(label)
                    else:
                        text = line.strip()    
                    self._vocab.update(text.lower().split())
        vocab_size = len([x for x in self._vocab if self._vocab[x] >= self.min_n])
        self.idx2word = EXTRA_VOCAB + list(next(zip(*self._vocab.most_common(self.max_vocab_size)))[:vocab_size])
        self.word2idx = dict((w, i) for (i, w) in enumerate(self.idx2word))
        self.idx2label = sorted(labels)
        self.label2idx = dict((w, i) for (i, w) in enumerate(self.idx2label))


class Data(object):
    def __init__(self, datapath, vocabs, max_length=None, with_label=False):
        word2idx, label2idx = vocabs
        data = []
        labels = []
        with open(datapath) as f:
            for line in f:
                if with_label:
                    label, text = line.strip().split('\t')
                else:
                    label, text = None, line.strip()
                words = text.lower().split()
                if max_length is not None:
                    words = words[:max_length]
                data.append([SOS_ID] + [word2idx.get(x, UNK_ID) for x in words] + [EOS_ID])
                if with_label:
                    labels.append(label2idx[label]) 
        self.texts = np.array(data)
        self.labels = np.array(labels) if with_label else None
        self.has_label = with_label

    @property
    def size(self):
        return len(self.texts)
    
    def get_batch(self, batch_size, start_id=None):
        if start_id is None:
            batch_idx = np.random.choice(np.arange(self.size), batch_size)
        else:
            batch_idx = np.arange(start_id, start_id + batch_size)
        batch_texts = self.texts[batch_idx]
        batch_labels = self.labels[batch_idx] if self.has_label else None
        lengths = np.array([len(x) - 1 for x in batch_texts])    # actual length is 1 less
        # sort by length in order to use packed sequence
        idx = np.argsort(lengths)[::-1]
        batch_texts = batch_texts[idx]
        batch_labels = batch_labels[idx] if self.has_label else None
        lengths = list(lengths[idx])
        max_len = int(lengths[0] + 1)
        text_tensor = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
        for i, x in enumerate(batch_texts):
            n = len(x)
            text_tensor[i][:n] = torch.from_numpy(np.array(x))
        label_tensor = torch.from_numpy(batch_labels) if self.has_label else None
        return text_tensor, label_tensor, lengths, np.argsort(idx)


class SSCorpus(object):
    def __init__(self, datadir, num_labeled, min_n=2, max_vocab_size=None, max_length=None):
        self.min_n = min_n
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.num_labeled = num_labeled
        filenames = ['train_with_label.txt', 'valid_with_label.txt', 'test_with_label.txt']
        self.textspaths = [os.path.join(datadir, x) for x in filenames]
        self._construct_vocab()
        self.train_data = SSData(self.textspaths[0], self.word2idx,
                                 max_length, self.num_classes, num_labeled)
        self.valid_data = SSData(self.textspaths[1], self.word2idx,
                                 max_length, self.num_classes)
        self.test_data = SSData(self.textspaths[2], self.word2idx, 
                                max_length, self.num_classes)

    def _construct_vocab(self):
        self._vocab = Counter()
        categories = set()
        for datapath in self.textspaths:
            with open(datapath) as f:
                # parse data files to construct vocabulary
                for line in f:
                    if line == '\n':
                        continue
                    cat, text = line.strip().split('\t')
                    categories.add(int(cat))
                    self._vocab.update(text.lower().split())
        vocab_size = len([x for x in self._vocab if self._vocab[x] >= self.min_n])
        self.idx2word = EXTRA_VOCAB + list(next(zip(*self._vocab.most_common(self.max_vocab_size)))[:vocab_size])
        self.word2idx = dict((w, i) for (i, w) in enumerate(self.idx2word))
        self.num_classes = len(categories)


class SSData(object):
    def __init__(self, datapath, vocab, max_length, num_classes, num_labeled=None):
        """Caution: this class assumes classes are balanced and training samples 
        are sorted by class.
        """
        
        self.vocab_size = len(vocab)
        data = []
        labels = []
                                                        
        with open(datapath) as f:
            for line in f:
                cat, text = line.strip().split('\t')
                words = text.lower().split()[:max_length]
                indices = [vocab.get(x, UNK_ID) for x in words]
                data.append([SOS_ID] + indices + [EOS_ID])
                labels.append(int(cat))
        self.texts = np.array(data)
        self.labels = np.array(labels)

        num_labeled = self.size if num_labeled is None else num_labeled
        # notice here if num_labeled or self.size is not a multiple of num_classes
        # the results are unwanted
        labeled_per_class = num_labeled // num_classes
        num_per_class = self.size // num_classes
        labeled_idx = np.arange(labeled_per_class)
        unlabeled_idx = np.arange(labeled_per_class, num_per_class)
        for k in range(1, num_classes):
            labeled_idx = np.append(labeled_idx,
                                    np.arange(num_per_class*k, num_per_class*k+labeled_per_class))
            unlabeled_idx = np.append(unlabeled_idx,
                                      np.arange(num_per_class*k+labeled_per_class, num_per_class*(k+1)))
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx

    @property
    def size(self):
        return len(self.texts)

    def get_batch(self, batch_size, start_id=None, labeled=True):
        if start_id is None:
            if labeled:
                batch_idx = np.random.choice(self.labeled_idx, batch_size)
            else:
                batch_idx = np.random.choice(self.unlabeled_idx, batch_size)
            batch_texts = self.texts[batch_idx]
            batch_labels = self.labels[batch_idx]
        else:
            batch_texts = self.texts[start_id:(start_id+batch_size)]
            batch_labels = self.labels[start_id:(start_id+batch_size)]
        lengths = np.array([len(x) - 1 for x in batch_texts])    # actual length is 1 less
        # sort by length in order to use packed sequence
        idx = np.argsort(lengths)[::-1]
        batch_texts = batch_texts[idx]
        batch_labels = batch_labels[idx]
        lengths = list(lengths[idx])
        max_len = int(lengths[0] + 1)
        data_tensor = torch.LongTensor(batch_size, max_len).fill_(PAD_ID)
        label_tensor = torch.LongTensor(batch_labels)
        for i, x in enumerate(batch_texts):
            n = len(x)
            data_tensor[i][:n] = torch.from_numpy(np.array(x))
        inputs = data_tensor[:, :-1].clone()
        targets = data_tensor[:, 1:].clone()
        return inputs, targets, lengths, label_tensor
                        
