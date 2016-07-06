from __future__ import print_function

import codecs
import collections
import gzip
import io
import itertools
from multiprocessing import Pool
import operator
import os
from random import shuffle
import subprocess

import numpy as np
from six import PY2
from six.moves import cPickle, zip, reduce, map

try:
    import lzma
except ImportError:
    from backports import lzma

class TextLoader(object):

    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        # self.input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        size_file = os.path.join(data_dir, "size.pkl")
        
        self.input_files = []
        
        for root, __, files in os.walk(data_dir, 'input'):
            for f in files:
                __, ext = os.path.splitext(f)
                if ext == '.xz':
                    self.input_files.append(os.path.join(root, f))
        
        # Shuffle for training?
        shuffle(self.input_files)

        if not (os.path.exists(vocab_file) and os.path.exists(size_file)):
            print("reading text file")
            self.preprocess(vocab_file, size_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, size_file)

        self.num_batches = int(self.size / (self.batch_size * self.seq_length))

    def preprocess(self, vocab_file, size_file):
        print("Generating counter")
        # PY3 code path is slower on PY2 due to missing C optimization
        if PY2:
            counter = {}
            counter_get = counter.get
            for char in self.get_data():
                counter_get[char] = counter_get(char, 0) + 1
        else:
            counter = collections.Counter(self.get_data())
        print("Done generating counter")

        count_pairs = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        self.chars = list(map(operator.itemgetter(0), count_pairs))
        self.vocab_size = len(self.chars)
        self.vocab = { char: i for i, char in enumerate(self.chars) }
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        self.size = sum(counter.values())

        with open(size_file, 'wb') as f:
            cPickle.dump(self.size, f)

    def load_preprocessed(self, vocab_file, size_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        with open(size_file, 'rb') as f:
            self.size = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = { char: i for i, char in enumerate(self.chars) }

    def get_data(self):
        def read():
            total = len(self.input_files)
            for i, input_file in enumerate(self.input_files):
                print("Reading file %s/%s" % (i+1, total))
                with lzma.open(input_file) as uncompressed:
                    with io.TextIOWrapper(uncompressed, encoding='utf-8') as f:
                        yield f.read()
                yield '\x00'
        return itertools.chain.from_iterable(read())

    def get_batches(self):

        # When the data (tensor) is too small, let's give them a better error message
        assert self.num_batches > 0, "Not enough data. Make seq_length and batch_size small."

        tensor = map(self.vocab.get, self.get_data())
        # Truncate dangling elements
        tensor = itertools.islice(tensor, self.num_batches * self.batch_size * self.seq_length)
        
        peek = next(tensor)
        left, right = itertools.tee(tensor)
        
        def batch(iterable):
            it = iter(iterable)
            count = self.batch_size * self.seq_length
            shape = (self.batch_size, self.seq_length)
            while True:
                chunk = list(itertools.islice(it, count))
                if not chunk:
                    return
                arr = np.array(chunk)
                arr = np.reshape(arr, shape)
                yield arr

        return zip(
            batch(itertools.chain([peek], left)), 
            batch(itertools.chain(right, [peek]))
        )

