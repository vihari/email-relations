# Fetched from: https://raw.githubusercontent.com/stanfordnlp/GloVe/master/eval/python/distance.py
import argparse
import numpy as np
import sys

EMBEDDING_SIZE=50

class Glove(object):
    def __init__(self, vectors_file):
        self.words = []
        with open(vectors_file, 'r') as f:
            self.vectors = {}
            for line in f:
                fields = line.rstrip().split(' ')
                self.words.append(fields[0])
                vals = fields
                self.vectors[vals[0]] = [float(x) for x in vals[1:]]

        self.W, self.vocab, self.ivocab = self.generate()
    
    def generate(self):
        vocab_size = len(self.words)
        vocab = {w: idx for idx, w in enumerate(self.words)}
        ivocab = {idx: w for idx, w in enumerate(self.words)}

        vector_dim = len(self.vectors[ivocab[0]])
        W = np.zeros((vocab_size, vector_dim))
        for word, v in self.vectors.items():
            if word == '<unk>':
                continue
            W[vocab[word], :] = v

        # normalize each word vector to unit variance
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T
        return (W_norm, vocab, ivocab)

    def vector(self, word):
        """Returns None if word is OOV"""
        if word in self.words:
            return self.vectors[word]
        return None
    
    def get_nparray(self):
        return np.asarray([self.vectors.get(self.ivocab.get(idx)) for idx in range(len(self.vocab))])
        
    def distance(self, input_term):
        for idx, term in enumerate(input_term.split(' ')):
            if term in self.vocab:
                print('Word: %s  Position in vocabulary: %i' % (term, self.vocab[term]))
                if idx == 0:
                    vec_result = np.copy(self.W[self.vocab[term], :])
                else:
                    vec_result += self.W[self.vocab[term], :] 
            else:
                print('Word: %s  Out of dictionary!\n' % term)
                return
        
        vec_norm = np.zeros(vec_result.shape)
        d = (np.sum(vec_result ** 2,) ** (0.5))
        vec_norm = (vec_result.T / d).T

        dist = np.dot(self.W, vec_norm.T)

        for term in input_term.split(' '):
            index = self.vocab[term]
            dist[index] = -np.Inf

        a = np.argsort(-dist)[:N]

        print("\n                               Word       Cosine distance\n")
        print("---------------------------------------------------------\n")
        for x in a:
            print("%35s\t\t%f\n" % (self.ivocab[x], dist[x]))

    def get_closest(self, query_matrix, n=5):
        """
        :param: query_matrix is [batch size x embedding size] matrix -- closest words are found for each of the rows
        :return: an array of length: batch size of array of length n containing the n closest words 
        """
        # normalize input
        query_matrix = query_matrix/(np.sum(query_matrix**2, axis=1, keepdims=True)**.5)
        dists = np.matmul(query_matrix, self.W.T)
        a = np.argsort(-dists)[:, :n]
        return [[(self.ivocab[x], dists[ri, x]) for x in row] for ri, row in enumerate(a)]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()
    N = 10;          # number of closest words that will be shown
    wv = Glove(args.vectors_file)
    print wv.get_closest(np.random.normal(size=[5, 50]))
    while True:
        input_term = raw_input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            wv.distance(input_term)
