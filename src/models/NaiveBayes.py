import sys
import os 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes:
    def __init__(self):
        self.model = MultinomialNB()
        self.feature_encoder = DictVectorizer()
        self.tag_encoder = LabelEncoder()
        
    def read_seqs(self, file_path):
        seqs = []
        with open(file_path, "r") as f:
            seq = []
            for token in f:
                if len(token) >= 5:
                    seq.append(token)
                else:
                    seqs.append(seq)
                    seq = []
        return seqs

    def seqs_to_train(self, seqs):
        X = []
        Y = []
        for seq in seqs:
            for token in seq:
                features = token.strip().split()
                X.append({i:xi for (i, xi) in enumerate(features[:-1]) } )           
                Y.append(features[-1])
        X = self.feature_encoder.fit_transform(X).toarray()
        Y = self.tag_encoder.fit_transform(Y)
        return (X, Y)
    
    def eval_tag_accuracy(self, truth_targets, pred_targets):
        total = 0.0
        hit = 0.0
        for (truth_target, pred_target) in zip(truth_targets, pred_targets):
            total += len(truth_target)
            hit += sum([1 for (t_t, p_t) in zip(truth_target, pred_target) if t_t == p_t] )
        return hit/total

    def eval_seq_accuracy(self, truth_targets, pred_targets):
        total = 0.0
        hit = 0.0
        for (truth_target, pred_target) in zip(truth_targets, pred_targets):
            hit += all(t_t == p_t for (t_t, p_t) in zip(truth_target, pred_target))
            total += 1
        return hit/total
    
    def score(self, truth_path, pred_path):
        truth_targets = []
        truth_seqs = self.read_seqs(truth_path)
        for truth_seq in truth_seqs:
            truth_targets.append([token.split()[-1] for token in truth_seq])
        pred_targets = []
        pred_seqs = self.read_seqs(pred_path)
        for pred_seq in pred_seqs:
            pred_targets.append([token.split()[-1] for token in pred_seq])
        
        return self.eval_tag_accuracy(truth_targets, pred_targets), self.eval_seq_accuracy(truth_targets, pred_targets)
    def fit(self, train_path):
        seqs = self.read_seqs(train_path)
        (X, Y) = self.seqs_to_train(seqs)
        self.model.fit(X, Y)
    
    def predict(self, test_path, pred_path):
        seqs = self.read_seqs(test_path)
        with open(pred_path, "w") as f:
            for seq in seqs:
                X = []
                for token in seq:
                    features = token.strip().split()
                    X.append({i:xi for (i, xi) in enumerate(features[:-1]) } )           
                X = self.feature_encoder.transform(X).toarray()
                Y = self.model.predict(X)
            
                for (i, token) in enumerate(seq):
                    for xi in token.strip().split()[:-1]:
                        f.write(xi+ " ")
                    f.write(self.tag_encoder.inverse_transform(Y[i]) + "\n")
                f.write("\n")
        print pred_path

    def cross_valid(self, data_path, n_fold = 5):
        seqs = self.read_seqs(data_path)
        (X, Y) = self.seqs_to_train(seqs)
        print np.mean(cross_val_score(self.model, X, Y, cv = 5))
