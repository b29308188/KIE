import sys
import os 
import numpy as np
from abc import ABCMeta, abstractmethod

class Model(object):
    """
    An abstract model class
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """
        This is the constructor.
        """
        pass
        
    def read_seqs(self, file_path):
        """
        Read the CRFPP file into a list of sequences:
        Each suquence represents an article and contains a list of tokens.
        Each token is stored in a line that contains several features seperated by " ".
        """
        seqs = []
        with open(file_path, "r") as f:
            seq = []
            for token in f:
                if len(token) >= 5:
                    seq.append(token)
                else:# a new sequence
                    seqs.append(seq)
                    seq = []
        return seqs
    
    def write_seqs(self, seqs ,file_name):
        """
        Write the  list of sequences into the CRFPP file:
        NOTE : This method MAY also write the label into the file because the label is the last segment in the token.
        """
        with open(file_name, "w") as f:
            for seq in seqs:
                for token in seq:
                    f.write(token)
                f.write("\n")

    def eval_tag_accuracy(self, truth_targets, pred_targets):
        """
        Calculate the tag accuracy of the predicted sequences.
        truth_targets : the list of true targetes
        pred_target : the list of predicted targets
        """
        total = 0.0
        hit = 0.0
        for (truth_target, pred_target) in zip(truth_targets, pred_targets):# for each sequence
            total += len(truth_target)
            hit += sum([1 for (t_t, p_t) in zip(truth_target, pred_target) if t_t == p_t] )
        return hit/total

    
    def score(self, truth_path, pred_path):
        """
        Calculate the scores.
        Temporarily there is only accuracy metric 
        truth_path and pred_path should be the files of CRFPP format
        """
        truth_targets = []
        truth_seqs = self.read_seqs(truth_path)
        for truth_seq in truth_seqs:
            #token.strip().split()[-1]: the tag of this token (label)
            truth_targets.append([token.strip().split()[-1] for token in truth_seq])
        pred_targets = []
        pred_seqs = self.read_seqs(pred_path)
        for pred_seq in pred_seqs:
            pred_targets.append([token.strip().split()[-1] for token in pred_seq])
        
        return self.eval_tag_accuracy(truth_targets, pred_targets)

    @abstractmethod
    def fit(self, train_path):
        """
        Train on the input CRFPP file and store the model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, test_path, pred_path):
        """
        Test on the test_path (CRFPP format) and write the prediction in pred_path (CRFPP format).
        """
        raise NotImplementedError

    @abstractmethod
    def cross_valid(self, data_path, n_fold = 5):
        """
        Do the cross_validation on the input CRFPP file.
        """
        raise NotImplementedError
