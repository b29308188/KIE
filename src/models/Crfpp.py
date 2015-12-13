import sys
import os 
import sys
import os
import numpy as np
from random import shuffle
from joblib import Parallel, delayed
def valid(self, train_seqs, test_seqs, file_suffix = ""):
    log_path = self.log_prefix+"log"+file_suffix
    model_path = self.mod_prefix+"mod"+file_suffix
    train_path = self.valid_data_prefix+"train"+file_suffix
    test_path = self.valid_data_prefix+"test"+file_suffix
    pred_path = self.valid_data_prefix+"pred"+file_suffix

    #for seq in train_seqs:
    #        shuffle(seq)
    #for seq in test_seqs:
    #        shuffle(seq)
            
    self.write_seqs(train_seqs, train_path)
    self.write_seqs(test_seqs, test_path)

    a = "CRF-L2"
    f = 3
    c = 1.0
    print("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s"%(self.crf_prefix, a, f, c, self.template_path, train_path, model_path, log_path))
    os.system("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s "%(self.crf_prefix, a, f, c, self.template_path,train_path, model_path, log_path))

    print("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, model_path, test_path))
    os.system("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, model_path, test_path))
    truth_targets = []
    truth_seqs = self.read_seqs(test_path)
    for truth_seq in truth_seqs:
        truth_targets.append([token.split()[-1] for token in truth_seq])
    pred_targets = []
    pred_seqs = self.read_seqs(pred_path)
    for pred_seq in pred_seqs:
        pred_targets.append([token.split()[-1] for token in pred_seq])
    
    return self.eval_tag_accuracy(truth_targets, pred_targets), self.eval_seq_accuracy(truth_targets, pred_targets)
class Crfpp:
    def __init__(self, crf_prefix, template_path,
            mod_prefix = "../mod/", 
            valid_data_prefix = "../valid_data/",
            log_prefix = "../log/"):
        
        self.crf_prefix = crf_prefix
        self.template_path = template_path
        self.mod_prefix = mod_prefix
        self.valid_data_prefix = valid_data_prefix
        self.log_prefix = log_prefix
        
        if not os.path.exists(mod_prefix):
            os.makedirs(mod_prefix)
        if not os.path.exists(valid_data_prefix):
            os.makedirs(valid_data_prefix)
        if not os.path.exists(log_prefix):
            os.makedirs(log_prefix)
    
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

    def write_seqs(self, seqs ,file_name):
        with open(file_name, "w") as f:
            for seq in seqs:
                for token in seq:
                    f.write(token)
                f.write("\n")
    
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

    def fit(self, train_path):
        log_path = self.log_prefix+"log"
        model_path = self.mod_prefix+"mod"
        self.model = model_path
        a = "CRF-L2"
        f = 3
        c = 1.0
        print("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s"%(self.crf_prefix, a, f, c, self.template_path, train_path, model_path, log_path))
        os.system("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s "%(self.crf_prefix, a, f, c, self.template_path, train_path, model_path, log_path))

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

    def predict(self, test_path, pred_path):
        print("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, self.model, test_path))
        os.system("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, self.model, test_path))

    def cross_valid(self, data_path, n_fold = 5):
        seqs = self.read_seqs(data_path)
        valid_size = int(len(seqs) / n_fold)
        ret = Parallel(n_jobs = n_fold)(delayed(valid)(self, seqs[:i*valid_size]+seqs[(i+1):valid_size], seqs[i*valid_size:(i+1)*valid_size], str(i) ) for i in range(n_fold))
        tag_acc = [r[0] for r in ret]
        seq_acc = [r[1] for r in ret]
        #for i in range(n_fold):
            #print valid(self, seqs[:i*valid_size]+seqs[(i+1):valid_size], seqs[i*valid_size:(i+1)*valid_size], str(i))
        print("tag accuracy = %f ; sequence accuracy = %f" %(np.mean(tag_acc), np.mean(seq_acc)))
