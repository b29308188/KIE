import sys
import os 
import numpy as np
from _base import Model
from joblib import Parallel, delayed

class Crfpp(Model):
    """
    This is CRF class which uses the CRFPP binary files to train and test.
    """
    def __init__(self, crf_prefix, template_path,
            mod_prefix = "../mod/", 
            valid_data_prefix = "../valid_data/",
            log_prefix = "../log/",
            params = {"a" : "CRF-L2", "f": 3, "c": 1} ):
        """
        This is the constructor.
        crf_prefix : the folder that should contains two CRFPP binary files : crf_train and crf_test
        template_path : the path to the CRFPP template
        mod_prefix : the folder to store CRFPP models
        valid_data_prefix : the folder to store the validation data in the "cross_valid" method
        log_prefix : the folder to store the CRFPP log file
        params : the dictionary that contains the parameters
        """

        self.crf_prefix = crf_prefix
        self.template_path = template_path
        self.mod_prefix = mod_prefix
        self.valid_data_prefix = valid_data_prefix
        self.log_prefix = log_prefix
        self.params = params

        if not os.path.exists(mod_prefix):
            os.makedirs(mod_prefix)
        if not os.path.exists(valid_data_prefix):
            os.makedirs(valid_data_prefix)
        if not os.path.exists(log_prefix):
            os.makedirs(log_prefix)
    
    def fit(self, train_path):
        """
        Train on the input CRFPP file and store the model.
        """
        
        #log_path = self.log_prefix+"log"
        log_path = "/dev/null" # no log
        model_path = self.mod_prefix+"mod"
        self.model = model_path

        #parameters
        a = self.params['a']
        c = self.params['c']
        f = self.params['f']

        #fit it
        #print("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s"%(self.crf_prefix, a, f, c, self.template_path, train_path, model_path, log_path))
        os.system("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s "%(self.crf_prefix, a, f, c, self.template_path, train_path, model_path, log_path))


    def predict(self, test_path, pred_path, last_feature = False):
        """
        Test on the test_path (CRFPP format) and write the prediction in pred_path (CRFPP format).
        """
        
        if last_feature == False:
            f_w = open(test_path+".tmp", "w")
            with open(test_path, "r") as f:
                for line in f:
                    line = line.strip().split()[:-1]# we don't need the last segment (label)
                    f_w.write("\t".join(line)+ "\n")
            f_w.close()

            #predict it
            os.system("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, self.model, test_path+".tmp"))
            os.remove(test_path+".tmp")
        else:
            #predict it
            os.system("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, self.model, test_path))


    def cross_valid(self, data_path, n_fold = 5):
        """
        Do the cross validation on the input CRFPP file.
        In order to parallize the validation part, we use the joblib package and write an indepdent function "valid" for different threads
        """
        
        seqs = self.read_seqs(data_path)
        valid_size = int(len(seqs) / n_fold)
        
        #for i in range(n_fold):
            #print _valid(self, seqs[:i*valid_size]+seqs[(i+1):valid_size], seqs[i*valid_size:(i+1)*valid_size], str(i))
        ret = Parallel(n_jobs = n_fold)(delayed(_valid)(self, seqs[:i*valid_size]+seqs[(i+1):valid_size], seqs[i*valid_size:(i+1)*valid_size], str(i) ) for i in range(n_fold))
        tag_acc = [r for r in ret]
        return np.mean(tag_acc)

def _valid(self, train_seqs, test_seqs, file_suffix = ""):
    """ 
    The independent function from the class used to train and valid the data
    """
    log_path = self.log_prefix+"log"+file_suffix
    model_path = self.mod_prefix+"mod"+file_suffix
    train_path = self.valid_data_prefix+"train"+file_suffix
    test_path = self.valid_data_prefix+"test"+file_suffix
    pred_path = self.valid_data_prefix+"pred"+file_suffix
            
    self.write_seqs(train_seqs, train_path)
    self.write_seqs(test_seqs, test_path)

    #parameters
    a = self.params['a']
    c = self.params['c']
    f = self.params['f']
    
    #fit it 
    #print("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s"%(self.crf_prefix, a, f, c, self.template_path, train_path, model_path, log_path))
    os.system("%s/crf_learn -p 24 -a %s -f %d -c %f %s %s %s >> %s "%(self.crf_prefix, a, f, c, self.template_path,train_path, model_path, log_path))

    #predict it
    #print("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, model_path, test_path))
    os.system("%s/crf_test -o %s -m %s %s"%(self.crf_prefix, pred_path, model_path, test_path))
    
    return self.score(test_path, pred_path)

