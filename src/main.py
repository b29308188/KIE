import sys
import os
sys.path.append(".")
import numpy as np
from datasets import Cora
from models import Crfpp, NaiveBayes
from utils import retag, conditional_join, strip

crfpp_prefix = "../bin/"
template_path = "../train.template"
cora_data_path = "../data/cora/cora.tagged.txt"

kb_train_path = "../data/cora/kb_cora_train.dat"
kb_test_path = "../data/cora/kb_cora_test.dat"
kb_pred_path = "../data/cora/kb_cora_pred.dat"

crf_train_path = "../data/cora/crf_cora_train.dat"
crf_added_train_path = "../data/cora/crf_cora_added_train.dat"
crf_mixed_train_path = "../data/cora/crf_cora_mixed_train.dat"
crf_true_train_path = "../data/cora/crf_cora_true_train.dat"
crf_test_path = "../data/cora/crf_cora_test.dat"
crf_pred_path = "../data/cora/crf_cora_pred.dat"

if __name__ == "__main__":
    dataset = Cora.Cora(cora_data_path) #totally 500 articles
    model1 = NaiveBayes.NaiveBayes()
    model2 = Crfpp.Crfpp(crfpp_prefix, template_path)
    
    #N : the articles on which Crfpp trains
    N = (0, 10)
    
    #M : the articles on which NaiveBayes trains
    M = (100, 200)
    
    #K : the articles on which NaiveBayes tests, and then add it with N 
    K = (10, 100)
    
    #N : the articles that CRFPP tests on
    R = (200, dataset.article_num)
     
    #NaiveBayes trains on M and tests on K
    dataset.extract_training_knowledge_base_features(kb_train_path, article_range = M)
    #strip(kb_train_path)
    dataset.extract_testing_knowledge_base_features(kb_test_path, article_range = K)
    #strip(kb_test_path)
    model1.fit(kb_train_path)
    model1.predict(kb_test_path, kb_pred_path, last_feature = False)
    print "model1 valid score:", model1.score(kb_test_path, kb_pred_path)
        
    # CRF trains on N and tests on R
    dataset.extract_crf_features(crf_train_path, article_range = N)
    #strip(crf_train_path)
    dataset.extract_crf_features(crf_test_path, article_range = R)
    #strip(crf_train_path)
    model2.fit(crf_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 without adding data",model2.score(crf_test_path, crf_pred_path)
    
    
    # CRF trains on (N + predicted K) and tests on R
    dataset.extract_crf_features(crf_added_train_path, article_range = K)
    retag(crf_added_train_path, kb_pred_path, last_feature = False) 
    os.system("cat %s %s > %s" % (crf_train_path, crf_added_train_path, crf_mixed_train_path))
    model2.fit(crf_mixed_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 after adding data",model2.score(crf_test_path, crf_pred_path)
    
    dataset.extract_crf_features("../data/tmp0", article_range = K)
    model2.predict("../data/tmp0", "../data/tmp1", last_feature = False)
    conditional_join(crf_added_train_path, "../data/tmp1", "../data/tmp2")
    os.system("cat %s %s > %s" % (crf_train_path, "../data/tmp2", crf_mixed_train_path))
    model2.fit(crf_mixed_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 after adding consensus data",model2.score(crf_test_path, crf_pred_path)
    
    # CRF trains on (N + ture K) and tests on R
    dataset.extract_crf_features(crf_true_train_path, article_range = (0, 100))
    #strip(crf_true_train_path)
    model2.fit(crf_true_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 upper Bound",model2.score(crf_test_path, crf_pred_path)
    
