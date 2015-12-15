import sys
import os
sys.path.append(".")
import numpy as np
from datasets import Cora
from models import Crfpp, NaiveBayes

crfpp_prefix = "../bin/"
template_path = "../train.template"
cora_data_path = "../data/cora/cora.tagged.txt"

kb_train_path = "../data/cora/kb_cora_train.dat"
kb_test_path = "../data/cora/kb_cora_test.dat"
kb_pred_path = "../data/cora/kb_cora_pred.dat"

crf_train_path = "../data/cora/crf_cora_train.dat"
crf_extend_train_path = "../data/cora/crf_cora_extend_train.dat"
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
    dataset.extract_testing_knowledge_base_features(kb_test_path, article_range = K)
    model1.fit(kb_train_path)
    model1.predict(kb_test_path, kb_pred_path, last_feature = False)
    print "model1 valid score:", model1.score(kb_test_path, kb_pred_path)
        
    # CRF trains on N and tests on R
    dataset.extract_crf_features(crf_train_path, article_range = N)
    dataset.extract_crf_features(crf_test_path, article_range = R)
    model2.fit(crf_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 without adding data",model2.score(crf_test_path, crf_pred_path)
    
    
    # CRF trains on (N + predicted K) and tests on R
    os.system("cat %s %s > %s" % (crf_train_path, kb_pred_path, crf_extend_train_path))
    model2.fit(crf_extend_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 after adding data",model2.score(crf_test_path, crf_pred_path)

    # CRF trains on (N + ture K) and tests on R
    dataset.extract_crf_features(crf_true_train_path, article_range = (0, 100))
    model2.fit(crf_true_train_path)
    model2.predict(crf_test_path, crf_pred_path, last_feature = False)
    print "model2 upper Bound",model2.score(crf_test_path, crf_pred_path)
    
