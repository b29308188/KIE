import sys
import os
sys.path.append(".")
import numpy as np
from datasets import MicrosoftAcademicGraph, Cora
from models import Crfpp, NaiveBayes

#import utils

#mag_prefix = "/tmp2/b02902030/Engdir"

crfpp_prefix = "../bin/"
#template_path = "../train.template"
template_path = "../parsCit.template"
cora_data_path = "../cora.tagged.txt"

kb_train_path = "../data/kb_cora_train.dat"
kb_test_path = "../data/kb_cora_test.dat"
kb_pred_path = "../data/kb_cora_pred.dat"

crf_data_path = "../data/crf_cora.dat"
crf_train_path = "../data/crf_cora_train.dat"
crf_extend_train_path = "../data/crf_cora_extend_train.dat"
crf_test_path = "../data/crf_cora_test.dat"
crf_pred_path = "../data/crf_cora_pred.dat"

if __name__ == "__main__":
    #dataset = MicrosoftAcademicGraph.MicrosoftAcademicGraph(mag_prefix)
    dataset = Cora.Cora(cora_data_path)
    model1 =  NaiveBayes.NaiveBayes()
    model2 = Crfpp.Crfpp(crfpp_prefix, template_path)
    
    dataset.extract_knowledge_base_features()
    dataset.write_kb_data(kb_train_path, article_range = (100, 200))
    dataset.write_kb_data(kb_test_path, article_range = (10, 100))
    #model1.cross_valid(kb_train_path)
    model1.fit(kb_train_path)
    model1.predict(kb_test_path, kb_pred_path)
    print "model1 valid score:", model1.score(kb_test_path, kb_pred_path)
        
    dataset.extract_crf_features()
    dataset.write_crfpp_data(crf_train_path, article_range = (0, 10))
    dataset.write_crfpp_data(crf_test_path, article_range = (200, dataset.article_num))
    model2.fit(crf_train_path)
    model2.predict(crf_test_path, crf_pred_path)
    print "model2 without adding data",model2.score(crf_test_path, crf_pred_path)
    
    os.system("cat %s %s > %s" % (crf_train_path, kb_pred_path, crf_extend_train_path))
    model2.fit(crf_extend_train_path)
    model2.predict(crf_test_path, crf_pred_path)
    print "model2 after adding data",model2.score(crf_test_path, crf_pred_path)

    dataset.write_crfpp_data(crf_train_path, article_range = (0, 100))
    model2.fit(crf_train_path)
    model2.predict(crf_test_path, crf_pred_path)
    print "model2 Upper Bound",model2.score(crf_test_path, crf_pred_path)
    
    
    #dataset.extract_knowledge_base_features()
    #`print dataset.prefix, dataset.article_num
        
    """
    for i in [1, 2, 5, 10, 20, 30, 50, 100, 200, 400]:
        dataset.extract_crf_features()
        dataset.write_crfpp_data(crf_train_path, article_range = (0, i))
        dataset.write_crfpp_data(crf_test_path, article_range = (i, dataset.article_num))
        
        model2 = Crfpp.Crfpp(crfpp_prefix, template_path)
        model2.fit(crf_train_path)
        model2.predict(crf_test_path, crf_pred_path)
        print i, model2.score(crf_test_path, crf_pred_path)[0]
    """
    #model2.cross_valid(crf_data_path ,n_fold = 5)

