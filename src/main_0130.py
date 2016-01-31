import sys
import os
sys.path.append(".")
import numpy as np
from datasets import Cora, MicrosoftAcademicGraph, Grobid
from models import Crfpp, NaiveBayes
from utils import retag, conditional_join, strip

crfpp_prefix = "../bin/"
template_path = "../train.template"
grobid_data_path = "../data/grobid/grobid.tagged.txt"
cora_data_path = "../data/cora/cora.tagged.txt"
ms_data_path = "/tmp2/b02902030/Engdir"
#ms_data_path = "/tmp2/KIE/PDFs"

test_path = "../data/test.dat"
pred_path = "../data/pred.dat"
train_path = "../data/train.dat"

if __name__ == "__main__":
    dataset2 = Grobid.Grobid(grobid_data_path) #totally 500 articles
    dataset1 = MicrosoftAcademicGraph.MicrosoftAcademicGraph(ms_data_path) #totally 500 articles
    model2 = Crfpp.Crfpp(crfpp_prefix, template_path)
    model1 = NaiveBayes.NaiveBayes()
    
    
    dataset1.extract_crf_features(train_path,first_n_token = 70)
    dataset2.extract_crf_features(test_path)
    strip(train_path)
    strip(test_path)
    model2.fit(train_path)
    model2.predict(test_path, pred_path, last_feature = False)
    print "Author Precision / Title Precision / Author Recall / Title Recall"
    print model2.score(test_path, pred_path)


