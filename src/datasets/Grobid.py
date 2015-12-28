import sys
import glob
import numpy as np
from _base import Dataset
"""
This is the class that handles all of the operations related to the Grobid dataset.
"""

class Grobid(Dataset):
    #inherit Dataset class from _base
    def __init__(self, file_path):
        """
        file_path : file that contains the dataset 
        article_num : the number of lines (articles)
        """
        self.file_path = file_path
        self.article_num = sum([1 for line in open(file_path, "r")])
        
    def _extract_token_features(self, token):
        """ 
        Extract features of a token
        x0 : token itself
        x1~x5: prefix 1~5 (e.g. abcdef -> (a, ab, abc, abcd, abcde))
        x6~x10: suffix 1~5 (e.g. abcdef -> (f, ef, def, cdef, bcdef))
        """
        
        x = []
        x.append(token)#token itself

        for i in range(5):#prefix 1~5
            if len(token) >= i+1:
                x.append(token[:i+1])
            else:#not enough length : the whole string
                x.append(token[:len(x)-1])

        for i in range(5):#suffix 1~5
            if len(token) >= i+1:
                x.append(token[-(i+1):])
            else:#not enough length: the whole string
                x.append(token[-(len(x)-1):])
        return x

    
    def extract_training_knowledge_base_features(self, output_path, article_range = (0, None)):
        """
        Extract the features from the knowledge base, these features are used by non-structured classifiers 
        , e.g. NaiveBayes, when "TRAINING".
        Since we only have the knowledge base, we may not extract structure features.
        We don't need to separate sequences becuase we may not have different sequences in the knwoledge base.
        However, for the consistent issues, we still write them into the CRFPP format.
        """
        #for Grobid, there's no knowledge base, so we have to extract them from the articles
        self.extract_testing_knowledge_base_features(output_path, article_range = article_range)
    
    def extract_testing_knowledge_base_features(self, output_path, article_range = (0, None)):
        """
        Extract the features from the articles, these features are used by non-structured classifiers 
        , e.g. NaiveBayes, when "TESTING".
        Since we only have the knowledge base when training, we may not extract structure features.
        But we should separate different articles because the predicted results should be used by CRF!! 
        We write them into the CRFPP format.
        """
        kb_seqs = []#knowledge base sequences
        
        #read from the dataset
        with open(self.file_path, "r") as f:
            #for each line(each article)
            for line in f.readlines()[article_range[0]:article_range[1]]:
                kb_seq = []
                tokens = line.strip().split()
                tag = None
                # for each token separated by " "
                for token in tokens:
                    #if it is a html tag
                    if "<" == token[0] and ">" == token[-1]:
                        if "/" == token[1]:#end </>
                            tag = None
                        else:#start <>
                            tag = token[1:-1]
                    #else a content token
                    else:
                        #extract the features of a token
                        x = self._extract_token_features(token)
                        #add label
                        x.append(tag)
                        kb_seq.append(x)
                kb_seqs.append(kb_seq)

        #write to the file
        self.write_crfpp_data(output_path, kb_seqs)
    
    def extract_crf_features(self, output_path, article_range = (0, None)):
        """
        Extract the features from the articles, these features are used by structured classifiers 
        , e.g. CRF, when "TRAINING" and TESTING"".
        We write them into the CRFPP format.
        crf_seqs : the list of sequences, each sequence is a list of tokens, and each token contains its features and its label 
        """
        crf_seqs = []#knowledge base sequences
        
        #read from the dataset
        with open(self.file_path, "r") as f:
            #for each line (each article)
            for line in f.readlines()[article_range[0]: article_range[1]]:
                crf_seq = []
                tokens = line.strip().split()
                tag = None
                # for each token separated by " "
                for token in tokens:
                    #if it is a html tag
                    if "<" == token[0] and ">" == token[-1]:
                        if "/" == token[1]:#end </>
                            tag = None
                        else:#start <>
                            tag = token[1:-1]
                    #else a content token
                    else:
                        #extract the features of a token
                        x = self._extract_token_features(token)
                        #add label
                        x.append(tag)
                        crf_seq.append(x)
                crf_seqs.append(crf_seq)

        #write to the file
        self.write_crfpp_data(output_path, crf_seqs)
