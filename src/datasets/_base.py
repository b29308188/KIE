import sys
import os 
import numpy as np
from abc import ABCMeta, abstractmethod

class Dataset(object):
    """
    An abstract dataset class
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        This is the constructor.
        """
        pass

    def write_crfpp_data(self, output_path, seqs):
        """ 
        Write the sequences of feafures into the CRFPP input format 
        seqs : the list of sequences, each sequence is a list of tokens, and each token contains its features and its label 
        """
        
        with open(output_path, "w") as f:
            for seq in seqs:
                for x in seq:
                    for xi in x:
                        f.write(xi+" ")
                    f.write("\n")
                f.write("\n")
    
    @abstractmethod
    def extract_training_knowledge_base_features(self, output_path):
        """
        Extract the features from the knowledge base, these features are used by non-structured classifiers 
        , e.g. NaiveBayes, when "TRAINING".
        Since we only have the knowledge base, we may not extract structure features.
        We don't need to separate sequences becuase we may not have different sequences in the knwoledge base.
        However, for the consistent issues, we still write them into the CRFPP format.
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_testing_knowledge_base_features(self, output_path, article_range = (0, None)):
        """
        Extract the features from the articles, these features are used by non-structured classifiers 
        , e.g. NaiveBayes, when "TESTING".
        Since we only have the knowledge base when training, we may not extract structure features.
        But we should separate different articles because the predicted results should be used by CRF!! 
        We write them into the CRFPP format.
        """
        raise NotImplementedError
     
    @abstractmethod
    def extract_crf_features(self, output_path, article_range = (0, None)):
        """
        Extract the features from the articles, these features are used by structured classifiers 
        , e.g. CRF, when "TRAINING" and TESTING"".
        We write them into the CRFPP format.
        """
        raise NotImplementedError
    
