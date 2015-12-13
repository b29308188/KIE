import sys
import glob
import numpy as np

"""
This is the class that handles all of the operations related to the Cora dataset.
"""

class Cora:
    def __init__(self, file_path):
        
        self.file_path = file_path
        self.article_num = sum([1 for line in open(file_path, "r")])
    
    def write_kb_data(self, output_path, article_range = (0, None)):
        start = article_range[0]
        end = article_range[1]
        
        with open(output_path, "w") as f:
            for seq in self.kb_seqs[start:end]:
                for x in seq:
                    for xi in x:
                        f.write(xi+" ")
                    f.write("\n")
                f.write("\n")
    
    def write_crfpp_data(self, output_path, article_range = (0, None)):
        """ 
        Write the sequences of feafures into the CRFPP input format 
        crf_seqs : the list of sequences, each sequence is a list of tokens, and each token contains its features and its label 
        artcle_range(start, end): the range of articles to write
        """
        
        start = article_range[0]
        end = article_range[1]
        
        with open(output_path, "w") as f:
            for seq in self.crf_seqs[start:end]:
                for x in seq:
                    for xi in x:
                        f.write(xi+" ")
                    f.write("\n")
                f.write("\n")

    def extract_token_features(self, token):
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


    def extract_knowledge_base_features(self):
        self.kb_seqs = []#list of sequences
        
        with open(self.file_path, "r") as f:
            #process title
            for line in f:
                kb_seq = []
                tokens = line.strip().split()
                tag = None
                for token in tokens:
                    if "<" == token[0] and ">" == token[-1]:
                        if "/" == token[1]:
                            tag = None
                        else:
                            tag = token[1:-1]
                    else:
                        x = self.extract_token_features(token)#extract token feature
                        x.append(tag)
                        kb_seq.append(x)
                self.kb_seqs.append(kb_seq)
    
    def extract_crf_features(self):
        """
        Extract the features of the first n tokens in the articles(HTML files), and this features are used for CRF.
        We store them into sequences (seqs), and the groups of tokens in the same article is a sequence
        crf_seqs : the list of sequences, each sequence is a list of tokens, and each token contains its features and its label 
        """

        self.crf_seqs = []#list of sequences
        
        with open(self.file_path, "r") as f:
            #process title
            for line in f:
                crf_seq = []
                tokens = line.strip().split()
                tag = None
                for token in tokens:
                    if "<" == token[0] and ">" == token[-1]:
                        if "/" == token[1]:
                            tag = None
                        else:
                            tag = token[1:-1]
                    else:
                        x = self.extract_token_features(token)#extract token feature
                        x.append(tag)
                        crf_seq.append(x)
                self.crf_seqs.append(crf_seq)
