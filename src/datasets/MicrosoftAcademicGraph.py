import sys
import glob
import numpy as np

"""
This is the class that handles all of the operations related to the Microsoft Academic Graph.
"""
head_ends = ["Introduction", "introduction", "INTRODUCTION", "Abstract", "ABSTRACT", "abstract", "OUTLINE, ""outline", "Outline"]

class MicrosoftAcademicGraph:
    def __init__(self, prefix):
        """
        prefix : the MicrosoftAcademicGraph folder that contains HTML files
        article_list : the list of the HTML files
        article_num : the number of HMTL files under the specified folder_pather
        """
        
        self.prefix = prefix
        self.article_list = [file_path for file_path in glob.glob(prefix+"/*")]
        self.article_num = len(self.article_list)
    
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

    def extract_crf_features(self, first_n_token = 100):
        """
        Extract the features of the first n tokens in the articles(HTML files), and this features are used for CRF.
        We store them into sequences (seqs), and the groups of tokens in the same article is a sequence
        crf_seqs : the list of sequences, each sequence is a list of tokens, and each token contains its features and its label 
        first_n_token : first n token in the article 
        NOTE : WE NOW JUST EXTRACT THE UNIGRAM TOKEN FEATURES WITHOUT ITS STRUCTURE INFORMATION
        NOTE : WE NEED TO BE MORE RIGOROUS WHEN MAP THE LABELS INTO TOKENS
        """

        self.crf_seqs = []#list of sequences
        
        for file_path in self.article_list:
            with open(file_path, "r") as f:
                crf_seq = []
                #process title
                s = f.readline()
                title = s[(s.find(">")+1):s.rfind("<")].split()#find title
                
                #process author 
                s = f.readline()
                authors = s[(s.find(">")+1):s.rfind("<")].replace(";", " ").split()#find authors and split them into tokens
                
                #find body
                while "<body>" not in s:
                    s = f.readline()
                f.readline()

                #process content 
                for i in range(100):
                    s = f.readline()
                    token = s[(s.find(">")+1):s.rfind("<")]
                    if any(e in token for e in head_ends):
                        break
                    x = self.extract_token_features(token)#extract token feature
                    #map label to the token
                    if token in authors:
                        x.append("authors")
                    elif token in title:
                        x.append("title")
                    else:
                        x.append("None")
                    crf_seq.append(x)
                self.crf_seqs.append(crf_seq)
        
