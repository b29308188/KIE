import sys
import glob
import numpy as np
from _base import Dataset

"""
This is the class that handles all of the operations related to the Microsoft Academic Graph.
"""
head_ends = ["Introduction", "introduction", "INTRODUCTION", "Abstract", "ABSTRACT", "abstract", "OUTLINE, ""outline", "Outline"]

class MicrosoftAcademicGraph(Dataset):
    def __init__(self, prefix):
        """
        prefix : the MicrosoftAcademicGraph folder that contains HTML files
        article_list : the list of the HTML files
        article_num : the number of HMTL files under the specified folder_pather
        """
        
        self.prefix = prefix
        self.article_list = [file_path for file_path in glob.glob(prefix+"/*")]
        self.article_num = len(self.article_list)
        self.cnt = 0
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

    def _map_tags(self, truths, tokens):
        tags = ["None" for i in range(len(tokens))]
        for tag in truths.keys():
            tagged_tokens = truths[tag]
            #mapping 
            for (i, token) in enumerate(tokens):
                if tags[i] == "None" and token in tagged_tokens:
                    tags[i] = tag
            #smoothing
            for i in range(1, len(tags) - 1):
                if tags[i-1] == tags[i+1]:
                    tags[i] = tags[i-1]                
            #remove noise
            candidates = []
            start = -1
            for i, t in enumerate(tags):
                if t == tag and start == -1:
                    start = i
                elif start != -1 and t != tag:
                    candidates.append((start, i))
                    start = -1
            kept = np.argmax([c[1] - c[0] for c in candidates])
            for i, c in enumerate(candidates):
                if i != kept:
                    for i in range(c[0], c[1]+1):
                        tags[i] = "None"
        return tags
        
    def extract_training_knowledge_base_features(self, output_path, article_range = (0, None)):
        """
        Extract the features from the knowledge base, these features are used by non-structured classifiers 
        , e.g. NaiveBayes, when "TRAINING".
        Since we only have the knowledge base, we may not extract structure features.
        We don't need to separate sequences becuase we may not have different sequences in the knwoledge base.
        However, for the consistent issues, we still write them into the CRFPP format.
        """
        #for Cora, there's no knowledge base, so we have to extract them from the articles
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

    def extract_crf_features(self, output_path, first_n_token = 30, article_range = (0, None)):
        """
        Extract the features of the first n tokens in the articles(HTML files), and this features are used for CRF.
        We store them into sequences (seqs), and the groups of tokens in the same article is a sequence
        crf_seqs : the list of sequences, each sequence is a list of tokens, and each token contains its features and its label 
        first_n_token : first n token in the article 
        NOTE : WE NOW JUST EXTRACT THE UNIGRAM TOKEN FEATURES WITHOUT ITS STRUCTURE INFORMATION
        """

        crf_seqs = []#list of sequences
        
        for file_path in self.article_list[article_range[0]: article_range[1]]:
            with open(file_path, "r") as f:
                #print file_path
                truths = {}
                crf_seq = []
                #process title
                s = f.readline()
                truths['title'] = s[(s.find(">")+1):s.rfind("<")].split()#find title
                
                #process author 
                s = f.readline()
                truths['author'] = s[(s.find(">")+1):s.rfind("<")].replace(";", " ").split()#find authors and split them into tokens
                
                #find body
                while "<body>" not in s:
                    s = f.readline()
                f.readline()
                tokens = []
                #process content
                flag = 0
                for i in range(first_n_token):
                    s = f.readline()
                    token = s[(s.find(">")+1):s.rfind("<")]
                    if any(e in token for e in head_ends):
                        flag = 1
                        break
                    x = self._extract_token_features(token)#extract token feature
                    #map label to the token
                    tokens.append(token)
                    crf_seq.append(x)
                if flag == 0:
                    continue
                try:
                    tags = self._map_tags(truths, tokens)
                except:
                    continue
                for i, token in enumerate(crf_seq):
                    token.append(tags[i])
                crf_seqs.append(crf_seq)
            self.cnt += 1
        #write to the file
        self.write_crfpp_data(output_path, crf_seqs)
