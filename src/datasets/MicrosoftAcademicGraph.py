import sys
import glob
import numpy as np
from _base import Dataset

"""
This is the class that handles all of the operations related to the Microsoft Academic Graph.
"""

#header separating words
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
        """
        Map the tokens with its tags (author, title ...)
        """
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
    
    def extract_testing_knowledge_base_features(self, output_path, first_n_token = 30, article_range = (0, None)):
        """
        Extract the features from the articles, these features are used by non-structured classifiers 
        , e.g. NaiveBayes, when "TESTING".
        Since we only have the knowledge base when training, we may not extract structure features.
        But we should separate different articles because the predicted results should be used by CRF!! 
        We write them into the CRFPP format.
        """

        kb_seqs = []#list of sequences
        
        for file_path in self.article_list[article_range[0]: article_range[1]]:
            with open(file_path, "r") as f:
                #print file_path
                truths = {}
                kb_seq = []
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
                    kb_seq.append(x)
                if flag == 0:
                    continue
                try:
                    tags = self._map_tags(truths, tokens)
                except:
                    continue
                for i, token in enumerate(kb_seq):
                    token.append(tags[i])
                kb_seqs.append(kb_seq)
            self.cnt += 1
        #write to the file
        self.write_crfpp_data(output_path, kb_seqs)
    
    def _info_feature(self, s):
        """
        Extract the content features from the PDF
        e.g. font, top ....
        And quantize them (convert them into bins)
        Please refer to the content of the PDF (after parsing ) to see the detailed format
        """
        
        f = []
        #ID
        ID = s[s.find("id=\"p")+5:s.find("\" style")]
        if len(ID) > 4:
            ID =  "0"
        f.append(ID)
        
        #top
        top = s[s.find("top:")+4: s.find("top:")+8]
        try:
            top = str(int(float(top) ))
        except:
            top = "-1"
        f.append(top)

        #left
        left = s[s.find("left:")+5:s.find("left:")+9]
        try:
            left = str(int(float(left)/10 ))

        except:
            left = "-1"
        f.append(left)
        
        #line height
        lh = s[s.find("line-height:")+12:s.find("line-height:")+16]
        try:
            lh = str(int(float(lh) ))
        except:
            lh = "-1"
        f.append(lh)
        
        #font family
        ff = s[s.find("font-family:")+12:s.find("font-family:")+17]
        if len(ID) > 5:
            ff =  "0"
        f.append(ff)
        
        #font size
        fs = s[s.find("font-size:")+10:s.find("font-size:")+12]
        try:
            fs = str(int(float(fs)/5 ))
        except:
            fs = "-1"
        f.append(fs)
        
        #font weight
        fw = s[s.find("font-weight:")+12:s.find("font-weight:")+16]
        if len(fw) != "bold":
            fw =  "0"
        f.append(fw)

        #eidth
        width = s[s.find("width:")+6:s.find("width")+10]
        try:
            width = str(int(float(width)/10 ))
        except:
            width = "-1"
        f.append(width)
        return f

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
                    if any(e in token for e in head_ends):#if there's any header separating words
                        flag = 1
                        break
                    x = self._extract_token_features(token)#extract token features
                    #map label to the token
                    tokens.append(token)
                    
                    """If you would like use PDF features, pelase remove # """
                    #F = self._info_feature(s)
                    #x += F
                    #print F, file_path

                    crf_seq.append(x)
                if flag == 0:# If find no header separating words
                    continue

                try:#try mapping
                    tags = self._map_tags(truths, tokens)
                except:
                    continue
                for i, token in enumerate(crf_seq):
                    token.append(tags[i])
                crf_seqs.append(crf_seq)
            self.cnt += 1
        #write to the file
        self.write_crfpp_data(output_path, crf_seqs)
