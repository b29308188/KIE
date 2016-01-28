import sys
import os 
import numpy as np
from _base import Model
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
class LogisticRegression(Model):
    """
    This class inherits the Model class from the ModelBase module.
    It uses the Multinomial Naive Bayes from the sklearn.
    Features are encoded by the one-hot encoding.
    """
    def __init__(self):
        #self.model = LR(C = 1) 
        self.model = SVC(C = 10, kernel = "linear") 
        self.feature_encoder = DictVectorizer()
        self.tag_encoder = LabelEncoder()
        
    def _seqs_to_train(self, seqs):
        """
        Convert the list of sequences into the training data format in sklearn.
        (X, Y) = (2-d array, 1-d array)
        """
        X = []
        Y = []
        for seq in seqs:
            for token in seq:
                features = token.strip().split()
                X.append({i:xi for (i, xi) in enumerate(features[:-1]) } )#features        
                Y.append(features[-1])#label

        #convert features of strings into a binary array by the one-hot encoder 
        X = self.feature_encoder.fit_transform(X).toarray()
        
        #convert the string label into a number
        Y = self.tag_encoder.fit_transform(Y)
        
        return (csr_matrix(X), Y)
    
    def fit(self, train_path):
        """
        Train on the input CRFPP file and store the model.
        """
        seqs = self.read_seqs(train_path)
        (X, Y) = self._seqs_to_train(seqs)
        self.model.fit(X, Y)
    
    def predict(self, test_path, pred_path, last_feature = False):
        """
        Test on the test_path (CRFPP format) and write the prediction in pred_path (CRFPP format).
        """
        seqs = self.read_seqs(test_path)
        with open(pred_path, "w") as f:
            for seq in seqs:
                #for each sequence (article), predict it separately
                #so We can separate the suquences when writing our predictions into the CRFPP file
                X = []
                for token in seq:
                    features = token.strip().split()
                    if last_feature is False:
                        X.append({i:xi for (i, xi) in enumerate(features[:-1]) } )           
                    else:
                        X.append({i:xi for (i, xi) in enumerate(features[:]) } )           
                X = self.feature_encoder.transform(X).toarray()
                Y = self.model.predict(csr_matrix(X))
            
                #write the features and the prediction into the ifle
                for (i, token) in enumerate(seq):
                    if last_feature is False:
                        for xi in token.strip().split()[:-1]:#features
                            f.write(xi+ " ")
                    else:
                        for xi in token.strip().split()[:]:#features
                            f.write(xi+ " ")
                    f.write(self.tag_encoder.inverse_transform(Y[i]) + "\n")#prediction
                f.write("\n")#the end of a sequence

    def cross_valid(self, data_path, n_fold = 5):
        """
        Do the cross_validation on the input CRFPP file.
        """
        seqs = self.read_seqs(data_path)
        (X, Y) = self._seqs_to_train(seqs)
        return np.mean(cross_val_score(self.model, X, Y, cv = 5))
