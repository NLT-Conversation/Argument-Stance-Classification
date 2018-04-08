import util.dataloader.IACDataLoader as iac
import numpy as np
import pickle
import os
import nltk
import re
import timeit
import gensim.models as g

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score

import doc2vec_loader

def main():
    model_para_path = "cnn_topic.model.pt"
    num_vecs = 300
    X_train, X_test, y_train, y_test = doc2vec_loader.load_data(max_num_vecs=num_vecs)

    print("Start training SVM model")
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv=5)
    print("5-fold cross validation: {}, mean: {}".format(scores, np.mean(scores)))
    with open("task1_doc2vec_svm.csv", "w") as output:
        output.write("{},{}\n".format(",".join([str(s) for s in scores]), np.mean(scores)))

if __name__ == "__main__":
    main()
