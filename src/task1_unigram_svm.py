import util.dataloader.IACDataLoader as iac
import numpy as np
import pickle
import os
import nltk
import re
import timeit

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import cross_val_score

import stem_unigram_loader

def main():

    X_train, X_test, y_train, y_test = stem_unigram_loader.load_data(remove_outlier=False)

    # svm_model_path = "svm_model.pkl"
    clf = None
    # if not os.path.exists(svm_model_path):
    if True:
        print("Initialize SVM model")
        clf = svm.SVC(kernel='linear', C=1)
        print("Fit the model")
        clf.fit(X_train, y_train)
        print("Accuracy of testing data: {}".format(clf.score(X_test, y_test)))
        # print("Dump the model...")
        # joblib.dump(clf, open(svm_model_path, "wb"))
        print("Done!")
    else:
        print("Load pre-trained SVM model")
        # clf = joblib.load(open(svm_model_path, "rb"))

    if clf:
        # scores = cross_val_score(clf, X_train, y_train, cv=5)
        # print("5-fold cross validation: {}".format(scores))
        # with open("task1.csv", "w") as output:
        #     output.write("{},{}\n".format(",".join([str(s) for s in scores]), np.mean(scores)))
        print("Traing accuracy: {}".format(clf.score(X_train, y_train)))
        print("Testing accuracy: {}".format(clf.score(X_test, y_test)))

if __name__ == "__main__":
    main()
