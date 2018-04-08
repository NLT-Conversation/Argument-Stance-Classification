import util.dataloader.IACDataLoader as iac
import numpy as np
import pickle
import os
import nltk
import re
import timeit
import gensim.models as g

from torch.autograd import Variable
import torch

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score

import doc2vec_loader
import util.classification.cnn_topic as cnn

def main():
    model_para_path = "cnn_topic.model.pt"
    num_vecs = 300
    X_train, X_test, y_train, y_test = doc2vec_loader.load_data(max_num_vecs=num_vecs)

    nn = cnn.CnnTopic(num_vec=num_vecs)
    nn.cuda()

    for epoch in range(50):
        if os.path.exists(model_para_path):
            print("{} is found. Loading CNN model".format(model_para_path))
            nn.load_state_dict(torch.load(model_para_path))

        print("Training CNN model, epoch: {}".format(epoch+1))
        nn.set_training_data(X_train=X_train, y_train=y_train)
        nn.train()
        torch.save(nn.state_dict(), model_para_path)

        y_pred = []
        for idx, x in enumerate(X_test):
            x = Variable(torch.FloatTensor(x).cuda())
            y_pred.append(nn.FromOneHot(nn.forward(x).data[0]))
        accu = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        with open("task1_doc2vec_cnn.txt", "a") as output:
            output.write("{},{}\n".format(epoch+1, accu))


if __name__ == "__main__":
    main()
