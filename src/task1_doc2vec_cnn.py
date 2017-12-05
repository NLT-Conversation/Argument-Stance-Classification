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

topic_list_file = "task1_doc2vec_cnn_topic_list.pkl"
vectors_file = "task1_doc2vec_cnn_vectors.pkl"

def get_preprocessed_words(text, cachedStopWords):
    # filter chars except words and numbers
    tmp_text = re.sub(r'[^\w]', ' ', text)
    # filter stopwords
    return [word for word in tmp_text.split() if word not in cachedStopWords]

def preprocess(discussion_dict, topic_dict):
    # load stopwords list
    nltk.download('stopwords')
    cachedStopWords = stopwords.words("english")
    # load pretrained doc2vec model
    print("Load Associated Press News Skip-gram corpus")
    d2v_model_path = "doc2vec/apnews_dbow/doc2vec.bin"
    d2v_model = g.Doc2Vec.load(d2v_model_path)
    topic_list = list(topic_dict.keys())
    X = []
    y = []

    for topic_idx, topic in enumerate(topic_list):
        for dis_id in topic_dict[topic]:
            posts = discussion_dict[dis_id].get_posts_text()
            sentence_vecs = []
            for p in posts:
                words = get_preprocessed_words(p, cachedStopWords)
                vec = d2v_model.infer_vector(words)
                sentence_vecs.append(vec)
            X.append(sentence_vecs)
            y.append(topic_idx)
    pickle.dump(X, open(vectors_file, "wb"))
    pickle.dump(y, open(topic_list_file, "wb"))
    return X, y

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
        with open("cnn_topic.model.accuracy.txt", "a") as output:
            output.write("{},{}\n".format(epoch+1, accu))


if __name__ == "__main__":
    main()
