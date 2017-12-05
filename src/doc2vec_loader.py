import util.dataloader.IACDataLoader as iac
import numpy as np
import pickle
import os
import nltk
import re
import timeit
import gensim.models as g
import random

from torch.autograd import Variable
import torch

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score

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
                sentence_vecs.append(vec.tolist())
            X.append(sentence_vecs)
            y.append(topic_idx)
    pickle.dump(X, open(vectors_file, "wb"))
    pickle.dump(y, open(topic_list_file, "wb"))
    return X, y

def load_data(test_size=0.25, max_num_vecs=100):
    X = []
    y = []
    if os.path.exists(topic_list_file) and os.path.exists(vectors_file):
        print("doc2vec files are found. Loading doc2vec vectors")
        X = pickle.load(open(vectors_file, "rb"))
        y = pickle.load(open(topic_list_file, "rb"))
    else:
        print("building doc2vec vectors")
        dataloader = iac.IACDataLoader()
        dataloader.set_dataset_dir("dataset/discussions")
        dataloader.set_topic_filepath("dataset/topic.csv")
        dataloader.set_stance_filepath("dataset/author_stance.csv")
        dataloader.load()
        topic_dict = dataloader.get_topic_dict()
        discussion_dict = dataloader.get_discussion_dict()
        author_stance_dict = dataloader.get_author_stance_dict()
        topic_list = sorted(topic_dict.keys())
        X, y = preprocess(discussion_dict, topic_dict)

    # max_num_vecs = max([len(x) for x in X])
    vec_size = len(X[0][0])

    print("Padding zero rows to docs")
    for idx, x in enumerate(X):
        len_x = len(x)
        if len_x < max_num_vecs:
            for i in range(len_x, max_num_vecs):
                X[idx].append([0]*vec_size)
        else:
            sampled_indices = random.sample(list(range(len(X[idx]))), max_num_vecs)
            X[idx] = [X[idx][i] for i in sampled_indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def main():
    pass
if __name__ == "__main__":
    main()
