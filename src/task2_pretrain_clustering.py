import collections
import smart_open
import random
import numpy as np
import pickle
import os
import nltk
import re
import timeit

import gensim.models as g

import util.dataloader.IACDataLoader as iac
import util.clustering.KmeansCosine as kmeans

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.cluster import adjusted_mutual_info_score

import os

def get_class_label(stance):
    pro, anti, other = stance
    if pro == anti or other == max(stance):
        return 1
    if pro == max(stance):
        return 2
    if anti == max(stance):
        return 3

def get_preprocessed_words(text, cachedStopWords):
    # filter chars except words and numbers
    tmp_text = re.sub(r'[^\w]', ' ', text)
    # filter stopwords
    return [word for word in tmp_text.split() if word not in cachedStopWords]

def main():
    # load stopwords list
    nltk.download('stopwords')
    cachedStopWords = stopwords.words("english")

    # load pretrained doc2vec model
    print("Load Associated Press News Skip-gram corpus")
    d2v_model_path = "doc2vec/apnews_dbow/doc2vec.bin"
    d2v_model = g.Doc2Vec.load(d2v_model_path)

    # init kmeans models
    cluster_model = kmeans.KmeansCosine(K=3)

    dataloader = iac.IACDataLoader()
    dataloader.set_dataset_dir("dataset/discussions")
    dataloader.set_topic_filepath("dataset/topic.csv")
    dataloader.set_stance_filepath("dataset/author_stance.csv")
    dataloader.load()
    topic_dict = dataloader.get_topic_dict()
    discussion_dict = dataloader.get_discussion_dict()
    author_stance_dict = dataloader.get_author_stance_dict()

    topic_list = list(topic_dict.keys())
    with open("task2_pretrain_clustering.csv", "w") as output:
        for topic in topic_list:
            X = []
            y = []
            print("==== Load discussions in topic {} ====".format(topic))
            for dis_id in topic_dict[topic]:
                authors = discussion_dict[dis_id].get_authors()
                idx = 0
                for idx, author in enumerate(authors):
                    if author in author_stance_dict and dis_id in author_stance_dict[author]:
                        posts = discussion_dict[dis_id].get_posts_by_author(author)
                        words = []
                        for post in posts:
                            tmp_text = post.get_raw_text()
                            words += get_preprocessed_words(tmp_text, cachedStopWords)
                            break
                        vec = d2v_model.infer_vector(words)
                        X.append(vec)
                        y.append(get_class_label(author_stance_dict[author][dis_id]))
            print("Start clustering")
            y_clustered = cluster_model.get_clusters(X)
            score = adjusted_mutual_info_score(y, y_clustered)
            print("Adjusted Mutual Infomation Score: {}".format(score))

            output.write("{},{}\n".format(topic, score))

if __name__ == "__main__":
    main()
