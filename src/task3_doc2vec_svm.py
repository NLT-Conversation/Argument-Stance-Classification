import collections
import smart_open
import random
import numpy as np
import pickle
import os
import nltk
import re
import timeit
import os

import pandas as pd

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

import matplotlib.pyplot as plt

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

    dataloader = iac.IACDataLoader()
    dataloader.set_dataset_dir("dataset/discussions")
    dataloader.set_topic_filepath("dataset/topic.csv")
    dataloader.set_stance_filepath("dataset/author_stance.csv")
    remove_outlier = True

    n_cv = 5
    train_columns = np.array(["remove_outlier"] + ["fold_{:d}".format(d+1) for d in list(range(n_cv))] + ["mean"])
    train_auccracy_df = pd.DataFrame(None, columns=train_columns)

    test_columns = np.array(["remove_outlier", "training_accuracy", "testing_accuracy"])
    test_auccracy_df = pd.DataFrame(None, columns=test_columns)

    for i in range(10):

        if not os.path.exists("results"):
            os.makedirs("results")
        result_folder = "results/{}/".format(i+1)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        for remove_outlier in [True, False]:
            print("Remove_outlier = {}".format(remove_outlier))
            dataloader.load(remove_outlier=remove_outlier)
            topic_dict = dataloader.get_topic_dict()
            discussion_dict = dataloader.get_discussion_dict()
            author_stance_dict = dataloader.get_author_stance_dict()

            topic_list = list(sorted(topic_dict.keys()))

            keys = sorted(author_stance_dict.keys())
            with open("author_stance_dict_{}.txt".format(remove_outlier), "w") as f:
                for key in keys:
                    f.write("{},{}\n".format(key, len(author_stance_dict[key].keys())))

            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for topic in topic_list:
                print("Load topic {}".format(topic))
                X_topic = []
                y_topic = []
                for dis_id in topic_dict[topic]:
                    authors = discussion_dict[dis_id].get_authors()
                    idx = 0
                    X_sub = []
                    y_sub = []
                    for idx, author in enumerate(authors):
                        if author in author_stance_dict and dis_id in author_stance_dict[author]:
                            posts = discussion_dict[dis_id].get_posts_by_author(author)
                            words = []
                            for post in posts:
                                tmp_text = post.get_raw_text()
                                words += get_preprocessed_words(tmp_text, cachedStopWords)
                            vec = d2v_model.infer_vector(words)
                            X_sub.append(vec)
                            y_sub.append(author_stance_dict[author][dis_id])
                    if len(X_sub)>0 and len(y_sub)>0:
                        X_topic.append(np.mean(X_sub, axis=0))
                        y_topic.append(get_class_label(np.sum(np.array(y_sub).astype(float), axis=0)))

                X_train_topic, X_test_topic, y_train_topic, y_test_topic = \
                    train_test_split(X_topic, y_topic, test_size=0.1)

                X_train += X_train_topic
                X_test += X_test_topic
                y_train += y_train_topic
                y_test += y_test_topic

            print("Number of training examples: {}, testing examples: {}".format(len(X_train), len(X_test)))

            clf = svm.SVC(kernel='linear', probability=True)

            # scores = cross_val_score(clf, X_train, y_train, cv=n_cv)
            # print("{}-fold cross validation: {}, mean: {}".format(n_cv, scores, np.mean(scores)))
            # new_row = np.array([remove_outlier] + scores.tolist() + [np.mean(scores)], dtype='str')
            # new_row = pd.DataFrame([new_row], columns=train_columns)
            # train_auccracy_df = train_auccracy_df.append(pd.DataFrame(new_row))
            # train_auccracy_df.to_csv("task3_pretrain_svm_cv={}_train_accuracy.csv".format(n_cv))

            clf.fit(X_train, y_train)
            new_row = np.array([remove_outlier, clf.score(X_train, y_train), clf.score(X_test, y_test)], dtype='str')
            new_row = pd.DataFrame([new_row], columns=test_columns)
            test_auccracy_df = test_auccracy_df.append(pd.DataFrame(new_row), ignore_index=True)
            test_auccracy_df.to_csv("task3_doc2vec_svm_test_accuracy.csv")

            # calculate the fpr and tpr for all thresholds of the classification
            probs = clf.predict_proba(X_test)
            for idx, y_t in enumerate(y_test):
                if y_t == 1:
                    y_test[idx] = [1,0,0]
                if y_t == 2:
                    y_test[idx] = [0,1,0]
                if y_t == 3:
                    y_test[idx] = [0,0,1]
            y_test = np.array(y_test)
            plt.figure()
            lw = 2
            for i in range(3):
                class_names = ["other", "pro", "anti"]
                fpr, tpr, threshold = metrics.roc_curve(y_test[:,i], probs[:,i])
                roc_auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         lw=lw, label='ROC curve for class {} (area = {:.2f})'.format(class_names[i], roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("{}task3_doc2vec_svm_auc_remove_outlier={}.png".format(result_folder, remove_outlier), dpi=200)

if __name__ == "__main__":
    main()
