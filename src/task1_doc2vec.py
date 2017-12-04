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

topic_list_file = "task1_doc2vec_topic_list.txt"
vectors_file = "task1_doc2vec_vectors.txt"

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
    with open(topic_list_file, "w") as output:
        for topic in topic_list:
            output.write("{}\n".format(topic))
    with open(vectors_file, "w") as output:
        for topic_idx, topic in enumerate(topic_list):
            for dis_id in topic_dict[topic]:
                posts = discussion_dict[dis_id].get_posts_text()
                words = []
                for p in posts:
                    words += get_preprocessed_words(p, cachedStopWords)
                vec = d2v_model.infer_vector(words)
                X.append(vec)
                y.append(topic_idx)
                output.write("{},{},{}\n".format(
                    dis_id,
                    topic_idx,
                    ",".join([str(v) for v in vec])
                ))
    return X, y

def main():
    X = []
    y = []
    if os.path.exists(topic_list_file) and os.path.exists(vectors_file):
        with open(vectors_file, "r") as f:
            for line in f:
                items = line.strip().split(",")
                X.append(items[2:])
                y.append(items[1])
    else:
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
    print("Start training SVM model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv=5)
    print("5-fold cross validation: {}, mean: {}".format(scores, np.mean(scores)))
    with open("task1_doc2vec.csv", "w") as output:
        output.write("{},{}\n".format(",".join([str(s) for s in scores]), np.mean(scores)))

if __name__ == "__main__":
    main()
