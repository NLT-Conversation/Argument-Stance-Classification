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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def preprocess(discussion_dict, topic_dict):
    unigram_dict_path = "unigram_dict.pickle"
    discussions_unigram_label_dict_path = "discussions_unigram_label_dict.txt"
    topic_list = sorted(topic_dict.keys())

    print("===== Start preprocessing =====")
    # load stopwords list
    nltk.download('stopwords')
    cachedStopWords = stopwords.words("english")
    # load Porter stemmer
    stemmer = PorterStemmer()

    # Collect unigram list
    unigram_dict = dict()
    if not os.path.exists(unigram_dict_path):
        print("Generate unigram_dict ... ")
        topics = topic_dict.keys()
        for topic in topics:
            print("Start processing topic {}".format(topic))
            discussion_ids = topic_dict[topic]
            idx = 0
            for idx, discussion_id in enumerate(discussion_ids):
                if (idx+1) % 100 == 0:
                    print("{}/{} discussions were processed".format(idx+1, len(discussion_ids)))
                discussion = discussion_dict[discussion_id]
                posts_text = discussion.get_posts_text()
                for post_text in posts_text:
                    # filter chars except words and numbers
                    tmp_text = re.sub(r'[^\w]', ' ', post_text)
                    # filter stopwords and perform Porter stemmer
                    tmp_text = ' '.join([stemmer.stem(word) for word in tmp_text.split() if word not in cachedStopWords])
                    words = tmp_text.split(' ')
                    for word in words:
                        if word in unigram_dict:
                            unigram_dict[word] += 1
                        else:
                            unigram_dict[word] = 1
            print("{}/{} discussions were processed".format(idx+1, len(discussion_ids)))
        pickle.dump(unigram_dict, open(unigram_dict_path, "wb"))
        with open("unigram_dict.txt", "w", encoding='utf-8') as output:
            for key in unigram_dict.keys():
                output.write("{}\t{}\n".format(key, unigram_dict[key]))
    else:
        print("Load unigram_dict ... ")
        unigram_dict = pickle.load(open(unigram_dict_path, "rb"))

    # Generate unigram vector for discussions
    X = []
    y = []
    unigram_list = sorted(unigram_dict.keys())
    with open("unigram_list.txt", "w", encoding='utf-8') as output:
        for u in unigram_list:
            output.write("{}\n".format(u))
    discussions_unigram_label_dict = dict()
    if not os.path.exists(discussions_unigram_label_dict_path):
        print("Generate unigram vector for discussions ...")
        topics = topic_dict.keys()
        for topic in topics:
            print("Start processing topic {}".format(topic))
            discussion_ids = topic_dict[topic]
            idx = 0
            for idx, discussion_id in enumerate(discussion_ids):
                if (idx+1) % 10 == 0:
                    print("{}/{} discussions were processed".format(idx+1, len(discussion_ids)))
                word_dict = dict()
                discussion = discussion_dict[discussion_id]
                posts_text = discussion.get_posts_text()
                for post_text in posts_text:
                    # filter chars except words and numbers
                    tmp_text = re.sub(r'[^\w]', ' ', post_text)
                    # filter stopwords and perform Porter stemmer
                    tmp_text = ' '.join([stemmer.stem(word) for word in tmp_text.split() if word not in cachedStopWords])
                    words = tmp_text.split(' ')
                    for word in words:
                        if word in word_dict:
                            word_dict[word] += 1
                        else:
                            word_dict[word] = 1
                unigram_vec = [0] * len(unigram_list)
                w_keys = word_dict.keys()
                for w_key in w_keys:
                    unigram_vec[unigram_list.index(w_key)] = word_dict[w_key]
                discussions_unigram_label_dict[discussion_id] = [unigram_vec, topic]
            print("{}/{} discussions were processed".format(idx+1, len(discussion_ids)))
        with open("discussions_unigram_label_dict.txt", "w") as output:
            for d_id in discussions_unigram_label_dict.keys():
                vec, label = discussions_unigram_label_dict[d_id]
                vec_str = ','.join([str(v) for v in vec])
                output.write("{},{},{}\n".format(d_id, label, vec_str))

        with open("discussions_unigram_label_dict.txt", "r") as f:
            idx = 0
            for idx, line in enumerate(f):
                if (idx+1)%1000 == 0:
                    print("{} discussions were loaded".format(idx+1))
                # if idx>=300:
                #     break
                items = line.strip().split(',')
                X.append(items[2:])
                y.append(topic_list.index(items[1]))
            print("{} discussions were loaded".format(idx))
    else:
        print("Load discussions_unigram_label_dict ... ")
        with open("discussions_unigram_label_dict.txt", "r") as f:
            idx = 0
            for idx, line in enumerate(f):
                if (idx+1)%1000 == 0:
                    print("{} discussions were loaded".format(idx+1))
                # if idx>=300:
                #     break
                items = line.strip().split(',')
                X.append(items[2:])
                y.append(topic_list.index(items[1]))
            print("{} discussions were loaded".format(idx))

    print("===== Done! =====")

    return X, y


def main():
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

    #dimension reduction

    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(X)
    with open("X_TSNE.txt", "w") as output:
        output.write(str(X_tsne))
    plt.scatter(X_tsne[:, 0], c=y)

    print("Divide data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, stratify=y, test_size=0.25)
    svm_model_path = "svm_model.pkl"
    clf = None
    if not os.path.exists(svm_model_path):
        print("Initialize SVM model")
        clf = svm.SVC(kernel='linear', C=1)
        print("Fit the model")
        clf.fit(X_train, y_train)
        print("Accuracy of testing data: {}".format(clf.score(X_test, y_test)))
        print("Dump the model...")
        joblib.dump(clf, open(svm_model_path, "wb"))
        print("Done!")
    else:
        print("Load pre-trained SVM model")
        clf = joblib.load(open(svm_model_path, "rb"))
        # clf = svm.SVC(kernel='linear', C=1)
        if clf:
            scores = cross_val_score(clf, X, y, cv=5)
            print("5-fold cross validation: {}".format(scores))
            with open("task1.csv", "w") as output:
                output.write("{},{}\n".format(",".join([str(s) for s in scores]), np.mean(scores)))

if __name__ == "__main__":
    main()
