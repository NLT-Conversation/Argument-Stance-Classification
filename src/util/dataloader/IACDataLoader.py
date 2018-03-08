import csv
import os
import json
import numpy as np

from util.dataloader.Topic import Topic
from util.dataloader.Discussion import Discussion
from util.dataloader.Post import Post

class IACDataLoader(object):
    def __init__(self):
        self.raw_topic_dict = dict()
        self.topic_discussion_dict = dict()
        self.author_stance_dict = dict()
        self.discussion_dict = dict()
        self.topic_path = None
        self.stance_path = None
        self.dataset_dir = None

    """
    set_dataset_dir

    Args:
        path: directory of dataset
    """
    def set_dataset_dir(self, dir):
        self.dataset_dir = dir

    """
    set_topic_filepath

    Args:
        path: filepath of topics
    """
    def set_topic_filepath(self, path):
        self.topic_path = path

    """
    set_stance_filepath

    Args:
        path: filepath of stance
    """
    def set_stance_filepath(self, path):
        self.stance_path = path

    def reject_outliers(self, d, m=2):
        d = np.array(d)
        stances = d[:,1].astype(float)
        return d[abs(stances - np.mean(stances)) < m * np.std(stances)]

    """
    load dataset, stance data

    """
    def load(self, remove_outlier=False):
        print("Loading dataset files...")
        self.raw_topic_dict = dict()
        self.discussion_dict = dict()
        idx = 0
        for idx, path in enumerate(os.listdir(self.dataset_dir)):
            if (idx+1)%5000 == 0:
                print("{0:d} dicussions were loaded".format(idx+1))
            # if idx > 2000:
            #     break
            filepath = os.path.join(self.dataset_dir, path)
            if filepath.endswith("json"):
                with open(filepath, 'r') as f:
                    posts, _, metadata = json.load(f)
                    topics = metadata["breadcrumbs"]
                    discussion_id = str(metadata["id_number"])
                    title = metadata["title"].strip()
                    # save posts
                    post_list = []
                    timestamp_list = []
                    for post in posts:
                        pid, _, author, text, _, quote_id, _, timestamp = post
                        timestamp_list.append(int(timestamp))
                        p = Post(pid, author, text, discussion_id, timestamp)
                        post_list.append(p)
                    # save discussion metatdata
                    for topic in topics:
                        if topic not in self.raw_topic_dict:
                            self.raw_topic_dict[topic] = Topic(topic)
                        self.raw_topic_dict[topic].discussion_ids.append(discussion_id)
                    self.discussion_dict[discussion_id] = Discussion(discussion_id, title, topics, post_list, min(timestamp_list), max(timestamp_list))
                    self.discussion_dict[discussion_id].authors = [post.get_author() for post in post_list]
                    self.discussion_dict[discussion_id].posts_text = [post.get_raw_text() for post in post_list]
        print("{0:d} dicussions were loaded".format(idx))

        print("Loading topic file...")
        self.topic_discussion_dict = dict()
        with open(self.topic_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(spamreader):
                if idx == 0:
                    continue
                d_id, topic = row
                topic = topic.replace('"', '').strip()
                if topic not in self.topic_discussion_dict:
                    self.topic_discussion_dict[topic] = [d_id]
                else:
                    self.topic_discussion_dict[topic].append(d_id)
                if d_id in self.discussion_dict:
                    self.discussion_dict[d_id].labeled_topic = topic

        print("Loading author stance file...")

        non_outlier_set = set()
        if remove_outlier:
            with open(self.stance_path, 'r') as csvfile:
                as_reader = csv.reader(csvfile)
                d_dict = dict()
                d_ids = []
                for idx, item in enumerate(as_reader):
                    if idx > 0:
                        topic,discussion_id,author,pro,anti,other = item
                        pro = int(pro)
                        anti = int(anti)
                        other = int(other)
                        stance = (pro-anti)/(pro+anti+other)
                        if discussion_id in d_dict:
                            d_dict[discussion_id].append([author, stance])
                        else:
                            d_ids.append(discussion_id)
                            d_dict[discussion_id] = [[author, stance]]
                discussion_dict_no_outlier = dict()
                for d_id in d_ids:
                    discussion_dict_no_outlier[d_id] = self.reject_outliers(d_dict[d_id])
                    for author_stance in discussion_dict_no_outlier[d_id]:
                        author, stance = author_stance
                        non_outlier_set.add("{}_{}".format(d_id, author))


        self.author_stance_dict = dict()
        with open(self.stance_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(spamreader):
                if idx == 0:
                    continue
                topic, discussion_id, author, pro, anti, other = row
                topic = topic.replace('"', '').strip()
                author = author.replace('"', '').strip()

                if remove_outlier:
                    key = "{}_{}".format(discussion_id, author)
                    if key not in non_outlier_set:
                        continue

                if author not in self.author_stance_dict:
                    self.author_stance_dict[author] = dict()
                self.author_stance_dict[author][discussion_id] = [pro, anti, other]


    """
    get_topic_names

    Returns:
        a list of topic names
    """
    def get_topic_names(self):
        return self.topic_discussion_dict.keys()

    """
    get_topic_dict

    Returns:
        topic dict (key: topic name, value: a list of discussion_ids)
    """
    def get_topic_dict(self):
        return self.topic_discussion_dict

    """
    get author_stance_dict

    Returns:
        author stance dict
        (key: author name, value: discussion stance dict of the author)
    """
    def get_author_stance_dict(self):
        return self.author_stance_dict

    """
    get_discussion_dict

    Returns:
        discussion dict (key: discussion_id, value: Discussion object)
    """
    def get_discussion_dict(self):
        return self.discussion_dict
