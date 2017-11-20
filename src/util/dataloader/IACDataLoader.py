import csv
import os
import json

from Topic import Topic
from Discussion import Discussion
from Post import Post

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

    """
    load dataset, stance data

    """
    def load(self):
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
                with open(filepath, 'rb') as f:
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
        print("{0:d} dicussions were loaded".format(idx+1))

        print("Loading topic file...")
        self.topic_discussion_dict = dict()
        with open(self.topic_path, 'rb') as csvfile:
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
        self.author_stance_dict = dict()
        with open(self.stance_path, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(spamreader):
                if idx == 0:
                    continue
                topic, discussion_id, author, pro, anti, other = row
                topic = topic.replace('"', '').strip()
                author = author.replace('"', '').strip()
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
