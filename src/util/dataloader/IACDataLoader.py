import csv
import os
import json

from Topic import Topic
from Discussion import Discussion
from Post import Post

class IACDataLoader(object):
    def __init__(self):
        self.topic_dict = dict()
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
        print("Loading topic file...")
        self.topic_discussion_dict = dict()
        with open(self.topic_path, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(spamreader):
                if idx == 0:
                    continue
                d_id, topic = row
                topic = topic.replace('"', '')
                if topic not in self.topic_discussion_dict:
                    self.topic_discussion_dict[topic] = [d_id]
                else:
                    self.topic_discussion_dict[topic].append(d_id)

        print("Loading author stance file...")
        self.author_stance_dict = dict()
        with open(self.stance_path, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(spamreader):
                if idx == 0:
                    continue
                topic, discussion_id, author, pro, anti, other = row
                topic = topic.replace('"', '')
                author = author.replace('"', '')
                if author not in self.author_stance_dict:
                    self.author_stance_dict[author] = dict()
                self.author_stance_dict[author][discussion_id] = [pro, anti, other]

        print("Loading dataset files...")
        self.topic_dict = dict()
        self.discussion_dict = dict()
        idx = 0
        for idx, path in enumerate(os.listdir(self.dataset_dir)):
            if (idx+1)%1000 == 0:
                print("{0:d} dicussions were loaded".format(idx+1))
            # if idx > 3000:
            #     break
            filepath = os.path.join(self.dataset_dir, path)
            if filepath.endswith("json"):
                with open(filepath, 'rb') as f:
                    posts, _, metadata = json.load(f)
                    topics = metadata["breadcrumbs"]
                    discussion_id = metadata["id_number"]
                    title = metadata["title"]
                    # save posts
                    post_list = []
                    timestamp_list = []
                    for post in posts:
                        pid, _, author, text, _, quote_id, _, timestamp = post
                        timestamp_list.append(int(timestamp))
                        p = Post(pid, author, text, discussion_id, timestamp)
                        post_list.append(post_list)
                    # save discussion metatdata
                    for topic in topics:
                        if topic not in self.topic_dict:
                            self.topic_dict[topic] = Topic(topic)
                        self.topic_dict[topic].discussion_ids.append(discussion_id)
                    self.discussion_dict[discussion_id] = Discussion(discussion_id, title, topics, post_list, min(timestamp_list), max(timestamp_list))
        print("{0:d} dicussions were loaded".format(idx+1))
        print("Done!")

        # for topic in self.topic_dict:
        #     print("topic_dict: {0} - {1:d}".format(self.topic_dict[topic].get_name(), len(self.topic_dict[topic].get_discussion_ids())))
        # for topic in self.topic_discussion_dict:
        #     print("topic_discussion_dict: {0} - {1:d}".format(topic, len(self.topic_discussion_dict[topic])))
        # print("# discussions in topic_discussion_dict: {}".format(sum([len(self.topic_discussion_dict[topic]) for topic in self.topic_discussion_dict.keys()])))
        #
        # count = 0
        # min_post = 5
        # for author in self.author_stance_dict:
        #     if len(self.author_stance_dict[author]) > min_post:
        #         print("author_stance_dict: {0} - {1:d}".format(author, len(self.author_stance_dict[author])))
        #         count += 1
        # print("# authors in author_stance_dict (post>{:d}): {}".format(min_post, count))

    """
    get_topic_names

    Returns:
        a list of topic names
    """
    def get_topic_names(self):
        return self.topic_dict.keys()

    """
    get_topics

    Returns:
        a list of Topic objects
    """
    def get_topics(self):
        return self.topic_list
