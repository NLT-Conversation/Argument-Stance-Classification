class Discussion(object):
    def __init__(self, id, title, topics, post_list, min_timestamp, max_timestamp):
        self.id = id
        self.title = title
        self.topics = topics
        self.labeled_topic = None
        self.post_list = post_list
        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp
        self.authors = []
        self.posts_text = []
    """
    get_id

    Returns:
        the discussion id
    """
    def get_id(self):
        return self.id

    """
    get_title

    Returns:
        the discussion title
    """
    def get_title(self):
        return self.title

    """
    get_authors

    Returns:
        a list of author id and name
    """
    def get_authors(self):
        return self.authors

    """
    get_posts

    Returns:
        a list of Post
    """
    def get_posts(self):
        return self.post_list

    """
    get_topics

    Returns:
        a list of Topics
    """
    def get_topics(self):
        return self.topics

    """
    get_labeled_topic

    Returns:
        topic labeled by dataset researchers
    """
    def get_labeled_topic(self):
        return self.labeled_topic

    """
    get_posts_by_author

    Returns:
        a list of Post written by an author
    """
    def get_posts_by_author(self, author):
        return [post for post in self.post_list if post.get_author() == author]

    """
    get_posts_text

    Returns:
        a list of Post text
    """
    def get_posts_text(self):
        return self.posts_text

    """
    get_start_timestamp

    Returns:
        the timestamp of first post
    """
    def get_start_timestamp(self):
        return self.min_timestamp

    """
    get_end_timestamp

    Returns:
        the timestamp of end post
    """
    def get_end_timestamp(self):
        return self.max_timestamp
