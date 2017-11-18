class Post(object):
    def __init__(self, id, author, raw_text, discussion_id, timestamp):
        self.id = id
        self.author = author
        self.raw_text = raw_text
        self.parent_id = discussion_id
        self.timestamp = timestamp
    """
    get_id

    Returns:
        the post id
    """
    def get_id(self):
        return self.id

    """
    get_author

    Returns:
        the name of author
    """
    def get_author(self):
        return self.author

    """
    get_raw_text

    Returns:
        the plaintext of this post
    """
    def get_raw_text(self):
        return self.raw_text

    """
    get_dicussion_id

    Returns:
        the parent discussion id
    """
    def get_dicussion_id(self):
        return self.parent_id

    """
    get_timestamp

    Returns:
        the timestamp
    """
    def get_timestamp(self):
        return self.timestamp
