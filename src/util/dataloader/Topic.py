class Topic(object):
    def __init__(self, name):
        self.name = name
        self.discussion_ids = []
    """
    get_name

    Returns:
        the name of this topic
    """
    def get_name(self):
        return self.name

    """
    get_discussions

    Returns:
        a list of Discussion objects
    """
    def get_discussion_ids(self):
        return self.discussion_ids
