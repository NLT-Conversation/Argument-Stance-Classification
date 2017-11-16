class IACDataLoader(object):
    def __init__(self):
        pass
    """
    set_dataset_filepath

    Args:
        path: filepath of dataset
    """
    def set_dataset_filepath(self, path):
        pass

    """
    set_stance_filepath

    Args:
        path: filepath of stance
    """
    def set_stance_filepath(self, path):
        pass

    """
    load dataset, stance data

    """
    def load(self):
        pass

    """
    get_topic_names

    Returns:
        a list of topic names
    """
    def get_topic_names(self):
        return []

    """
    get_topics

    Returns:
        a list of Topic objects
    """
    def get_topics(self):
        return []
