import os

from settings import DATA_FOLDER


def data(path, *paths):
    """
    Converts relative path to data into absolute one.
    """
    path = os.path.join(path, *paths)
    filename = os.path.abspath(
        os.path.join(DATA_FOLDER, os.path.expandvars(path)))
    if not os.path.exists(filename):
        return None
    return filename
