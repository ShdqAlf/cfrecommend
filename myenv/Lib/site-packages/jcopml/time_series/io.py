import pickle


def load_model(filepath):
    return pickle.load(open(filepath, "rb"))
