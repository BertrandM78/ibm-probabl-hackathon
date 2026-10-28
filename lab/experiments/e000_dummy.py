"""Floor: predict the train mean for every visit."""
import numpy as np

DESCRIPTION = "Dummy : moyenne du target"
FEATURES = "aucune"


def fit_predict(train, valid):
    return np.full(len(valid), train["target"].mean())
