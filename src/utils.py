from pathlib import Path

import numpy as np
import pandas as pd

SEED = 40

# data/processed/ of the project, found from this file's location, so it works from any notebook
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_data(path=PROCESSED):
    """Clean households, split into training (with CV fold) and test villages.

    Files are written by notebooks/01_data_prep.ipynb (steps 8 and 9).
    Use the test set only once, at the very end.
    """
    path = Path(path)
    households = pd.read_pickle(path / "households_clean.pkl")
    test_villages = pd.read_csv(path / "test_villages.csv")["hhea"]
    folds = pd.read_csv(path / "cv_folds.csv")

    train = households.merge(folds, on="hhea", how="inner")       # training villages + their fold
    test = households[households["hhea"].isin(test_villages)]     # used once, at the very end
    assert len(train) + len(test) == len(households), "a village is in neither or both parts"
    return train, test


def select_by_quota(data, score, higher_is_poorer):
    # share is a fraction of the village's households, so it is applied to the households actually
    # in the village: 8 in most villages, 7 in seven (see 01_data_prep). A fixed 8 would hand out
    # one transfer too many in three villages. data must contain whole villages (split and folds do).
    village_size = data.groupby("hhea")["hhea"].transform("size")
    n_slots = (village_size * data["share"]).round()
    rank = data.groupby("hhea")[score].rank(ascending=not higher_is_poorer, method="first")
    return (rank <= n_slots).astype(int)


def recall_at_quota(data, selected):
    poor_selected = ((selected == 1) & (data["poor"] == 1)).sum()
    all_poor = (data["poor"] == 1).sum()
    return poor_selected / all_poor


def within_village_auc(data, score, higher_is_poorer):
    correct = 0
    total = 0
    for village, g in data.groupby("hhea"):
        s = g[score] if higher_is_poorer else -g[score]
        s_poor = s[g["poor"] == 1].values
        s_nonpoor = s[g["poor"] == 0].values
        for a in s_poor:
            for b in s_nonpoor:
                total += 1
                if a > b:
                    correct += 1
                elif a == b:
                    correct += 0.5
    return correct / total