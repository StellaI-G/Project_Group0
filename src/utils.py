import numpy as np
import pandas as pd

SEED = 40


def select_by_quota(data, score, higher_is_poorer):
    n_slots = (8 * data["share"]).round()
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