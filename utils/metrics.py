import numpy as np


def get_confmat_metrics(confusion_matrix):
    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)  # TP/P
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)  # TP/T
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1