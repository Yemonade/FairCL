import itertools

import numpy as np
from fairlearn.metrics import MetricFrame, equalized_odds_difference, selection_rate
from sklearn import metrics

from collections import Counter

from sklearn.metrics import confusion_matrix, accuracy_score


class Evaluator():
    """ Evaluate the accuracy and fairness on demographic parity and equal opportunity
    Attributes:
        s: ndarray, containing all the sensitive attributes (categorical or binary)
        name: str, the name of evaluation
        all_grp: set, the sorted set of sensitive attributes
        grp_num: dict, e.g., {0: number, [...]}
        normalized_factor: int, used to calculate the mean loss over all classes

        We default set 1 as the protected group, e.g., Male and
        also default set 1 as the positive label.
    """

    def __init__(self, s, name):
        """ Inits Evalutor with all the sensitive attributes and the name. """
        self.s = s
        self.name = name

        self.all_grp = sorted(set(self.s))
        self.grp_num = Counter(self.s)
        self.normalized_factor = len(self.all_grp) * (len(self.all_grp) - 1) / 2.

    def generate_grp_dict(self, y, pred):
        """ Generate the dict of dicts about the information about true label and predict label of each group,
            e.g. {g0: {"y": [...], "pred": [...]}, g1: {...}}.

        Args:
            y: ndarray, the true label
            pre: ndarray, the predictive label

        Returns:
            grp_dict, the list of dict
        """
        grp_dict = {g: {} for g in self.all_grp}
        for g in self.all_grp:
            grp_dict[g]["y"] = [e for i, e in enumerate(y) if self.s[i] == g]
            grp_dict[g]["pred"] = [e for i, e in enumerate(pred) if self.s[i] == g]
        # for g, v in grp_dict.items():
        #     print("%d: {num_y_1=%d, num_pred_1=%d}" % (g, sum(v["y"]), sum(v["pred"])))
        return grp_dict
    @staticmethod
    def acc(y, pred):
        """ Calculate the accuracy. """
        return metrics.accuracy_score(y, pred)

    @staticmethod
    def dp(y_true, y_pred, s):
        gm = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=s)
        return gm.group_max() - gm.group_min()

    @staticmethod
    def eop(y_true, y_pred, s):
        return equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=s)

    @staticmethod
    def difference_average_odds(y_true, y_pred, s):
        """ Mean ABS difference in True positive rate and False positive rate of the two groups. """

        g0_y_pred = y_pred[s == 0]
        g0_y_true = y_true[s == 0]
        g1_y_pred = y_pred[s == 1]
        g1_y_true = y_true[s == 1]
        g0_TN, g0_FP, g0_FN, g0_TP = confusion_matrix(g0_y_true, g0_y_pred, labels=[0,1]).ravel()
        g1_TN, g1_FP, g1_FN, g1_TP = confusion_matrix(g1_y_true, g1_y_pred, labels=[0,1]).ravel()

        # print(g0_TP + g0_FN, sum(g0_y_true))
        # print(g1_TP + g1_FN, sum(g1_y_true))
        return 0.5 * (abs(g0_FP / sum(g0_y_true) - g1_FP / sum(g1_y_true)) +
                      abs(g0_TP / sum(g0_y_true) - g1_TP / sum(g1_y_true)))



    def __call__(self, y, pred=None, no_train=True, verbose=True):
        """ Evaluate the models.
        Args:
            y: ndarray, the true label
            pre: ndarray, the predictive label

        Returns:
            res: dict, the score of accuracy, dp, eoo and group-wise accuracy
        """
        if pred is not None:
            no_train = False
        elif no_train:
            pred = y

        assert len(y) == len(pred)

        grp_dict = self.generate_grp_dict(y, pred)

        dp = Evaluator.dp(y, pred, self.s)
        eop = Evaluator.eop(y, pred, self.s)

        # if len(self.all_grp) == 2:
        #     difference_avg_odds = self.difference_average_odds(y, pred)
        overall_acc = self.acc(y, pred)
        if verbose:
            print("=" * 10, "Results on %s" % self.name, "=" * 10)
        group_acc = []
        for g in self.all_grp:
            if no_train:
                if verbose:
                    print("Grp. %d - #instance: %d; #pos : %d" %
                          (g, self.grp_num[g], sum(grp_dict[g]["pred"])))
            else:
                g_acc = self.acc(grp_dict[g]["y"], grp_dict[g]["pred"])
                group_acc.append(g_acc)
                if verbose:
                    print("Grp. %d - #instance: %d; #pos. pred: %d; Acc.: %.6f" %
                      (g, self.grp_num[g], sum(grp_dict[g]["pred"]), g_acc))
        if no_train:
            if verbose:
                print("Demographic parity: %.6f; Equal opportunity: %.6f"
                      % (dp, eop))
            res = {"dp": dp, "eop": eop}
        else:
            if verbose:
                print("Overall acc.: %.6f; Demographic parity: %.6f; "
                      "Equal opportunity: %.6f"
                      % (overall_acc, dp, eop))
            res = {"overall_acc": overall_acc, "dp": dp, "eop": eop}
            res.update({"grp_%s_acc" % i: acc for i, acc in enumerate(group_acc)})

        return res
