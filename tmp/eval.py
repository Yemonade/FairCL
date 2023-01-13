import itertools

import numpy as np
from sklearn import metrics

from collections import Counter

from sklearn.metrics import confusion_matrix


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

    def dp(self, grp_dict):
        """ Calculate the Demographic parity.
        P_{g_1}(pred=1) - P_{g_0}(pred=1)

        Args:
            grp_dict: the dict of dicts, containing the information about true label and predict label of each group,
                      e.g. {{"y": [...], "pred": [...]}, {...}}.

        Returns:
            dp, float, Demographic parity
        """

        if len(self.all_grp) == 2:
            dp = sum(grp_dict[1.]["pred"]) / self.grp_num[1.] - sum(grp_dict[0.]["pred"]) / self.grp_num[0.]
        else:
            dp = 0.
            for g_i, g_j in itertools.combinations(self.all_grp, 2):
                gap = sum(grp_dict[g_i]["pred"]) / self.grp_num[g_i] \
                      - sum(grp_dict[g_j]["pred"]) / self.grp_num[g_j]
                dp += abs(gap)

            dp /= self.normalized_factor

        return dp

    def eop(self, grp_dict):
        """ Calculate the Equal opportunity.
        P_{g_1, Y=1}(pred=1) - P_{g_0, Y=1}(pred=1)

        Args:
            grp_dict: the dict of dicts, containing the information about true label and predict label of each group,
                      e.g. {{"y": [...], "pred": [...]}, {...}}.

        Returns:
            eop, float, Equal opportunity
        """
        grp_y_1 = {}  # {0: [...], [...]}, the predict value of positive sample in each group.
        for g in self.all_grp:
            grp_y_1[g] = [pred for i, pred in enumerate(grp_dict[g]["pred"]) if grp_dict[g]["y"][i] == 1]

        if len(self.all_grp) == 2:
            eop = sum(grp_y_1[1]) / len(grp_y_1[1]) - sum(grp_y_1[0]) / len(grp_y_1[0])
        else:
            eop = 0.
            for g_i, g_j in itertools.combinations(self.all_grp, 2):
                gap = sum(grp_y_1[g_i]) / len(grp_y_1[g_i]) \
                      - sum(grp_y_1[g_j]) / len(grp_y_1[g_j])
                eop += abs(gap)

            eop /= self.normalized_factor

        return eop
        # for g in self.all_grp:
        #     grp_dict[g]["pred_cond_pos"] = [e for i, e in enumerate(grp_dict[g]["pred"]) if grp_dict[g]["y"][i] == 1]
        #
        # if len(self.all_grp) == 2:
        #     eop = sum(grp_dict[1.]["pred_cond_pos"]) / len(grp_dict[1.]["pred_cond_pos"]) \
        #           - sum(grp_dict[0.]["pred_cond_pos"]) / len(grp_dict[0.]["pred_cond_pos"])
        # else:
        #     eop = 0.
        #     all_grp = list(self.all_grp)
        #     for i, g_1 in enumerate(self.all_grp):
        #         for g_2 in all_grp[i + 1:]:
        #             gap = sum(grp_dict[g_1]["pred_cond_pos"]) / len(grp_dict[g_1]["pred_cond_pos"]) \
        #                   - sum(grp_dict[g_2]["pred_cond_pos"]) / len(grp_dict[g_2]["pred_cond_pos"])
        #             eop += abs(gap)
        #
        #     eop /= self.normalized_factor

        # return eop

    # def eo2(self, y_pred, y_real, s, privileged, unprivileged, labels):
    #     '''
    #     ABS Difference in True positive Rate between the two groups
    #     :param y_pred: prediction
    #     :param y_real: real label
    #     :param SensitiveCat: Sensitive feature name
    #     :param outcome: Outcome feature name
    #     :param privileged: value of the privileged group
    #     :param unprivileged: value of the unprivileged group
    #     :param labels: both priv-unpriv value for CFmatrix
    #     :return:
    #     '''
    #     y_priv = y_pred[s == privileged]
    #     y_real_priv = y_real[s == privileged]
    #     y_unpriv = y_pred[s == unprivileged]
    #     y_real_unpriv = y_real[s == unprivileged]
    #     TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv, y_priv, labels=labels).ravel()
    #     TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv, y_unpriv,
    #                                                                   labels=labels).ravel()
    #
    #     return abs(TP_unpriv / sum(y_real_unpriv) - TP_priv / sum(y_real_priv))

    def difference_average_odds(self, y, y_pred):
        """ Mean ABS difference in True positive rate and False positive rate of the two groups. """
        labels = np.unique(y)
        assert len(self.all_grp) == 2 and len(labels) == 2

        g0_y_pred = y_pred[self.s == 0]
        g0_y_true = y[self.s == 0]
        g1_y_pred = y_pred[self.s == 1]
        g1_y_true = y[self.s == 1]
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

        dp = self.dp(grp_dict)
        eop = self.eop(grp_dict)
        # eop2 = self.eo2(pred, y, self.s, 1,0, [0,1])
        if len(self.all_grp) == 2:
            difference_avg_odds = self.difference_average_odds(y, pred)
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
            if len(self.all_grp) == 2:
                if verbose:
                    print("Demographic parity: %.6f; Equal opportunity: %.6f; Average odds difference: %.6f"
                          % (dp, eop, difference_avg_odds))
                res = {"dp": dp, "eop": eop, "average_odds_difference": difference_avg_odds}
            else:
                if verbose:
                    print("Demographic parity: %.6f; Equal opportunity: %.6f" % (dp, eop))
                res = {"dp": dp, "eop": eop}
        else:
            if len(self.all_grp) == 2:
                if verbose:
                    print("Overall acc.: %.6f; Demographic parity: %.6f; "
                          "Equal opportunity: %.6f; Average odds difference: %.6f"
                          % (overall_acc, dp, eop, difference_avg_odds))
                res = {"overall_acc": overall_acc, "dp": dp, "eop": eop, "average_odds_difference": difference_avg_odds}
            else:
                if verbose:
                    print("Overall acc.: %.6f; Demographic parity: %.6f; Equal opportunity: %.6f"
                          % (overall_acc, dp, eop))
                res = {"overall_acc": overall_acc, "dp": dp, "eop": eop}
            res.update({"grp_%s_acc" % i: acc for i, acc in enumerate(group_acc)})

        return res
