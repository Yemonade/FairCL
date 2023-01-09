import numpy as np

from dataset import fetch_data
from models.AdversarialDebiasing import AdversarialDebiasing
from eval import Evaluator
import torch
from utils import get_curriculum_stages

if __name__ == "__main__":
    # get the dataset
    data = fetch_data("adult")
    print("data.x_train.shape: ", data.x_train.shape)
    print("data.x_test.shape: ", data.x_test.shape)
    origin_evaluator, train_evaluator, test_evaluator = Evaluator(data.s_train, "origin"), Evaluator(data.s_train, "train"), Evaluator(data.s_test, "test")
    if data.s_val is not None:
        val_evaluator = Evaluator(data.s_val, "val")

    n_features, n_classes, n_groups = data.x_train.shape[1], len(np.unique(data.y_train)), len(np.unique(data.s_train))
    if n_classes == 2:
        n_classes = 1
    if n_groups == 2:
        n_groups = 1

    print("========== before train ==========")
    origin_res = origin_evaluator(data.y_train, no_train=True)

    print('\n========== Starting Training without Mitigation... ==========')
    # clf_no_debias = AdversarialDebiasing(n_features, n_classes, n_groups, num_epochs=100, batch_size=256,
    #                                      classifier_num_hidden_units=256, random_state=42, debias=False)
    # clf_no_debias = AdversarialDebiasing(n_features, n_classes, n_groups, num_epochs=500, batch_size=512,
    #                      classifier_num_hidden_units=512, random_state=42, debias=False)
    #
    # clf_no_debias.fit(data.x_train, data.y_train, data.s_train)
    #
    # print("========== after train(without debiasing) ==========")
    # pred_label_train = clf_no_debias.predict(data.x_train)
    # train_res = train_evaluator(data.y_train, pred_label_train, no_train=False)
    #
    # pred_label_test = clf_no_debias.predict(data.x_test)
    # test_res = test_evaluator(data.y_test, pred_label_test, no_train=False)


    print('\n========== Starting Training with Mitigation... ==========')
    clf = AdversarialDebiasing(n_features, n_classes, n_groups, adversary_loss_weight=0.1, num_epochs=500, batch_size=512,
                               classifier_num_hidden_units=512, random_state=42, debias=True)

    clf.fit(data.x_train, data.y_train, data.s_train)
    print("========== after train(with debiasing) ==========")
    with torch.no_grad():
        clf.clf_model.eval()
        clf.adv_model.eval()

        pred_label_train = clf.predict(data.x_train)
        train_res = train_evaluator(data.y_train, pred_label_train, no_train=False)

        if data.s_val is not None:
            pred_label_val = clf.predict(data.x_val)
            train_res = val_evaluator(data.y_val, pred_label_val, no_train=False)

        pred_label_test = clf.predict(data.x_test)
        test_res = test_evaluator(data.y_test, pred_label_test, no_train=False)

    order_and_save_idx = True
    if order_and_save_idx:
        clf.sorted_loss(data.x_train, data.y_train, data.s_train, idx_path='data/adult/sorted_idx_%d.json' % data.num_val)

    if False:
        clf = AdversarialDebiasing(adversary_loss_weight=0.1, num_epochs=100, batch_size=256,
                               classifier_num_hidden_units=256, random_state=42, debias=True)
        stages = get_curriculum_stages('data/adult/sorted_idx.json', N=3)
        for stage in stages:
            x_train = data.x_train[stage, :]
            y_train = data.y_train[stage]
            s_train = data.s_train[stage]
            clf.fit(x_train, y_train, s_train)
