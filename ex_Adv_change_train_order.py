from dataset import fetch_data
from models.AdversarialDebiasing import AdversarialDebiasing
from eval import Evaluator
from utils import get_curriculum_stages

if __name__ == "__main__":
    # get the dataset
    data = fetch_data("adult")
    print("data.x_train.shape: ", data.x_train.shape)
    print("data.x_test.shape: ", data.x_test.shape)
    origin_evaluator, train_evaluator, test_evaluator = Evaluator(data.s_train, "origin"), Evaluator(data.s_train, "train"), Evaluator(data.s_test, "test")
    if data.s_val is not None:
        val_evaluator = Evaluator(data.s_val, "val")
    print("========== before train ==========")
    origin_res = origin_evaluator(data.y_train, no_train=True)

    print('\n========== Starting Training without Mitigation... ==========')
    # clf_no_debias = AdversarialDebiasing(num_epochs=100, batch_size=256,
                                         # classifier_num_hidden_units=256, random_state=42, debias=False)

    # clf_no_debias.fit(data.x_train, data.y_train, data.s_train)

    # print("========== after train(without debiasing) ==========")
    # pred_label_train = clf_no_debias.predict(data.x_train)
    # train_res = train_evaluator(data.y_train, pred_label_train, no_train=False)
    #
    # pred_label_test = clf_no_debias.predict(data.x_test)
    # test_res = test_evaluator(data.y_test, pred_label_test, no_train=False)


    print('\n========== Starting Training with Mitigation... ==========')
    clf = AdversarialDebiasing(adversary_loss_weight=0.1, num_epochs=500, batch_size=512,
                               classifier_num_hidden_units=512, random_state=42, debias=True)

    clf.fit(data.x_train, data.y_train, data.s_train)
    print("========== after train(with debiasing) ==========")
    pred_label_train = clf.predict(data.x_train)
    train_res = train_evaluator(data.y_train, pred_label_train, no_train=False)

    pred_label_test = clf.predict(data.x_test)
    test_res = test_evaluator(data.y_test, pred_label_test, no_train=False)

    clf.loss(data.x_train, data.y_train, data.s_train, idx_path='data/adult/sorted_idx.json')

    if False:
        clf = AdversarialDebiasing(adversary_loss_weight=0.1, num_epochs=100, batch_size=256,
                               classifier_num_hidden_units=256, random_state=42, debias=True)
        stages = get_curriculum_stages('data/adult/sorted_idx.json', N=3)
        for stage in stages:
            x_train = data.x_train[stage, :]
            y_train = data.y_train[stage]
            clf.fit_again(x_train, y_train)
