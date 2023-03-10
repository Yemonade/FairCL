import json
import os
import random
import numpy as np
import scipy.special
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from livelossplot import PlotLosses
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eval import Evaluator
from utils import EarlyStopping


class classifier_model(nn.Module):
    def __init__(self, feature, Hneuron1, output, dropout, seed1, seed2):
        super(classifier_model, self).__init__()
        self.feature = feature
        self.hN1 = Hneuron1
        self.output = output
        self.dropout = dropout
        self.seed1 = seed1
        self.seed2 = seed2
        self.FC1 = nn.Linear(self.feature, self.hN1)
        self.FC2 = nn.Linear(self.hN1, self.output)
        self.sigmoid = torch.sigmoid
        self.relu = F.relu
        self.Dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.Dropout(self.relu(self.FC1(x)))
        # print("x: ", x)
        # print("self.FC1.weight: ", self.FC1.bias.data)
        # print("="*50)
        x_logits = self.FC2(x)
        x_pred = self.sigmoid(x_logits)
        return x_pred, x_logits

    # 尝试一下两层的
    # def __init__(self, feature, Hneuron1, output, dropout, seed1, seed2):
    #     super(classifier_model, self).__init__()
    #     self.feature = feature
    #     self.hN1 = Hneuron1
    #     self.hN2 = Hneuron1 // 2
    #     self.output = output
    #     self.dropout = dropout
    #     self.FC1 = nn.Linear(self.feature, self.hN1)
    #     self.FC2 = nn.Linear(self.hN1, self.hN2)
    #     self.FC3 = nn.Linear(self.hN2, self.output)
    #
    #     self.sigmoid = torch.sigmoid
    #     self.relu = F.relu
    #     self.Dropout1 = nn.Dropout(p=self.dropout)
    #     self.Dropout2 = nn.Dropout(p=self.dropout)
    #
    # def forward(self, x):
    #     x = self.Dropout1(self.relu(self.FC1(x)))
    #     x = self.Dropout2(self.relu(self.FC2(x)))
    #     x_logits = self.FC3(x)
    #     x_pred = self.sigmoid(x_logits)
    #     return x_pred, x_logits


class adversary_model(nn.Module):
    def __init__(self, seed3, n_groups=1):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        self.c = torch.FloatTensor([1.0])
        self.FC1 = nn.Linear(3, n_groups)
        self.sigmoid = torch.sigmoid

    # see the paper for the detail
    def forward(self, pred_logits, true_labels):
        s = self.sigmoid((1 + torch.abs(self.c.to(pred_logits.device))) * pred_logits)
        pred_protected_attribute_logits = self.FC1(torch.cat([s, s * true_labels, s * (1.0 - true_labels)], 1))
        pred_protected_attribute_labels = self.sigmoid(pred_protected_attribute_logits)
        return pred_protected_attribute_labels, pred_protected_attribute_logits


class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    """Debiasing with adversarial learning.

    'Torch implementation of AIF360.adversarialdebiasing and fairer reproduction
    of Zhang et al. work.'

    Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [#zhang18]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [#zhang18] `B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating
           Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on
           Artificial Intelligence, Ethics, and Society, 2018.
           <https://dl.acm.org/citation.cfm?id=3278779>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            debiasing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            classifier.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the classifier.
        sess_ (tensorflow.Session): The TensorFlow Session used for the
            computations. Note: this can be manually closed to free up resources
            with `self.sess_.close()`.
        classifier_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the classifier.
        adversary_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the adversary.
    """

    def __init__(self, n_features, n_classes, n_groups, scope_name='classifier',
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=256, starter_learning_rate=0.001,
                 classifier_num_hidden_units=200, debias=True, verbose=False,
                 random_state=None):
        r"""
        Args:
            scope_name (str, optional): TensorFlow "variable_scope" name for the
                entire model (classifier and adversary).
            adversary_loss_weight (float or ``None``, optional): If ``None``,
                this will use the suggestion from the paper:
                :math:`\alpha = \sqrt(global_step)` with inverse time decay on
                the learning rate. Otherwise, it uses the provided coefficient
                with exponential learning rate decay.
            num_epochs (int, optional): Number of epochs for which to train.
            batch_size (int, optional): Size of mini-batch for training.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier.
            debias (bool, optional): If ``False``, learn a classifier without an
                adversary.
            verbose (bool, optional): If ``True``, print losses every 200 steps.
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for shuffling data and seeding weights.
        """

        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        self.random_state = random_state
        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_adv = self.loss_clf = F.binary_cross_entropy_with_logits

        # define the model
        rng = check_random_state(self.random_state)
        if self.random_state is not None:
            self.set_all_seed(self.random_state)
        else:
            self.set_all_seed(42)
        ii32 = np.iinfo(np.int32)
        self.s1, self.s2, self.s3 = rng.randint(ii32.min, ii32.max, size=3)

        self.batch_id = 0
        self.stopped_batch_ids = []

        # starter_learning_rate = 0.001
        self.clf_model = classifier_model(feature=n_features, Hneuron1=self.classifier_num_hidden_units,
                                          output=n_classes, dropout=0.2,
                                          seed1=self.s1, seed2=self.s2).to(self.device)
        self.init_parameters(self.clf_model)

        self.starter_learning_rate = starter_learning_rate
        self.n_groups = n_groups
        # self.classifier_opt = torch.optim.Adam(self.clf_model.parameters(), lr=starter_learning_rate, weight_decay=1e-5)
        # self.clf_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.classifier_opt,
        #                                                                    T_max=num_epochs)
        #
        if debias:
            self.adv_model = adversary_model(seed3=self.s3, n_groups=n_groups).to(self.device)
            self.init_parameters(self.adv_model)
        #     self.adversary_opt = torch.optim.Adam(self.adv_model.parameters(), lr=starter_learning_rate,
        #                                           weight_decay=1e-5)
        #     self.adv_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.adversary_opt,
        #                                                                        T_max=num_epochs)
        #
        # else:
        #     self.adv_model, self.adversary_opt = None, None

        self.logs = {}
        groups = {'accuracy': ['train_acc', 'val_acc'],
                  'loss': ['train_loss', 'val_loss']}
        # if self.debias:
        groups['dp'] = ['train_dp', 'val_dp']
        groups['eop'] = ['train_eop', 'val_eop']
        # groups['aod'] = ['train_aod', 'val_aod']
        self.liveloss = PlotLosses(groups=groups)


    def set_all_seed(self, seed):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_parameters(self, net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def fit(self, X, y, s, early_stopping=False, patience=10, validation_set=None):
        """ Train the classifier and adversary (if ``debias == True``) with the
        given training data.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
            s (array-like): Sensitive attributes

        Returns:
            self
        """

        self.classes_ = np.unique(y)
        # if early_stopping == True:
        self.patience = patience
        self.early_stopping = EarlyStopping(patience=patience)
        X_val, y_val, s_val = validation_set

        train_evaluator, val_evaluator = self.train_info(s, s_val)

        if scipy.sparse.issparse(X_val):
            X_val = X_val.todense()
        X_val = torch.tensor(X_val.astype(np.float32)).to(self.device)
        y_val = torch.tensor(y_val.astype(np.float32)).to(self.device)
        s_val = torch.tensor(s_val.astype(np.float32)).to(self.device)
        y_val = y_val.unsqueeze(1)
        s_val = s_val.unsqueeze(1)
        val_loss_list = []

        # else:
        # train_evaluator, _ = self.train_info(s)

        if scipy.sparse.issparse(X):
            X = X.todense()
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        y = torch.tensor(y.astype(np.float32)).to(self.device)
        s = torch.tensor(s.astype(np.float32)).to(self.device)
        y = y.unsqueeze(1)
        s = s.unsqueeze(1)

        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(starter_learning_rate,
        #                                                                decay_steps=1000, decay_rate=0.96,
        #                                                                staircase=True)
        # classifier_opt = tf.optimizers.Adam(learning_rate)
        # classifier_vars = [var for var in self.clf_model.trainable_variables]

        train_loss_list = []
        # batch_list = []
        train_eval_list = []
        val_eval_list = []
        self.batch_id = 0
        # plt.ion()
        dataBatch = DataLoader(TensorDataset(X, y, s), batch_size=self.batch_size, shuffle=True,
                               drop_last=False)

        # optimizer
        self.classifier_opt = torch.optim.Adam(self.clf_model.parameters(), lr=self.starter_learning_rate, weight_decay=1e-5)
        # self.classifier_opt = torch.optim.LBFGS(self.clf_model.parameters(), lr=self.starter_learning_rate)

        self.clf_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.classifier_opt,
                                                                           T_max=self.num_epochs)

        # self.clf_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.classifier_opt, 'min')


        # self.clf_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.classifier_opt, T_0=2, T_mult=2)

        if self.debias:
            self.adversary_opt = torch.optim.Adam(self.adv_model.parameters(), lr=self.starter_learning_rate,
                                                  weight_decay=1e-5)
            self.adv_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.adversary_opt,
                                                                                  T_max=self.num_epochs)

            # self.adv_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.adversary_opt, T_0=2, T_mult=2)

        else:
            self.adv_model, self.adversary_opt = None, None

        if self.debias:

            # decayRate = 0.96
            # adv_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adversary_opt, gamma=decayRate)
            # with tqdm(range(self.num_epochs // 2)) as epochs:
            #     epochs.set_description("Classifcation PreTraining Epoch")
            #     for epoch in epochs:
            #         self.clf_model.train()
            #         for X_b, y_b, s_b in dataBatch:
            #             classifier_opt.zero_grad()
            #             pred_labels, pred_logits = self.clf_model.forward(X_b)
            #             # print("pred_labels: ", pred_labels)
            #             # print("y_b: ", y_b)
            #             loss = self.loss_clf(pred_logits, y_b, reduction='mean')
            #             loss.backward()
            #             classifier_opt.step()
            #
            #             clf_lr_scheduler.step()
            #
            #             acc_b = (pred_labels.round() == y_b).float().sum().item()/X_b.size(0)
            #             epochs.set_postfix(loss=loss.item(), acc=acc_b)
            #
            # with tqdm(range(10)) as epochs:
            #     epochs.set_description("Adversarial PreTraining Epoch")
            #     for epoch in epochs:
            #         self.adv_model.train()
            #         self.clf_model.eval()
            #         for X_b, y_b, s_b in dataBatch:
            #             adversary_opt.zero_grad()
            #             pred_labels, pred_logits = self.clf_model.forward(X_b)
            #             pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
            #                 pred_logits, y_b)
            #             loss = self.loss_adv(pred_protected_attributes_logits, s_b, reduction='mean')
            #             loss.backward()
            #             adversary_opt.step()
            #             adv_lr_scheduler.step()
            #
            #             acc_b = (pred_protected_attributes_labels.round() == s_b).float().sum().item()/X_b.size(0)
            #             epochs.set_postfix(loss=loss.item(), acc=acc_b)

            # clf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=classifier_opt, gamma=decayRate)
            # adv_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adversary_opt, gamma=decayRate)

            with tqdm(range(self.num_epochs), colour='green') as epochs:
                epochs.set_description("Adversarial Debiasing Training Epoch")
                for epoch in epochs:
                    for X_b, y_b, s_b in dataBatch:
                        self.adv_model.train()
                        self.clf_model.train()
                        self.classifier_opt.zero_grad()
                        self.adversary_opt.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(X_b)
                        loss1 = self.loss_clf(pred_logits, y_b, reduction='mean')
                        loss1.backward(retain_graph=True)
                        # dW_LP
                        clf_grad = [torch.clone(par.grad.detach()) for par in self.clf_model.parameters()]

                        self.classifier_opt.zero_grad()
                        self.adversary_opt.zero_grad()

                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, y_b)
                        loss2 = self.loss_adv(pred_protected_attributes_logits, s_b, reduction='mean')
                        loss2.backward()
                        print("loss1: ", loss1.item(), "loss2: ", loss2.item())
                        # dW_LA
                        adv_grad = [
                            torch.clone(par.grad.detach()) for par in self.clf_model.parameters()
                        ]

                        for i, par in enumerate(self.clf_model.parameters()):
                            # Normalization
                            unit_adversary_grad = adv_grad[i] / (torch.norm(adv_grad[i]) + torch.finfo(float).tiny + 1e-8)
                            # projection proj_{dW_LA}(dW_LP)
                            proj = torch.sum(torch.inner(unit_adversary_grad, clf_grad[i]))
                            # integrating into the CLF gradient
                            par.grad = clf_grad[i] - (proj * unit_adversary_grad) - (
                                    self.adversary_loss_weight * adv_grad[i])

                        self.classifier_opt.step()
                        # optimizing dU_LA
                        self.adversary_opt.step()

                        acc_adv = (pred_protected_attributes_labels.round() == s_b).float().sum().item() / X_b.size(0)
                        acc_clf = (pred_labels.round() == y_b).float().sum().item() / X_b.size(0)
                        epochs.set_postfix(lossCLF=loss1.item(), lossADV=loss2.item(), accCLF=acc_clf,
                                           accADV=acc_adv)

                        self.batch_id += 1
                        # train_loss_list.append((loss1 + self.adversary_loss_weight * loss2).item())
                        if self.batch_id % 50 == 0:
                            # train_loss_list.append((loss1 + self.adversary_loss_weight * loss2).item())
                            # batch_list.append(self.batch_id)
                            # plt.plot(batch_list, loss_list, 'r-')
                            # plt.xlabel('batch num')
                            # plt.ylabel('loss')
                            # plt.title("loss")
                            # plt.pause(0.1)


                            with torch.no_grad():
                                # train loss
                                self.clf_model.eval()
                                self.adv_model.eval()
                                # total_loss_train = (loss1 + self.adversary_loss_weight * loss2).item()

                                pred_labels_train, pred_logits_train = self.clf_model.forward(X)
                                loss1 = self.loss_clf(pred_logits_train, y, reduction='mean')
                                pred_protected_attributes_labels_train, pred_protected_attributes_logits_train = self.adv_model.forward(
                                    pred_logits_train, y)
                                loss2 = self.loss_adv(pred_protected_attributes_logits_train, s, reduction='mean')
                                total_loss_train = (loss1 + self.adversary_loss_weight * loss2).item()
                                train_loss_list.append(total_loss_train)

                                # val loss
                                pred_labels_val, pred_logits_val = self.clf_model.forward(X_val)
                                loss1 = self.loss_clf(pred_logits_val, y_val, reduction='mean')
                                pred_protected_attributes_labels_val, pred_protected_attributes_logits_val = self.adv_model.forward(
                                    pred_logits_val, y_val)
                                loss2 = self.loss_adv(pred_protected_attributes_logits_val, s_val, reduction='mean')
                                total_loss_val = (loss1 + self.adversary_loss_weight * loss2).item()
                                val_loss_list.append(total_loss_val)

                                # evaluate on train and val
                                pred_label_train = self.predict(X.squeeze(1).detach().numpy())
                                train_res = train_evaluator(y.squeeze(1).detach().numpy(), pred_label_train,
                                                            no_train=False, verbose=False)
                                train_eval_list.append(train_res)

                                pred_label_val = self.predict(X_val.squeeze(1).detach().numpy())
                                val_res = val_evaluator(y_val.squeeze(1).detach().numpy(), pred_label_val,
                                                        no_train=False, verbose=False)
                                val_eval_list.append(val_res)

                                self.logs['train_loss'] = total_loss_train
                                self.logs['val_loss'] = total_loss_val
                                self.logs['train_acc'] = train_res['overall_acc']
                                self.logs['val_acc'] = val_res['overall_acc']

                                self.logs['train_dp'] = train_res['dp']
                                self.logs['val_dp'] = val_res['dp']
                                self.logs['train_eop'] = train_res['eop']
                                self.logs['val_eop'] = val_res['eop']
                                # self.logs['train_aod'] = train_res['average_odds_difference']
                                # self.logs['val_aod'] = val_res['average_odds_difference']

                                self.liveloss.update(self.logs)
                                self.liveloss.send()

                                if early_stopping:
                                    self.early_stopping(total_loss_train, self)
                                    if self.early_stopping.early_stop:
                                        break
                    if early_stopping and self.early_stopping.early_stop:
                        break
                    self.clf_lr_scheduler.step()
                    self.adv_lr_scheduler.step()

            self.stopped_batch_ids.append(self.batch_id)

            state = {
                # 'clf_model': self.clf_model.state_dict(),
                # 'adv_model': self.adv_model.state_dict(),
                'clf_optimizer': self.classifier_opt.state_dict(),
                'adv_optimizer': self.adversary_opt.state_dict(),
                # 'epoch': epoch + 1,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')  # 2 、 建立一个保存参数的文件夹
            torch.save(state, './checkpoint/clf_adv_optimizer_state.ckpt')
        else:
            with tqdm(range(self.num_epochs), colour='green') as epochs:
                epochs.set_description("Classifier Training Epoch")
                for epoch in epochs:
                    for X_b, y_b, s_b in dataBatch:
                        self.clf_model.train()
                        self.classifier_opt.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(X_b)
                        loss = self.loss_clf(pred_logits, y_b, reduction='mean')
                        loss.backward()
                        self.classifier_opt.step()

                        acc_b = (pred_labels.round() == y_b).float().sum().item() / X_b.size(0)
                        epochs.set_postfix(loss=loss.item(), acc=acc_b)

                        self.batch_id += 1
                        if self.batch_id % 50 == 0:
                            # loss_list.append(loss.item())
                            # batch_list.append(self.batch_id)
                            # plt.plot(batch_list, loss_list, 'r-')
                            # plt.xlabel('batch num')
                            # plt.ylabel('loss')
                            # plt.title("loss")
                            # plt.pause(0.1)
                            with torch.no_grad():
                                self.clf_model.eval()
                                # train loss
                                pred_labels_train, pred_logits_train = self.clf_model.forward(X)
                                loss1_train = self.loss_clf(pred_logits_train, y, reduction='mean')
                                train_loss_list.append(loss1_train.item())

                                # val loss
                                pred_labels_val, pred_logits_val = self.clf_model.forward(X_val)
                                loss1_val = self.loss_clf(pred_logits_val, y_val, reduction='mean')
                                val_loss_list.append(loss1_val.item())
                                
                                # evaluate on train and val
                                pred_label_train = self.predict(X.squeeze(1).detach().numpy())
                                train_res = train_evaluator(y.squeeze(1).detach().numpy(), pred_label_train,
                                                            no_train=False, verbose=False)
                                train_eval_list.append(train_res)

                                pred_label_val = self.predict(X_val.squeeze(1).detach().numpy())
                                val_res = val_evaluator(y_val.squeeze(1).detach().numpy(), pred_label_val,
                                                        no_train=False, verbose=False)
                                val_eval_list.append(val_res)

                                self.logs['train_loss'] = loss1_train
                                self.logs['val_loss'] = loss1_val
                                self.logs['train_acc'] = train_res['overall_acc']
                                self.logs['val_acc'] = val_res['overall_acc']

                                self.logs['train_dp'] = train_res['dp']
                                self.logs['val_dp'] = val_res['dp']
                                self.logs['train_eop'] = train_res['eop']
                                self.logs['val_eop'] = val_res['eop']
                                # self.logs['train_aod'] = train_res['average_odds_difference']
                                # self.logs['val_aod'] = val_res['average_odds_difference']

                                self.liveloss.update(self.logs)
                                self.liveloss.send()

                                if early_stopping:
                                    self.early_stopping(-val_res['overall_acc'])
                                    if self.early_stopping.early_stop:
                                        break

                    if early_stopping and self.early_stopping.early_stop:
                        break

                    # val loss
                    # with torch.no_grad():
                    # pred_labels_val, pred_logits_val = self.clf_model.forward(X_val)
                    # loss1_val = self.loss_clf(pred_logits_val, y_val, reduction='mean')
                    self.clf_lr_scheduler.step()

            self.stopped_batch_ids.append(self.batch_id)
            
            state = {
                # 'clf_model': self.clf_model.state_dict(),
                # 'adv_model': self.adv_model.state_dict(),
                'clf_optimizer': self.classifier_opt.state_dict(),
                # 'epoch': epoch + 1,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')  # 2 、 建立一个保存参数的文件夹
            torch.save(state, './checkpoint/clf_optimizer_state.ckpt')
        # plt.ioff()

        return train_loss_list, val_loss_list, train_eval_list, val_eval_list

    def train_info(self, s_train, s_val=None):
        train_evaluator = Evaluator(s_train, "train")
        if s_val is not None:
            val_evaluator = Evaluator(s_val, "val")
            return train_evaluator, val_evaluator
        return train_evaluator, None

    def decision_function(self, X):
        """Soft prediction scores.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Confidence scores per (sample, class) combination. In
            the binary case, confidence score for ``self.classes_[1]`` where >0
            means this class would be predicted.
        """
        if scipy.sparse.issparse(X):
            X = X.todense()
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        # n_classes = len(self.classes_)

        # if n_classes == 2:
        #    n_classes = 1 # lgtm [py/unused-local-variable]

        # self.clf_model.eval()
        pred_labels_list = []
        dataBatch = DataLoader(X, batch_size=self.batch_size, shuffle=False,
                               drop_last=False)
        for X_b in dataBatch:
            self.clf_model.eval()
            pred_labels, pred_logits = self.clf_model.forward(X_b)
            pred_labels_list += pred_labels.cpu().detach().numpy().tolist()

        scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        decision = self.decision_function(X)

        if decision.ndim == 1:
            decision_2d = np.c_[np.zeros_like(decision), decision]
        else:
            decision_2d = decision
        return scipy.special.softmax(decision_2d, axis=1)

    def predict(self, X):
        """Predict class labels for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            if X.shape[0] == 1:
                indices = (scores > 0.5).astype(np.int).reshape((-1,))
            else:
                indices = (scores > 0.5).astype(np.int).reshape((-1,))
        else:
            indices = scores.argmax(axis=1)

        return self.classes_[indices]

    def sorted_loss(self, X, y, s, idx_path):
        """ return the loss of each sample"""
        print("========== sort and save ==========")
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        y = torch.tensor(y.astype(np.float32)).to(self.device)
        s = torch.tensor(s.astype(np.float32)).to(self.device)

        y = y.unsqueeze(1)
        s = s.unsqueeze(1)
        with torch.no_grad():
            self.clf_model.eval()
            if self.debias:
                self.adv_model.eval()

            pred_labels, pred_logits = self.clf_model.forward(X)
            loss1 = self.loss_clf(pred_logits, y, reduction='none')

            if self.debias:
                pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                    pred_logits, y)
                loss2 = self.loss_adv(pred_protected_attributes_logits, s, reduction='none')
                total_loss_list = torch.flatten((loss1 + self.adversary_loss_weight * loss2))
            else:
                total_loss_list = torch.flatten(loss1)
            print(total_loss_list)

            sorted_loss_idx_value = sorted(enumerate(total_loss_list), key=lambda x: x[1])
            idx = [i[0] for i in sorted_loss_idx_value]

            with open(idx_path, "w") as f:
                json.dump(idx, f)

            return
