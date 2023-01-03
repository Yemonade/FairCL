import json
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter, defaultdict

from sklearn.compose import ColumnTransformer


class DataTemplate:
    def __init__(self, x_train, y_train, s_train, x_val, y_val, s_val, x_test, y_test, s_test):
        self.num_train = x_train.shape[0]
        self.num_val = x_val.shape[0] if x_val is not None else 0
        self.num_test = x_test.shape[0]
        self.dim = x_train.shape[1]
        self.num_s_feat = len(Counter(s_train))

        self.x_train = x_train
        self.y_train = y_train
        self.s_train = s_train
        self.x_val = x_val
        self.y_val = y_val
        self.s_val = s_val
        self.x_test = x_test
        self.y_test = y_test
        self.s_test = s_test

        print("Dataset statistic - #total: %d; #train: %d; #val.: %d; #test: %d; #dim.: %.d\n"
              % (self.num_train + self.num_val + self.num_test,
                 self.num_train, self.num_val, self.num_test, self.dim))


class Dataset:
    """Generate dataset
    Assure in a binary group case, Grp. 1 is the privileged group and Grp. 0 is the unprivileged group
    Assure in a binary label case, 1. is the positive outcome and 0. is the negative outcome
    Sensitive feature is not excluded from data
    """

    def __init__(self, name, df, target_feat, sensitive_feat, test_df=None, categorical_feat=None,
                 drop_feat=None, label_mapping=None, shuffle=False, load_idx=True, idx_path=None,
                 test_p=0.10, val_p=0.01, *args, **kwargs):
        """
        Arguments:
            name: dataset name
            df: dataset DataFrame
            target_feat: feature to be predicted
            sensitive_feat: sensitive feature
            test_df: DataFrame for testing, optional
            categorical_feat: categorical features to be processed into one-hot encoding
            drop_feat: features to drop
            label_mapping: mapping for one-hot encoding for some features
                i.e., {"Sex": {"Male": 0, "Female": 1}, [...]}
            shuffle: shuffle the dataset
            load_idx: loading shuffled row index
            idx_path: path for the shuffled index file
            test_p: proportion of test data
            val_p: proportion of validation data
        """

        print("Loading %s dataset.." % name)

        self.categorical_feat = categorical_feat if categorical_feat is not None else []
        self.num_feat = df.columns.difference(self.categorical_feat)
        self.categorical_feat.remove(sensitive_feat)
        self.categorical_feat.remove(target_feat)


        # shuffle the rows
        if shuffle:
            if load_idx and os.path.exists(idx_path):
                with open(idx_path) as f:
                    shuffle_idx = json.load(f)
            else:
                shuffle_idx = np.random.permutation(df.index)
                with open(idx_path, "w") as f:
                    json.dump(shuffle_idx.tolist(), f)

            df = df.reindex(shuffle_idx)

        # split the dataset to train_val and test
        if test_df is None:
            df = self.preprocessing(df, drop_feat, label_mapping, sensitive_feat, target_feat)
            num_test = round(len(df) * test_p)
            num_train_val = len(df) - num_test
            train_val_df = df.iloc[:num_train_val]
            test_df = df.iloc[num_train_val:]
        else:
            train_val_num = df.shape[0]
            total_df = pd.concat([df, test_df])
            total_df = self.preprocessing(total_df, drop_feat, label_mapping, sensitive_feat, target_feat)
            train_val_df = total_df.iloc[: train_val_num, :]
            test_df = total_df.iloc[train_val_num:, :]

        print("train_val_df.shape: ", train_val_df.shape)
        print("test_df.shape: ", test_df.shape)
        # get sensitive columns after mapping
        s_train_val, s_test = train_val_df[sensitive_feat].to_numpy(), test_df[sensitive_feat].to_numpy()

        train_val_df.drop(columns=sensitive_feat, inplace=True)
        test_df.drop(columns=sensitive_feat, inplace=True)

        # separate target feature from dataset
        y_train_val, y_test = train_val_df[target_feat].to_numpy(), test_df[target_feat].to_numpy()
        train_val_df, test_df = train_val_df.drop(columns=target_feat), test_df.drop(columns=target_feat)

        # split the train_val to train and validation
        num_val = round(len(train_val_df) * val_p)
        num_train = len(train_val_df) - num_val

        if num_val != 0:
            x_train, x_val = train_val_df.iloc[:num_train], train_val_df.iloc[num_train:]
            self.y_train, self.y_val = y_train_val[:num_train], y_train_val[num_train:]
            self.s_train, self.s_val = s_train_val[:num_train], s_train_val[num_train:]

        else:
            x_train, x_val = train_val_df, None
            self.y_train, self.y_val = y_train_val, None
            self.s_train, self.s_val = s_train_val, None

        self.y_test, self.s_test = y_test, s_test

        # transform
        # total = 0
        # for c in self.categorical_feat:
        #     n = x_train[c].nunique()
        #     print("%s: %d" % (c, n))
        #     total += n
        # print("total: ", total)

        self.x_train, scaler = self.transform(x_train)
        if num_val != 0:
            self.x_val, _ = self.transform(x_val, scaler)
        else:
            self.x_val = None
        self.x_test, _ = self.transform(test_df, scaler)

    def preprocessing(self, df, drop_feat, label_mapping, sensitive_feat, target_feat):
        # drop useless columns
        df.dropna(inplace=True)
        if drop_feat is not None and len(drop_feat) != 0:
            drop_feat = drop_feat.split(",")
            for drop_f in drop_feat:
                if drop_f in self.categorical_feat:
                    self.categorical_feat.remove(drop_f)
            df.drop(columns=drop_feat, inplace=True)

        # map sensitive feature and target
        if label_mapping is not None:
            assert sensitive_feat, target_feat in label_mapping
            df[sensitive_feat] = df[sensitive_feat].map(label_mapping[sensitive_feat])
            df[target_feat] = df[target_feat].map(label_mapping[target_feat])

        # df = pd.get_dummies(df, columns=self.categorical_feat, prefix_sep='=')

        return df

    def transform(self, X, scaler=None):
        if scaler is None:
            transformations = ColumnTransformer(
                transformers=[
                    ('num', preprocessing.StandardScaler(), self.num_feat),
                    ('cat', preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_feat)
                ])

            scaler = transformations.fit(X)
        scaled = scaler.transform(X)

        return scaled, scaler


    @property
    def data(self):
        return DataTemplate(self.x_train, self.y_train, self.s_train,
                            self.x_val, self.y_val, self.s_val,
                            self.x_test, self.y_test, self.s_test)


class AdultDataset(Dataset):
    """ https://archive.ics.uci.edu/ml/datasets/adult """

    def __init__(self):
        meta = json.load(open("data/adult/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"], header=0, names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        # remove the "." at the end of each "income"
        test["income"] = [e[:-1] for e in test["income"].values]
        # train = self.custom_preprocessing(train)
        # test = self.custom_preprocessing(test)

        # meta["categorical_feat"].extend(["Education Years", "Age (decade)"])
        # print("meta[categorical_feat]:", meta["categorical_feat"])
        super(AdultDataset, self).__init__(name="Adult", df=train, test_df=test, **meta, shuffle=False)

    def custom_preprocessing(self, df):
        # # Group age by decade
        # df['Age (decade)'] = df['age'].apply(lambda x: x // 10 * 10)
        #
        # # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)
        #
        # def group_edu(x):
        #     if x <= 5:
        #         return 0
        #     elif x >= 13:
        #         return 1
        #     else:
        #         return 2
        #
        # def age_cut(x):
        #     if x >= 70:
        #         return 1
        #     else:
        #         return 0
        #
        # def group_race(x):
        #     if x == "White":
        #         return 1.0
        #     else:
        #         return 0.0
        #
        # df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        # df['Education Years'] = df['Education Years'].astype('category')
        #
        # df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))
        # df['race'] = df['race'].apply(lambda x: group_race(x))
        #
        # df.drop(columns=['age', 'education-num', 'education', 'marital-status', 'relationship', 'native-country'], inplace=True)

        return df

class CompasDataset(Dataset):
    """ https://github.com/propublica/compas-analysis """

    def __init__(self):
        meta = json.load(open("data/compas/meta.json"))

        df = pd.read_csv(meta["train_path"], index_col='id')
        df = self.default_preprocessing(df)
        df = df[meta["features_to_keep"].split(",")]

        super(CompasDataset, self).__init__(name="Compas", df=df, **meta, shuffle=False)

    @staticmethod
    def default_preprocessing(df):
        """
        Perform the same preprocessing as the original analysis:
        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """

        def race(row):
            return 'Caucasian' if row['race'] == "Caucasian" else 'Not Caucasian'

        def two_year_recid(row):
            return 'Did recid.' if row['two_year_recid'] == 1 else 'No recid.'

        df['race'] = df.apply(lambda row: race(row), axis=1)
        df['two_year_recid'] = df.apply(lambda row: two_year_recid(row), axis=1)

        return df[(df.days_b_screening_arrest <= 30)
                  & (df.days_b_screening_arrest >= -30)
                  & (df.is_recid != -1)
                  & (df.c_charge_degree != 'O')
                  & (df.score_text != 'N/A')]


class GermanDataset(Dataset):
    """ https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 """

    def __init__(self):
        meta = json.load(open("data/german/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        df = pd.read_csv(meta["train_path"], sep=" ", names=meta["column_names"].split(","))
        df = self.default_preprocessing(df)
        df["credit"] = df["credit"].astype("str")

        super(GermanDataset, self).__init__(name="German", df=df, **meta, shuffle=False)

    @staticmethod
    def default_preprocessing(df):
        """
        Adds a derived sex attribute based on personal_status.
        https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/german_dataset.py
        """

        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                      'A92': 'female', 'A95': 'female'}
        df['sex'] = df['personal_status'].replace(status_map)

        return df


def fair_stat(data):
    """
    Arguments:
        data: DataTemplate
    """
    s_cnt = Counter(data.s_train)
    s_pos_cnt = defaultdict(int)
    for i in range(data.num_train):
        if data.y_train[i] == 1:
            s_pos_cnt[data.s_train[i]] += 1

    print("-" * 10, "Statistic of fairness")
    for s in s_cnt.keys():
        print("Grp. %d - #instance: %d; #pos.: %d; ratio: %.3f" % (s, s_cnt[s], s_pos_cnt[s], s_pos_cnt[s] / s_cnt[s]))

    print("Overall - #instance: %d; #pos.: %d; ratio: %.3f" % (sum(s_cnt.values()), sum(s_pos_cnt.values()),
                                                               sum(s_pos_cnt.values()) / sum(s_cnt.values())))

    return


def fetch_data(name):
    if name == "adult":
        return AdultDataset().data
    elif name == "compas":
        return CompasDataset().data
    elif name == "german":
        return GermanDataset().data
    else:
        raise ValueError


if __name__ == "__main__":
    data = fetch_data("adult")
    fair_stat(data)