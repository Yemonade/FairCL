from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

import tensorflow.compat.v1 as tf

if __name__ == "__main__":
    # get the dataset
    tf.disable_eager_execution()
    dataset_orig = load_preproc_data_adult(protected_attributes=['sex'])
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    print("dataset_orig_train.shape", dataset_orig_train.features.shape)
    # Metric for the original dataset
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

    sess = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                       unprivileged_groups=unprivileged_groups,
                                       scope_name='plain_classifier',
                                       debias=False,
                                       sess=sess)

    plain_model.fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

    # Metrics for the dataset from plain model (without debiasing)
    metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                                unprivileged_groups=unprivileged_groups,
                                                                privileged_groups=privileged_groups)

    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

    metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                               unprivileged_groups=unprivileged_groups,
                                                               privileged_groups=privileged_groups)

    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                              dataset_nodebiasing_test,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                          unprivileged_groups=unprivileged_groups,
                                          scope_name='debiased_classifier',
                                          debias=True,
                                          sess=sess)

    debiased_model.fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

    # Metrics for the dataset from plain model (without debiasing)
    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

    # Metrics for the dataset from model with debiasing
    metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)

    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

    metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups)

    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())

    print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                            dataset_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())
