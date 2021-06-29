from copy import deepcopy
from random import seed

import numpy as np
from secml.ml.peval.metrics import CMetricAccuracy

from attacks import custom_log_reg_poisoning_attack
from attacks import log_reg_poisoning_attack
from commons import SEED
from metrics import average_odds_difference
from metrics import disparate_impact
from metrics import error_rates
from scenario import construct_dimp_scenarios
from train import train_logistic
from train import train_SVM


def train_base_model(train_set, test_set, test_sens_attributes):
    """Train base model.

    Trains a base model that will server to evaluate the poisoning strategy on.

    Args:
        train_set (CDataset): the training set.
        test_set (CDataset): the testing set.
        test_sens_attributes (CArray): the sensitive attributes.

    Returns:
        dict: dictionary with the trained base model and its performance metrics.
    """
    # train model
    base_model, base_acc = train_logistic(train_set=train_set, test_set=test_set)

    # make prediction
    y_pred = base_model.predict(test_set.X)

    # calculate performance metrics
    fnr, fpr = error_rates(y_true=test_set.Y.get_data(), y_pred=y_pred.get_data(),
                           sensitive_attributes=test_sens_attributes)
    dimp = disparate_impact(y=y_pred.get_data(), sensitive_attributes=test_sens_attributes)
    odds_diff = average_odds_difference(y_true=test_set.Y.get_data(), y_pred=y_pred.get_data(),
                                        sensitive_attributes=test_sens_attributes)
    # structure output
    output = {
        'base_clf': base_model,
        'base_clf_acc': base_acc,
        'base_clf_dimp': dimp,
        'base_clf_odds': odds_diff,
        'base_clf_FNR': fnr,
        'base_clf_FPR': fpr
    }
    return output


def whitebox_attack(base_clf, train_set, test_set, val_set,
                    test_sensitive_attribute, val_sensitive_attribute):
    """Perform white box attack on logistic regression.

    Args:
        base_clf ([type]): the base model.
        train_set (CDataset): the training set.
        test_set (CDataset): the testing set.
        val_set (CDataset): the validation set.
        test_sensitive_attribute (CArray): the sensitive attributes for the test set.
        val_sensitive_attribute (CArray]): the sensitive attributes for the validation set.

    Returns:
        dict: dictionary with the trained white poisoned model and its performance metrics.
    """

    white_pois_clf = deepcopy(base_clf)

    # perform attack
    white_pois_pts, white_pois_tr = custom_log_reg_poisoning_attack(
        white_pois_clf, train_set, val_set, test_set, test_sensitive_attribute, val_sensitive_attribute)

    # retraining with poisoned points
    white_pois_clf = white_pois_clf.fit(white_pois_tr)
    white_pois_ypred = white_pois_clf.predict(test_set.X)

    # calculate perfomance metrics
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=test_set.Y, y_pred=white_pois_ypred)
    dimp = disparate_impact(y=white_pois_ypred.get_data(), sensitive_attributes=test_sensitive_attribute)
    odds_diff = average_odds_difference(y_true=test_set.Y.get_data(), y_pred=white_pois_ypred.get_data(),
                                        sensitive_attributes=test_sensitive_attribute)
    fnr, fpr = error_rates(y_true=test_set.Y.get_data(), y_pred=white_pois_ypred.get_data(),
                           sensitive_attributes=test_sensitive_attribute)

    # structure output
    output = {
        'white_pois_clf': white_pois_clf,
        'white_pois_pts': white_pois_pts,
        'white_pois_dimp': dimp,
        'white_pois_odds': odds_diff,
        'white_pois_ypred': white_pois_ypred,
        'white_pois_acc': acc,
        'white_pois_FNR': fnr,
        'white_pois_FPR': fpr
    }
    return output


def blackbox_attack(base_clf, train_set, test_set, val_set,
                    test_sensitive_attribute, val_sensitive_attribute):
    """Perform blackbox attack on logistic regression.

    Args:
        base_clf ([type]): the base model.
        train_set (CDataset): the training set.
        test_set (CDataset): the testing set.
        val_set (CDataset): the validation set.
        test_sensitive_attribute (CArray): the sensitive attributes for the test set.
        val_sensitive_attribute (CArray]): the sensitive attributes for the validation set.

    Returns:
        dict: dictionary with the trained blackbox poisoned model and its performance metrics.
    """

    real_model, real_acc = train_SVM(train_set, test_set)

    surrogate_clf = deepcopy(base_clf)

    black_pois_points, black_pois_tr = custom_log_reg_poisoning_attack(surrogate_clf, train_set,
                                                                       val_set, test_set, test_sensitive_attribute,
                                                                       val_sensitive_attribute)

    # retraining with poisoned points
    black_pois_clf = deepcopy(real_model)
    black_pois_clf = black_pois_clf.fit(black_pois_tr)
    black_pois_ypred = black_pois_clf.predict(test_set.X)

    # calculate performance metrics
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=test_set.Y, y_pred=black_pois_ypred)
    dimp = disparate_impact(y=black_pois_ypred.get_data(), sensitive_attributes=test_sensitive_attribute)
    odds_diff = average_odds_difference(y_true=test_set.Y.get_data(), y_pred=black_pois_ypred.get_data(),
                                        sensitive_attributes=test_sensitive_attribute)
    fnr, fpr = error_rates(y_true=test_set.Y.get_data(), y_pred=black_pois_ypred.get_data(),
                           sensitive_attributes=test_sensitive_attribute)

    # structure output
    output = {
        'black_poisoned_clf': black_pois_clf,
        'black_poisoned_pts': black_pois_points,
        'black_pois_dimp': dimp,
        'black_odds': odds_diff,
        'black_pois_ypred': black_pois_ypred,
        'black_pois_acc': acc,
        'black_pois_FNR': fnr,
        'black_pois_FPR': fpr
    }
    return output


def classic_poisoning_attack(base_clf, train_set, test_set, val_set,
                             test_sensitive_attribute, val_sensitive_attribute):
    """Perform standard poisoning attack to logistic regression.

    Args:
        base_clf ([type]): the base model.
        train_set (CDataset): the training set.
        test_set (CDataset): the testing set.
        val_set (CDataset): the validation set.
        test_sensitive_attribute (CArray): the sensitive attributes for the test set.
        val_sensitive_attribute (CArray]): the sensitive attributes for the validation set.

    Returns:
        dict: dictionary with the trained poisoned model and its performance metrics.
    """
    normal_pois_clf = deepcopy(base_clf)

    # perform attack to standard logistic regression.
    normal_pois_points, normal_pois_tr = log_reg_poisoning_attack(normal_pois_clf, train_set, val_set,
                                                                  test_set, test_sensitive_attribute,
                                                                  val_sensitive_attribute)
    # retraining with poisoned points
    normal_pois_clf = normal_pois_clf.fit(normal_pois_tr)
    normal_pois_ypred = normal_pois_clf.predict(test_set.X)

    # calculate performance metrics
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=test_set.Y, y_pred=normal_pois_ypred)
    dimp = disparate_impact(y=normal_pois_ypred.get_data(), sensitive_attributes=test_sensitive_attribute)
    odds_diff = average_odds_difference(y_true=test_set.Y.get_data(), y_pred=normal_pois_ypred.get_data(),
                                        sensitive_attributes=test_sensitive_attribute)
    fnr, fpr = error_rates(y_true=test_set.Y.get_data(), y_pred=normal_pois_ypred.get_data(),
                           sensitive_attributes=test_sensitive_attribute)

    # structure output
    output = {
        'normal_poisoned_classifier': normal_pois_clf,
        'normal_poisoned_points': normal_pois_points,
        'normal_pois_d_imp': dimp,
        'normal_odds': odds_diff,
        'normal_pois_y_pred': normal_pois_ypred,
        'normal_pois_acc': acc,
        'normal_pois_FNR': fnr,
        'normal_pois_FPR': fpr
    }
    return output


if __name__ == '__main__':
    # set seeds for repeated experiments.
    seed(SEED)
    np.random.seed(SEED)

    # construct disparate impact (dimp) scenarios
    dimp_scenarios, scenario_dimps = construct_dimp_scenarios()

    for scenario in dimp_scenarios:
        print(f"\n\n ==== {scenario['name']} ====")
        print(f"\t- {scenario['description']}")

        # train base model.
        base_output = train_base_model(train_set=scenario["training"], test_set=scenario["test"],
                                       test_sens_attributes=scenario["test_sensitive_att"])
        print('--> Whitebox attack...')
        whitebox_output = whitebox_attack(base_clf=base_output['base_clf'], train_set=scenario["training"],
                                          test_set=scenario["test"], val_set=scenario["validation"],
                                          test_sensitive_attribute=scenario["test_sensitive_att"],
                                          val_sensitive_attribute=scenario["validation_sensitive_att"])
