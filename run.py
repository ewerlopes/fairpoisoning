from copy import deepcopy
from random import seed

import numpy as np
from secml.ml.peval.metrics import CMetricAccuracy

from attacks import custom_log_reg_poisoning_attack
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
    base_model, base_acc = train_logistic(training_set=train_set, test_set=test_set)

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

    # retraining with poisoned points
    white_pois_clf = white_pois_clf.fit(white_pois_tr)
    white_pois_ypred = white_pois_clf.predict(test_set.X)

    # calculate perfomance metrics
    metric = CMetricAccuracy()
    white_pois_acc = metric.performance_score(y_true=test_set.Y, y_pred=white_pois_ypred)
    white_pois_dimp = disparate_impact(y=white_pois_ypred.get_data(), sensitive_attributes=test_sensitive_attribute)
    white_odds_diff = average_odds_difference(y_true=test_set.Y.get_data(), y_pred=white_pois_ypred.get_data(),
                                              sensitive_attributes=test_sensitive_attribute)
    white_pois_FNR, white_pois_FPR = error_rates(y_true=test_set.Y.get_data(), y_pred=white_pois_ypred.get_data(),
                                                 sensitive_attributes=test_sensitive_attribute)

    # structure output
    output = {
        'white_pois_clf': white_pois_clf,
        'white_pois_pts': white_pois_pts,
        'white_pois_dimp': white_pois_dimp,
        'white_pois_odds': white_odds_diff,
        'white_pois_ypred': white_pois_ypred,
        'white_pois_acc': white_pois_acc,
        'white_pois_FNR': white_pois_FNR,
        'white_pois_FPR': white_pois_FPR
    }
    return output


def blackbox_attack(original_model):
    """Perform blackbox attack.

    Args:
        original_model ([type]): [description]
    """

    real_model, real_acc = train_SVM(scenario["training"], scenario["test"])

    surrogate_clf = deepcopy(original_model)

    black_pois_points, black_pois_tr = custom_log_reg_poisoning_attack(surrogate_clf, scenario["training"], scenario["validation"], scenario["test"], scenario["test_sensible_att"], scenario["validation_sensible_att"])

    # Retraining with poisoned points
    black_pois_clf = deepcopy(real_model)
    black_pois_clf = black_pois_clf.fit(black_pois_tr)
    black_pois_y_pred = black_pois_clf.predict(scenario["test"].X)

    black_pois_acc = metric.performance_score(y_true=scenario["test"].Y, y_pred=black_pois_y_pred)
    print("->> black")
    black_pois_disparate_imp = calculate_disparate_impact(black_pois_y_pred.get_data(), scenario["test_sensible_att"])
    black_odds_diff = get_average_odds_difference(scenario["test"].Y.get_data(), black_pois_y_pred.get_data(), scenario["test_sensible_att"])
    black_pois_FNR, black_pois_FPR = get_error_rates(scenario["test"].Y.get_data(), black_pois_y_pred.get_data(), scenario["test_sensible_att"], 1, 1)

    scenario['black_poisoned_classifier'] = black_pois_clf
    scenario['black_poisoned_points'] = black_pois_points
    scenario['black_pois_d_imp'] = black_pois_disparate_imp
    scenario['black_odds'] = black_odds_diff
    scenario['black_pois_y_pred'] = black_pois_y_pred
    scenario['black_pois_acc'] = black_pois_acc
    scenario['black_pois_FNR'] = black_pois_FNR
    scenario['black_pois_FPR'] = black_pois_FPR


def classic_poisoning_attack(original_model):
    ################################
    ### CLASSIC POISONING ATTACK ###
    ################################
    normal_pois_clf = deepcopy(original_model)

    privileged_condition_valid = np.ones(scenario['validation'].num_samples)
    privileged_condition_valid[scenario["validation_sensible_att"] == 0] == -1

    normal_pois_points, normal_pois_tr = execute_normal_poisoning_attack(normal_pois_clf, scenario["training"], scenario["validation"], scenario["test"], scenario["test_sensible_att"], scenario["validation_sensible_att"])
    ## Retraining with poisoned points
    normal_pois_clf = normal_pois_clf.fit(normal_pois_tr)
    normal_pois_y_pred = normal_pois_clf.predict(scenario["test"].X)

    metric = CMetricAccuracy()
    normal_pois_acc = metric.performance_score(scenario["test"].Y, y_pred=normal_pois_y_pred)
    print("->> normal")
    normal_pois_disparate_imp = calculate_disparate_impact(normal_pois_y_pred.get_data(), scenario["test_sensible_att"])
    normal_odds_diff = get_average_odds_difference(scenario["test"].Y.get_data(), normal_pois_y_pred.get_data(), scenario["test_sensible_att"])
    normal_pois_FNR, normal_pois_FPR = get_error_rates(scenario["test"].Y.get_data(), normal_pois_y_pred.get_data(), scenario["test_sensible_att"], 1, 1)

    scenario['normal_poisoned_classifier'] = normal_pois_clf
    scenario['normal_poisoned_points'] = normal_pois_points
    scenario['normal_pois_d_imp'] = normal_pois_disparate_imp
    scenario['normal_odds'] = normal_odds_diff
    scenario['normal_pois_y_pred'] = normal_pois_y_pred
    scenario['normal_pois_acc'] = normal_pois_acc
    scenario['normal_pois_FNR'] = normal_pois_FNR
    scenario['normal_pois_FPR'] = normal_pois_FPR


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
                                       test_sens_attributes=scenario["test_sensible_att"])
