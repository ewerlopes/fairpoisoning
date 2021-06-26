import numpy as np

from commons import Group
from commons import Outcome


def disparate_impact(y, sensitive_attributes, verbose=True):
    """Calculate the Disparate impact as a difference

    Args:
        y (CArray): binary vector representing predictions.
        sensitive_attributes (CArray): binary vector representing the sensitive attribute.
        verbose (bool, optional): print outputs. Defaults to True.

    Returns:
        float: calculated disparate impact
    """

    privileged = y[sensitive_attributes == Group.PRIVILEGED]
    unprivileged = y[sensitive_attributes == Group.UNPRIVILEGED]

    unprivileged_favorable = unprivileged[unprivileged == Outcome.POSITIVE]
    privileged_favorable = privileged[privileged == Outcome.POSITIVE]

    n1 = len(unprivileged_favorable) / len(unprivileged)
    n2 = len(privileged_favorable) / len(privileged)
    disparate_impact = n1 - n2  # / (max(n2,0.0001))

    if verbose:
        print("\tUnprivileged favorable: ", n1)
        print("\tPrivileged favorable: ", n2)
        print("DIMP: ", disparate_impact)

    return disparate_impact


def error_rate_ratio(y_true, y_pred, sensitive_attributes, verbose=True):
    """Calculate the ratio between the error for the unprivileged group and the privileged one.

    Args:
        y_true (np.ndarray): the ground truth.
        y_pred (np.ndarray): the prediction.
        sensitive_attributes (np.ndarray): the sensitive attributes.
        verbose (bool, optional): whether to print the results. Defaults to True.

    Returns:
        float: the error rate.
    """
    privi_y_true = y_true[sensitive_attributes == Group.PRIVILEGED]
    unprivi_y_true = y_true[sensitive_attributes == Group.UNPRIVILEGED]

    privi_y_pred = y_pred[sensitive_attributes == Group.PRIVILEGED]
    unprivi_y_pred = y_pred[sensitive_attributes == Group.UNPRIVILEGED]

    privi_nerrors = len(privi_y_true) - len(np.where(np.isclose(privi_y_true, privi_y_pred))[0])
    unprivi_nerrors = len(unprivi_y_true) - len(np.where(np.isclose(unprivi_y_true, unprivi_y_pred))[0])

    n1 = unprivi_nerrors / len(unprivi_y_true)
    n2 = privi_nerrors / len(privi_y_true)
    error_rate = n1 / n2

    if verbose:
        print("\tUnprivileged error: ", n1)
        print("\tPrivileged error: ", n2)
        print("Error Rate: ", error_rate)

    return error_rate


def false_positive_rate(y_true, y_pred):
    """Calculate the False Positive Rate (FPR).

    Args:
        y_true (np.ndarray): the ground truth.
        y_pred (np.ndarray): the prediction.

    Returns:
        float: the FPR.
    """
    pred_given_true_is_neg = y_pred[y_true == Outcome.NEGATIVE]
    fp = pred_given_true_is_neg[pred_given_true_is_neg == Outcome.POSITIVE]
    N = len(y_true[y_true == Outcome.NEGATIVE])
    if N == 0:
        return 0
    return len(fp) / N


def true_positive_rate(y_true, y_pred):
    """Calculate the True Positive Rate (TPR).

    Args:
        y_true (np.ndarray): the ground truth.
        y_pred (np.ndarray): the prediction.

    Returns:
        float: the TPR.
    """
    pred_given_true_is_pos = y_pred[y_true == Outcome.POSITIVE]
    tp = pred_given_true_is_pos[pred_given_true_is_pos == Outcome.POSITIVE]
    P = len(y_true[y_true == Outcome.POSITIVE])
    if P == 0:
        return 0
    return len(tp) / P


def false_negative_rate(y_true, y_pred):
    """Calculate the False Negative Rate (FNR).

    Args:
        y_true (np.ndarray): the ground truth.
        y_pred (np.ndarray): the prediction.

    Returns:
        float: the FNR.
    """
    pred_given_true_is_pos = y_pred[y_true == Outcome.POSITIVE]
    fn = pred_given_true_is_pos[pred_given_true_is_pos == Outcome.NEGATIVE]
    P = len(pred_given_true_is_pos)
    if P == 0:
        return 0
    return len(fn) / P


# 12[(𝐹𝑃𝑅𝐷=unprivileged−𝐹𝑃𝑅𝐷=privileged)+(𝑇𝑃𝑅𝐷=unprivileged−𝑇𝑃𝑅𝐷=privileged))]
def average_odds_difference(y_true, y_pred, sensitive_attributes):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    sensitive_attributes = np.array(sensitive_attributes)

    privileged_y_true = y_true[sensitive_attributes == Group.PRIVILEGED]
    unprivileged_y_true = y_true[sensitive_attributes == Group.UNPRIVILEGED]

    privileged_y_pred = y_pred[sensitive_attributes == Group.PRIVILEGED]
    unprivileged_y_pred = y_pred[sensitive_attributes == Group.UNPRIVILEGED]

    FPR_unprivileged = false_positive_rate(unprivileged_y_true, unprivileged_y_pred, Outcome.POSITIVE)
    FPR_privileged = false_positive_rate(privileged_y_true, privileged_y_pred, Outcome.POSITIVE)
    TPR_unprivileged = true_positive_rate(unprivileged_y_true, unprivileged_y_pred, Outcome.POSITIVE)
    TPR_privileged = true_positive_rate(privileged_y_true, privileged_y_pred, Outcome.POSITIVE)

    return 0.5 * ((FPR_unprivileged - FPR_privileged) + (TPR_unprivileged - TPR_privileged))


def error_rates(y_true, y_pred, sensitive_attributes, verbose=False):
    """Calculate error rates False Negative Rate (FNR) and False Positive Rates (FPR).

    Does the calculation for the general error rates (FNR and FPR) and by the rates
    given Privileged and Unprivileged groups.

    Args:
        y_true (np.ndarray): the ground truth.
        y_pred (np.ndarray): the prediction.
        sensitive_attributes (np.ndarray): the sensitive attributes.
        verbose (bool, optional): whether to print the results. Defaults to False.

    Returns:
        (dict, dict): dictionary for the error rates.
    """
    privileged_y_true = y_true[sensitive_attributes == Group.PRIVILEGED]
    unprivileged_y_true = y_true[sensitive_attributes == Group.UNPRIVILEGED]

    privileged_y_pred = y_pred[sensitive_attributes == Group.PRIVILEGED]
    unprivileged_y_pred = y_pred[sensitive_attributes == Group.UNPRIVILEGED]

    FNR_privileged = false_negative_rate(privileged_y_true, privileged_y_pred, Outcome.POSITIVE)
    FNR_unprivileged = false_negative_rate(unprivileged_y_true, unprivileged_y_pred, Outcome.POSITIVE)

    FPR_privileged = false_positive_rate(privileged_y_true, privileged_y_pred, Outcome.POSITIVE)
    FPR_unprivileged = false_positive_rate(unprivileged_y_true, unprivileged_y_pred, Outcome.POSITIVE)

    if verbose:
        print("\tFNR_Privileged: ", FNR_privileged)
        print("\tFNR_Unprivileged: ", FNR_unprivileged)
        print("\tFPR_Privileged: ", FPR_privileged)
        print("\tFPR_Unprivileged: ", FPR_unprivileged)

    try:
        FNR = FNR_unprivileged / FNR_privileged
    except ZeroDivisionError:
        FNR = None

    try:
        FPR = FPR_unprivileged / FPR_privileged
    except ZeroDivisionError:
        FPR = None

    FNR_out = {
        "FNR": FNR,
        "FNR_privileged": FNR_privileged,
        "FNR_unprivileged": FNR_unprivileged,
    }

    FPR_out = {
        "FPR": FPR,
        "FPR_privileged": FPR_privileged,
        "FPR_unprivileged": FPR_unprivileged
    }
    return (FNR_out, FPR_out)
