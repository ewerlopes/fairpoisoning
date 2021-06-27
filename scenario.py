import numpy as np
from secml.data.c_dataset import CDataset
from sklearn.model_selection import train_test_split

from commons import SEED
from generate import generate_synthetic_data
from metrics import disparate_impact

N = 9  # max euclidean distance between average of distributions

dimp_in_data = []
euc_distances = []
dimp_scenarios = []

RANGE = np.arange(0, 10, 1)

for n in RANGE:

    # Generating data
    euc_dist = n
    i = np.sqrt((euc_dist**2)/2)
    X, y, X_control = generate_synthetic_data(False, np.array([i, i]))
    formatted_X = np.array([X[:, 0], X[:, 1], X_control['s1']]).T  # concatenating X with sensible att

    sec_ml_dataset_all = CDataset(X, y)
    sensible_att_all = X_control['s1']

    euc_distances.append(n)
    dimp_in_data.append(disparate_impact(sec_ml_dataset_all.Y.get_data(), sensible_att_all))

    # Splitting data.
    X_train_val, X_test, y_train_val, y_test = train_test_split(formatted_X, y, test_size=0.2,
                                                                random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5,
                                                      random_state=SEED)

    train_set = CDataset(X_train[:, :2], y_train)
    train_sensitive_att = X_train[:, 2]

    val_set = CDataset(X_val[:, :2], y_val)
    val_sensitive_att = X_val[:, 2]
    val_lambda = np.zeros(val_set.num_samples)

    # Creating lambda vector
    val_lambda[np.where((val_sensitive_att == 0) & (y_val == 0))[0]] == 1  # Unprivileged denied
    val_lambda[np.where((val_sensitive_att == 0) & (y_val == 1))[0]] == 1  # Unprivileged granted
    val_lambda[np.where((val_sensitive_att == 1) & (y_val == 0))[0]] == -1  # Privileged denied
    val_lambda[np.where((val_sensitive_att == 1) & (y_val == 1))[0]] == -1  # Privileged granted

    test_set = CDataset(X_test[:, :2], y_test)
    test_sensitive_att = X_test[:, 2]

    ########################################
    # GENERATING DATA FOR WHITE BOX ATTACK #
    ########################################

    X2, y2, X_control2 = generate_synthetic_data(False, np.array([i, i]))
    formatted_X2 = np.array([X2[:, 0], X2[:, 1], X_control2['s1']]).T  # concatenating X with sensible att

    sec_ml_dataset_all2 = CDataset(X2, y2)
    sensible_att_all2 = X_control2['s1']

    # Splitting data.
    X_train_val2, X_test2, y_train_val2, y_test2 = train_test_split(formatted_X2, y2, test_size=0.2,
                                                                    random_state=SEED)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train_val2, y_train_val2, test_size=0.5,
                                                          random_state=SEED)

    training2 = CDataset(X_train2[:, :2], y_train2)
    training_sensible_att2 = X_train2[:, 2]

    validation2 = CDataset(X_val2[:, :2], y_val2)
    validation_sensible_att2 = X_val2[:, 2]
    val_lambda2 = np.zeros(validation2.num_samples)

    # Creating lambda vector
    val_lambda2[np.where((validation_sensible_att2 == 0) & (y_val2 == 0))[0]] == 1  # Unprivileged denied
    val_lambda2[np.where((validation_sensible_att2 == 0) & (y_val2 == 1))[0]] == 1  # Unprivileged granted
    val_lambda2[np.where((validation_sensible_att2 == 1) & (y_val2 == 0))[0]] == -1  # Privileged denied
    val_lambda2[np.where((validation_sensible_att2 == 1) & (y_val2 == 1))[0]] == -1  # Privileged granted

    test2 = CDataset(X_test2[:, :2], y_test)
    test_sensible_att2 = X_test2[:, 2]

    scenario = {
        "name": "Use case 4 - {}".format(n),
        "description": "Disparate impact attack. \n Euclidean distance between group averages: {}\n".format(n),
        "training": train_set,
        "training_sensible_att": train_sensitive_att,
        "validation": val_set,
        "validation_sensible_att": val_sensitive_att,
        "lambda_validation": val_lambda,
        "test": test_set,
        "test_sensible_att": test_sensitive_att,
        "all_data": sec_ml_dataset_all,
        "all_sensible_att": sensible_att_all,
        "black_box_training": training2,
        "black_box_training_sensible_att": training_sensible_att2,
        "black_box_validation": validation2,
        "black_box_validation_sensible_att": validation_sensible_att2,
        "black_box_lambda_validation": val_lambda2,
        "black_box_test": test2,
        "black_box_test_sensible_att": test_sensible_att2,
        "black_box_all_data": sec_ml_dataset_all2,
        "black_box_all_sensible_att": sensible_att_all2,
    }

    dimp_scenarios.append(scenario)
