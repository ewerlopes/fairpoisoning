from secml.adv.attacks.poisoning.c_attack_poisoning_logistic_regression import CAttackPoisoningLogisticRegression

from commons import SEED
from loss import CLossDisparateImpact


def custom_log_reg_poisoning_attack(surrogate_clf, training_set, validation_set, test_set,
                                    sensible_att_in_test, privileged_condition_validation,
                                    percentage_pois=0.25):
    """Execute adversarial poisoning attack with custom loss.

    Args:
        surrogate_clf ([type]): [description]
        training_set ([type]): [description]
        validation_set ([type]): [description]
        test_set ([type]): [description]
        sensible_att_in_test ([type]): [description]
        privileged_condition_validation ([type]): [description]
        percentage_pois (float, optional): [description]. Defaults to 0.25.

    Returns:
        tuple: poisoning dataset and the training dataset joined with the poisoning dataset.
    """

    NUM_SAMPLES_TRAIN = training_set.num_samples
    n_poisoning_points = int(NUM_SAMPLES_TRAIN * percentage_pois)  # Number of poisoning points to generate
    print(f"Creating {n_poisoning_points} poisoning samples ")

    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 0.05,
        'eta_min': 0.05,
        'eta_max': None,
        'max_iter': 1000,
        'eps': 1e-6
    }

    pois_attack = CAttackPoisoningLogisticRegression(
        classifier=surrogate_clf,
        training_data=training_set,
        surrogate_classifier=surrogate_clf,
        surrogate_data=validation_set,
        val=validation_set,
        distance='l2',
        dmax=40,
        lb=validation_set.X.min(), ub=validation_set.X.max(),
        solver_params=solver_params,
        random_seed=SEED,
        init_type="random"
    )

    pois_attack.n_points = n_poisoning_points
    dimp_loss = CLossDisparateImpact(privileged_condition_validation)
    pois_attack._attacker_loss = dimp_loss

    # Run the poisoning attack
    print("Attack started...")
    pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(test_set.X, test_set.Y)
    print("Attack complete!")

    pois_tr = training_set.deepcopy().append(pois_ds)  # Join the training set with the poisoning points

    return pois_ds, pois_tr


def log_reg_poisoning_attack(surrogate_clf, training_set, validation_set, test_set,
                             sensible_att_in_test, privileged_condition_validation,
                             percentage_pois=0.25):
    """Execute Adversarial Poisoning Attack to Logistic Regression classifier without custom loss.

    Args:
        surrogate_clf ([type]): [description]
        training_set ([type]): [description]
        validation_set ([type]): [description]
        test_set ([type]): [description]
        sensible_att_in_test ([type]): [description]
        privileged_condition_validation ([type]): [description]
        percentage_pois (float, optional): [description]. Defaults to 0.25.

    Returns:
        [type]: [description]
    """

    NUM_SAMPLES_TRAIN = training_set.num_samples
    n_poisoning_points = int(NUM_SAMPLES_TRAIN * percentage_pois)  # Number of poisoning points to generate
    print(f"Creating {n_poisoning_points} poisoning samples ")
    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 0.05,
        'eta_min': 0.05,
        'eta_max': None,
        'max_iter': 1000,
        'eps': 1e-6
    }

    pois_attack = CAttackPoisoningLogisticRegression(
        classifier=surrogate_clf,
        training_data=training_set,
        surrogate_classifier=surrogate_clf,
        surrogate_data=validation_set,
        val=validation_set,
        distance='l2',
        dmax=40,
        lb=validation_set.X.min(), ub=validation_set.X.max(),
        solver_params=solver_params,
        random_seed=SEED,
        init_type="random"
    )

    pois_attack.n_points = n_poisoning_points

    # Run the poisoning attack
    print("Attack started...")
    pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(test_set.X, test_set.Y)
    print("Attack complete!")

    pois_tr = training_set.deepcopy().append(pois_ds)  # Join the training set with the poisoning points

    return pois_ds, pois_tr
