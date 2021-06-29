from secml.data.splitter import CDataSplitterKFold
from secml.ml.classifiers import CClassifierLogistic
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelLinear
from secml.ml.peval.metrics import CMetricAccuracy

from commons import SEED


def train_logistic(train_set, test_set):
    """Train a Logistic Classifier.

    Args:
        train_set (secml.data.c_dataset.CDataset): the training set.
        test_set (secml.data.c_dataset.CDataset): the testing set.

    Returns:
        tuple: classifier model and its accuracy.
    """

    clf = CClassifierLogistic()

    # parameters for the Cross-Validation procedure
    xval_params = {'C': [1, 10]}

    # create a 3-Fold data splitter
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=SEED)

    print("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=train_set,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )
    print("The best training parameters are: ", best_params)

    # We can now fit the classifier
    clf.fit(train_set.X, train_set.Y)
    print("Training of classifier complete!")

    # Compute predictions on a test set
    y_pred = clf.predict(test_set.X)

    # Evaluate the accuracy of the classifier
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=test_set.Y, y_pred=y_pred)

    print("Accuracy on test set: {:.2%}".format(acc))

    return clf, acc


def train_SVM(train_set, test_set):
    """Train Support Vector Machine (SVM)

    Args:
        train_set (secml.data.c_dataset.CDataset): the training set.
        test_set (secml.data.c_dataset.CDataset): the testing set.

    Returns:
        tuple: classifier model and its accuracy.
    """
    clf = CClassifierSVM(kernel=CKernelLinear())  # Linear kernel.

    # Parameters for the Cross-Validation procedure
    xval_params = {'C': [1, 10]}

    # Let's create a 3-Fold data splitter

    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=SEED)

    # Select and set the best training parameters for the classifier
    print("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=train_set,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )

    print("The best training parameters are: ", best_params)

    # We can now fit the classifier
    clf.fit(train_set.X, train_set.Y)
    print("Training of classifier complete!")

    # Compute predictions on a test set
    y_pred = clf.predict(test_set.X)

    # Evaluate the accuracy of the classifier
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=test_set.Y, y_pred=y_pred)

    print("Accuracy on test set: {:.2%}".format(acc))

    return clf, acc
