from copy import deepcopy

import joblib
from secml.data.c_dataset import CDataset
from secml.figure import CFigure


def plot_2d_decision_boundary(use_case):
    # import warnings
    # warnings.filterwarnings(action='once')

    clf = use_case['base_clf']
    pois_ds = use_case['white_pois_pts']

    training_set = use_case['training']
    validation_set = use_case['validation']
    test_set = use_case['test']

    sensible_att_train = use_case['training_sensitive_att']
    sensible_att_val = use_case['validation_sensitive_att']
    sensible_att_test = use_case['test_sensitive_att']

    # training data
    tr_tmp_X = training_set.X.get_data()
    tr_tmp_Y = training_set.Y.get_data().ravel()

    tr1 = CDataset(tr_tmp_X[sensible_att_train == 0], tr_tmp_Y[sensible_att_train == 0])
    tr2 = CDataset(tr_tmp_X[sensible_att_train == 1], tr_tmp_Y[sensible_att_train == 1])

    # validation data
    val_tmp_X = validation_set.X.get_data()
    val_tmp_Y = validation_set.Y.get_data().ravel()

    val1 = CDataset(val_tmp_X[sensible_att_val == 0], val_tmp_Y[sensible_att_val == 0])
    val2 = CDataset(val_tmp_X[sensible_att_val == 1], val_tmp_Y[sensible_att_val == 1])

    # test data
    ts_tmp_X = test_set.X.get_data()
    ts_tmp_Y = test_set.Y.get_data().ravel()

    ts1 = CDataset(ts_tmp_X[sensible_att_test == 0], ts_tmp_Y[sensible_att_test == 0])
    ts2 = CDataset(ts_tmp_X[sensible_att_test == 1], ts_tmp_Y[sensible_att_test == 1])

    pois_clf = use_case['white_pois_clf']

    pois_points_X = pois_ds.X.get_data()
    pois_points_Y = pois_ds.Y.get_data().ravel()

    pois_ds1 = CDataset(pois_points_X[pois_points_Y == 1], pois_points_Y[pois_points_Y == 1])
    pois_ds2 = CDataset(pois_points_X[pois_points_Y == 0], pois_points_Y[pois_points_Y == 0])

    pois_tr = deepcopy(training_set).append(pois_ds)

    # Define common bounds for the subplots
    min_limit = min(pois_tr.X.min(), training_set.X.min())
    max_limit = max(pois_tr.X.max(), training_set.X.max())
    grid_limits = [[min_limit, max_limit], [min_limit, max_limit]]

    fig = CFigure(15, 10)

    fig.subplot(3, 2, 1)
    fig.sp.title("Original classifier (training set)")
    fig.sp.plot_decision_regions(clf, n_grid_points=200, grid_limits=grid_limits, cmap="RdYlGn")

    fig.sp.plot_ds(tr1, markersize=7, colors=['red', 'green'], markers='X')
    fig.sp.plot_ds(tr2, markersize=5, colors=['green', 'red'], markers='o')
    # , markerfacecolor="None", markeredgecolor='g', markeredgewidth=1.5)
    fig.sp.grid(grid_on=True)

    fig.subplot(3, 2, 2)
    fig.sp.title("Poisoned classifier (training set + poisoning points)")
    fig.sp.plot_decision_regions(pois_clf, n_grid_points=200, grid_limits=grid_limits, cmap="RdYlGn")
    fig.sp.plot_ds(tr1, markersize=7, colors=['red', 'green'], markers='X')
    fig.sp.plot_ds(tr2, markersize=5, colors=['green', 'red'], markers='o')
    fig.sp.plot_ds(pois_ds1, markers='*', markersize=12, colors='darkred')
    fig.sp.plot_ds(pois_ds2, markers='*', markersize=12, colors='darkgreen')
    fig.sp.grid(grid_on=True)

    fig.subplot(3, 2, 3)
    fig.sp.title("Original classifier (test set)")
    fig.sp.plot_decision_regions(clf, n_grid_points=200, grid_limits=grid_limits, cmap="RdYlGn")

    fig.sp.plot_ds(ts1, markersize=7, colors=['red', 'green'], markers='X')
    fig.sp.plot_ds(ts2, markersize=5, colors=['red', 'green'], markers='o')
    # fig.sp.text(0.05, -0.25, "Accuracy on test set: {:.2%}".format(acc),bbox=dict(facecolor='white'))
    fig.sp.grid(grid_on=True)

    fig.subplot(3, 2, 4)
    fig.sp.title("Poisoned classifier (test set)")
    fig.sp.plot_decision_regions(pois_clf, n_grid_points=200, grid_limits=grid_limits, cmap="RdYlGn")

    fig.sp.plot_ds(ts1, markersize=7, colors=['red', 'green'], markers='X')
    fig.sp.plot_ds(ts2, markersize=5, colors=['red', 'green'], markers='o')
    # fig.sp.text(0.05, -0.25, "Accuracy on test set: {:.2%}".format(pois_acc),bbox=dict(facecolor='white'))

    fig.subplot(3, 2, 5)
    fig.sp.title("Original classifier (validation set)")
    fig.sp.plot_decision_regions(clf, n_grid_points=200, grid_limits=grid_limits, cmap="RdYlGn")
    fig.sp.plot_ds(val1, markersize=7, colors=['red', 'green'], markers='X')
    fig.sp.plot_ds(val2, markersize=5, colors=['green', 'red'], markers='o')
    fig.sp.grid(grid_on=True)

    fig.subplot(3, 2, 6)
    fig.sp.title("Poisoned classifier (validation set + poisoning points)")
    fig.sp.plot_decision_regions(pois_clf, n_grid_points=200, grid_limits=grid_limits, cmap="RdYlGn")
    fig.sp.plot_ds(val1, markersize=7, colors=['red', 'green'], markers='X')
    fig.sp.plot_ds(val2, markersize=5, colors=['green', 'red'], markers='o')
    fig.sp.plot_ds(pois_ds1, markers='*', markersize=12, colors='darkred')
    fig.sp.plot_ds(pois_ds2, markers='*', markersize=12, colors='darkgreen')
    fig.sp.grid(grid_on=True)

    fig.savefig('fig.png')


if __name__ == '__main__':
    dimp_scenarios = joblib.load('dimp_scenarios.pkl')
    plot_2d_decision_boundary(dimp_scenarios[0])
