"""Defines the script to generate artificial datasets for poisoning fairness in ML."""
from enum import IntEnum
import math

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Determines the initial discrimination in the data.
# decrease it to generate more discrimination
DEFAULT_DISC_FACTOR = math.pi / 4

# Determines the distance between distributions
DEFAULT_DISTRIB_DISTANCE = np.array([5, 5])

MU1 = np.array([2, 2])
SIGMA1 = np.array([[5, 1], [1, 5]])
SIGMA2 = np.array([[10, 1], [1, 3]])


class Outcome(IntEnum):
    """Defines the classification outcome."""
    POSITIVE = 1
    NEGATIVE = 0


class Group(IntEnum):
    """Defines the groups involved."""
    UNPRIVILEGED = 0
    PRIVILEGED = 1


def plot_data(X, y, S, title="", ax=None, n_disp=200):
    """Plot the synthetic data set.

    Args:
        X (np.array): the covariates (features).
        y (np.array): the target variable.
        S (np.array): the sensitive attributes.
        title (str, optional): The plot title. Defaults to "".
        ax ([type], optional): The axis to plot. Defaults to None.
        n_disp (int, optional): the number of points to display as to limit cluttering. Defaults to 200.
    """
    xs = X[:n_disp]
    ys = y[:n_disp]
    sens = S[:n_disp]

    unpriv = xs[sens == Group.UNPRIVILEGED]
    priv = xs[sens == Group.PRIVILEGED]
    y_unpriv = ys[sens == Group.UNPRIVILEGED]
    y_priv = ys[sens == Group.PRIVILEGED]

    if ax is not None:
        ax.scatter(unpriv[y_unpriv == Outcome.POSITIVE][:, 0], unpriv[y_unpriv == Outcome.POSITIVE][:, 1],
                   color='green', marker='x', s=30, linewidth=1.5, label="Unprivileged favorable")
        ax.scatter(unpriv[y_unpriv == Outcome.NEGATIVE][:, 0], unpriv[y_unpriv == Outcome.NEGATIVE][:, 1],
                   color='red', marker='x', s=30, linewidth=1.5, label="Unprivileged unfavorable")
        ax.scatter(priv[y_priv == Outcome.POSITIVE][:, 0], priv[y_priv == Outcome.POSITIVE][:, 1],
                   color='green', marker='o', facecolors='none', s=30, label="Privileged favorable")
        ax.scatter(priv[y_priv == Outcome.NEGATIVE][:, 0], priv[y_priv == Outcome.NEGATIVE][:, 1],
                   color='red', marker='o', facecolors='none', s=30, label="Privileged unfavorable")

        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        ax.set_title(title)

        # plt.xlim((-15,10))
        # plt.ylim((-10,15))
        # plt.savefig("img/data.png")
        # plt.show()
    else:
        plt.scatter(unpriv[y_unpriv == Outcome.POSITIVE][:, 0], unpriv[y_unpriv == Outcome.POSITIVE][:, 1],
                    color='green', marker='x', s=30, linewidth=1.5, label="Unprivileged favorable")
        plt.scatter(unpriv[y_unpriv == Outcome.NEGATIVE][:, 0], unpriv[y_unpriv == Outcome.NEGATIVE][:, 1],
                    color='red', marker='x', s=30, linewidth=1.5, label="Unprivileged unfavorable")
        plt.scatter(priv[y_priv == Outcome.POSITIVE][:, 0], priv[y_priv == Outcome.POSITIVE][:, 1],
                    color='green', marker='o', facecolors='none', s=30, label="Privileged favorable")
        plt.scatter(priv[y_priv == Outcome.NEGATIVE][:, 0], priv[y_priv == Outcome.NEGATIVE][:, 1],
                    color='red', marker='o', facecolors='none', s=30, label="Privileged unfavorable")

        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

        plt.legend(fontsize=8)
        # plt.xlim((-15,10))
        # plt.ylim((-10,15))
        # plt.savefig("img/data.png")
        plt.show()


def gen_gaussian(mean, cov, class_label, n_samples):
    """Generate multivariate Gaussian dataset.

    Args:
        mean (np.array): the gaussian mean.
        cov (np.array): covariance.
        class_label (int): the label to associate with the data.
        n_samples (int): number of samples to produce.

    Returns:
        tuple: normal distribution, samples, class_labels
    """
    normal_var = multivariate_normal(mean=mean, cov=cov)
    samples = normal_var.rvs(n_samples)
    class_labels = np.ones(n_samples, dtype=float) * class_label
    return normal_var, samples, class_labels


def generate_synthetic_data(n_samples=400, seed=999, disc_factor=DEFAULT_DISC_FACTOR,
                            mu1=MU1, sigma1=SIGMA1, sigma2=SIGMA2,
                            distrib_distance=DEFAULT_DISTRIB_DISTANCE):
    """Generate 2D synthetic data for the experiment.

    Data will have two non-sensitive features and one sensitive feature.
    A sensitive attibute value of 0 means the point is considered to be
    in the protected group (e.g., female). A value of 1 means it's from
    the non-protected group (e.g., male).

    Args:
        distrib_distance ([type], optional): [description]. Defaults to np.array([5,5]).
        n_samples (int, optional): generate these many data points per class. Defaults to 400.

    Returns:
        [type]: [description]
    """
    np.random.seed(seed)

    # Step 1. Generate the non-sensitive features randomly (one gaussian cluster per class)
    mu2 = np.array(mu1-distrib_distance)  # second clusters is separated from the first by distrib_distance.
    class1, X1, y1 = gen_gaussian(mu1, sigma1, Outcome.POSITIVE, n_samples)
    class2, X2, y2 = gen_gaussian(mu2, sigma2, Outcome.NEGATIVE, n_samples)

    # create dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = np.arange(n_samples*2)
    np.random.shuffle(perm)
    X = X[perm]
    y = y[perm]

    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)],
                              [math.sin(disc_factor), math.cos(disc_factor)]])
    sens_att_position = np.dot(X, rotation_mult)

    # Step 2. Generate the sensitive attribute
    sens_attributes = []  # this array holds the sensitive feature value
    for i in range(len(X)):
        x = sens_att_position[i]

        # probability that a point belongs to each cluster/class
        p1 = class1.pdf(x)
        p2 = class2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s

        r = np.random.uniform()  # sample uniformly

        if r < p1:  # the first cluster is the positive class
            sens_attributes.append(Group.PRIVILEGED)
        else:
            sens_attributes.append(Group.UNPRIVILEGED)

    return X, y, np.array(sens_attributes)


if __name__ == '__main__':
    X, y, S = generate_synthetic_data()
    plot_data(X, y, S)
