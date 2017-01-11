from timeit import default_timer
from itertools import chain
import random
import time

import matplotlib.pyplot as plt
import numpy as np


def sample_images(data, img_size, nrow=3, ncol=3):
    """Utility function used to visually verify that dataset wasn't malformed
    due to various transformations and normalization techniques.
    """
    rows, columns = data.shape
    fig, axes = plt.subplots(nrow, ncol)
    indicies = []
    for ax in chain(*axes):
        index = random.randint(0, rows)
        img = data[index, :].reshape(48, 48)
        ax.imshow(img, cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        indicies.append(index)
    return fig, indicies


def padsize(index):
    return len(str(abs(index)))


def save_model(model, folder, name):
    from sklearn.externals import joblib
    filename = os.path.join(folder, name)
    joblib.dump(model, filename)


def load_model(folder, name):
    from sklearn.externals import joblib
    filename = os.path.join(folder, name)
    return joblib.load(filename)
    

class Timer:
    """Simple util to measure execution time.

    Examples
    --------
    >>> import time
    >>> with Timer() as timer:
    ...     time.sleep(1)
    >>> print(timer)
    00:00:01
    """
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = default_timer() - self.start

    def __str__(self):
        return self.verbose()

    def verbose(self):
        if self.elapsed is None:
            return '<not-measured>'
        return time.strftime('%H:%M:%S', time.gmtime(self.elapsed))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve.

    Notes
    -----
    Method was taken from [1] and slightly modified.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    References
    ----------

    [1] http://scikit-learn.org/stable/auto_examples/model_selection/
        plot_learning_curve.html

    """
    from sklearn.model_selection import learning_curve

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.legend(loc="best")

    return fig


def plot_roc_curve(y_true, y_pred, pos_label=None, verbose_label='',
                   show_diagonal=True, color='darkorange', ax=None):
    """
    >>> plot_roc_curve(1, 2)
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    area = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    if pos_label:
        verbose = verbose_label or str(pos_label)
        legend_label = '%s (area=%s)' % (verbose, area)
    else:
        legend_label = 'ROC curve (area=%s)' % area
    ax.plot(fpr, tpr, color=color, lw=2, label=legend_label)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if show_diagonal:
        ax.plot([0, 1], [0, 1], color='darkgray', lw=2, linestyle='--')
    ax.legend(loc='lower right')
    return ax.get_figure()
