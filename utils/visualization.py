import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from skimage.filters import gaussian

def browse_images_overlay(image, image2):
    x, y, z = image.shape
    def view_image(i):
        plt.figure(figsize=(8,8))
        plt.imshow(image2[i,:, :], interpolation='nearest', cmap='gray')
        plt.imshow(image[:,:, i], interpolation='nearest', cmap = 'viridis', alpha=0.4)
        plt.show()
    interact(view_image, i=(0,z-1),continuous_update=False)

def browse_images(image):
    z, x, y = image.shape
    def view_image(i):
        plt.figure(figsize=(8,8))
        plt.imshow(image[i,:, :], interpolation='nearest', cmap = 'viridis', vmax=image.max()/2)
        plt.show()
    interact(view_image, i=(0,z-1),continuous_update=False)

def browse_blobs(image, blobs):
    z, x, y = image.shape
    def view_image(i):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image[i], vmax=0.3)
        for blob in blobs:
            z,x,y,s = blob
            if z ==i:
                ax.scatter(y, x, s=s*50, facecolors='none', edgecolors='r')
        plt.show()
    interact(view_image, i=(0,z-1),continuous_update=False)

def plot_mosaic(X, title):
    '''
    title: list of len X
    '''
    Nbr_row_col = int(np.ceil(np.sqrt(len(X))))
    # set up the figure
    fig = plt.figure(figsize=(10, 10))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the faces:
    for i in range(len(X)):
        ax = fig.add_subplot(Nbr_row_col, Nbr_row_col, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i, :, :], cmap = 'viridis')
        ax.text(0, 7, str(title[i]), color="red")

def _plot_learning_curve(estimator, title, X, Y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

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

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_learning_curve(dat, Y, estimator, title = "linear SVC"):
    title = "Learning curve for our "+title
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    _plot_learning_curve(estimator, title, dat, Y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

def heatmap(result, image, window_z_step=4):
    z,x,y = image.shape

    heat_map = np.zeros((z,x,y))
    for i in range(0,len(heat_map),window_z_step):
        step = result[np.where(result[:,2]==i)]
        for res in step:
            heat_map[i, int(res[1])+35,int(res[0])+35] = res[3]
    return gaussian(heat_map, sigma=4)
