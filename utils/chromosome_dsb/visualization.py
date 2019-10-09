import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from skimage.filters import gaussian
import matplotlib.patches as patches

def convert_view(img):
    # Feature scaling (min-max normalization):
    to_visualize = ((img- img.min())/(img.max() - img.min()))*255
    return to_visualize.astype(np.uint16)


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

    plt.savefig('/Users/Espenel/Desktop/learning_curve.pdf', bbox_inches="tight", pad_inches=0,transparent=True)
    return plt

def plot_learning_curve(dat, Y, estimator, title = "linear SVC"):
    title = "Learning curve for our "+title
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    _plot_learning_curve(estimator, title, dat, Y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

def heatmap(result):

    heat_map = np.pad(result, ((0,0), (35,35),(35,35)), 'constant')

    return gaussian(heat_map, sigma=4)

def plot_background(image, ch1, ch2):
    z, x, y, _ = image.shape
    @widgets.interact(
        channel=[1, 2], pos=(0, y-1), slices = (0,z-1))

    def view_image(pos = 100, channel = 1, slices = 20):
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes[0,0].set_title("raw")
        axes[0,0].imshow(image[slices,:,:,channel])
        axes[0,0].axhline(y=pos, color='r', linestyle='-')
        axes[0,1].set_title("after background sub")
        if channel == 1:
            axes[0,1].imshow(ch1[slices])
            axes[0,1].axhline(y=pos, color='r', linestyle='-')
            axes[1,0].plot(image[slices,pos,:,channel], color='r', linestyle='-');
            axes[1,1].plot(ch1[slices,pos,:], color='r', linestyle='-');
        if channel == 2:
            axes[0,1].imshow(ch2[slices])
            axes[0,1].axhline(y=pos, color='r', linestyle='-')
            axes[1,0].plot(image[slices,pos,:,channel], color='r', linestyle='-');
            axes[1,1].plot(ch2[slices,pos,:], color='r', linestyle='-');


def plot_result(img, results, bbox_ML,cts, num, meta, directory, save = False, plot = True):
    if plot == True:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.amax(img,axis=0), vmax=img.max())
        for blob in results:
            z,x,y,s = blob
            loci = ax.scatter(y, x, s=20, facecolors='none', edgecolors='y')
        for coord, val, cell in zip(bbox_ML,cts, num):
            if val == 0:
                circles1 = patches.Circle((coord[0]+30,coord[1]+30),32, linewidth=3,edgecolor='r',facecolor='none', alpha = 0.2)
            elif val > 0:
                circles1 = patches.Circle((coord[0]+30,coord[1]+30),32, linewidth=3,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(circles1)
            ax.text(coord[0]+15,coord[1], "Cell_{}".format(str(cell)),color = 'r', weight='bold')
            ax.text(coord[0]+15,coord[1]+35, "{} COSA-1".format(str(val)),color = 'w', weight='bold')
        plt.legend([circles1, loci], ["Nucleus found with ML",  "FOCI"],loc=0,fontsize='small')
        if save:
            try:
                filename = meta['Name']+'.pdf'
                plt.savefig(directory+'/'+filename, transparent=True)
            except FileNotFoundError:
                plt.savefig(filename, transparent=True)
    elif plot ==False:
        plt.ioff()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.amax(img,axis=0), vmax=img.max())
        for blob in results:
            z,x,y,s = blob
            loci = ax.scatter(y, x, s=10, facecolors='none', edgecolors='y')
        for coord, val, cell in zip(bbox_ML,cts, num):
            if val == 0:
                circles1 = patches.Circle((coord[0]+30,coord[1]+30),30, linewidth=3,edgecolor='r',facecolor='none', alpha = 0.2)
            elif val > 0:
                circles1 = patches.Circle((coord[0]+30,coord[1]+30),30, linewidth=3,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(circles1)
            ax.text(coord[0]+15,coord[1], "Cell_{}".format(str(cell)),color = 'r', weight='bold')
            ax.text(coord[0]+15,coord[1]+35, "{} COSA-1".format(str(val)),color = 'w', weight='bold')
        plt.legend([circles1, loci], ["Nucleus found with ML",  "FOCI"],loc=0,fontsize='small')
        if save:
            try:
                filename = meta['Name']+'.pdf'
                plt.savefig(directory+'/'+filename, transparent=True)
            except FileNotFoundError:
                plt.savefig(filename, transparent=True)
        plt.close(fig)


'''
def heatmap(result, image, window_z_step=4):
    z,x,y = image.shape

    heat_map = np.zeros((z,x,y))
    for i in range(0,len(heat_map),window_z_step):
        step = result[np.where(result[:,2]==i)]
        for res in step:
            heat_map[i, int(res[1])+35,int(res[0])+35] = res[3]
    return gaussian(heat_map, sigma=4)
'''
