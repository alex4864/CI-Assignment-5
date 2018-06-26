#Filename: HW5_skeleton.py
#Author: Christian Knoll
#Edited: May, 2018

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import multivariate_normal
import pdb

import sklearn
from sklearn import datasets
#--------------------------------------------------------------------------------
# Assignment 5
def main():
    #------------------------
    # 0) Get the input
    ## (a) load the modified iris data
    data, labels = load_iris_data()

    ## (b) construct the datasets
    x_2dim = data[:, [0,2]]
    x_4dim = data
    #TODO: implement PCA
    x_2dim_pca = PCA(data,nr_dimensions=2,whitening=False)

    ## (c) visually inspect the data with the provided function (see example below)
    plot_iris_data(x_2dim,labels)

    #------------------------
    # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm
    scenario = 1
    dim = 2
    nr_components = 3

    #TODO set parameters
    tol = .01  # tolerance
    max_iter = 200  # maximum iterations for GN
    nr_components = 3 #n number of components

    #TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(x_2dim,dimension = dim, nr_components= nr_components, scenario=scenario)
    (alpha_0, mean_0, cov_0, arr_log_likelihood, class_labels) = EM(x_2dim,nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    initial_centers = init_k_means(x_2dim, dimension = dim, nr_clusters=nr_components, scenario=scenario)
    centers, cumulative_distance, labels = k_means(x_2dim, nr_components, initial_centers, max_iter, tol)

    #TODO visualize your results
    draw_EM(x_2dim,mean_0, cov_0, arr_log_likelihood, class_labels)
    draw_kmeans(x_2dim, centers, labels, cumulative_distance)

    #------------------------
    # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm
    scenario = 2
    dim = 4
    nr_components = 3

    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    #nr_components = ... #n number of components

    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    #initial_centers = init_k_means(dimension = dim, nr_cluster=nr_components, scenario=scenario)
    #... = k_means(x_2dim,nr_components, initial_centers, max_iter, tol)

    #TODO: visualize your results by looking at the same slice as in 1)


    #------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data
    scenario = 3
    dim = 2
    nr_components = 3

    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    #nr_components = ... #n number of components

    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_2dim_pca, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    #initial_centers = init_k_means(dimension = dim, nr_cluster=nr_components, scenario=scenario)
    #... = k_means(x_2dim_pca, nr_components, initial_centers, max_iter, tol)

    #TODO: visualize your results
    #TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)

    #pdb.set_trace()

    alpha = np.array([[0.1, 0.3, 0.6]])
    mu = np.array([[4, 4], [-2, 2], [0, 0]])
    cov = np.array([
        [[.2, 0],
         [0, .2]],
        [[.3, 0],
         [0, .3]],
        [[.2, .1],
         [.1, .2]]
    ])

    Y = sample_GMM(alpha, mu, cov, 100)

    plt.scatter(Y[:,0], Y[:,1])
    plt.show()

def sample_GMM(alpha, mu, cov, N):
    #assert sum(alpha) == 1
    num_gaussians = alpha.shape[1]

    alpha_cumsum = [0]
    for i in np.nditer(alpha):
        alpha_cumsum.append(alpha_cumsum[-1] + i)

    # array to store the number of samples assigned to each distribution
    num_samples = [0] * num_gaussians
    for i in range(N):
        rand = np.random.rand()
        for j in range(num_gaussians):
            if alpha_cumsum[j] <= rand and alpha_cumsum[j + 1] > rand:
                num_samples[j] += 1
                break

    samples = []
    for i in range(num_gaussians):
        samples.append(np.random.multivariate_normal(mu[i], cov[i], num_samples[i]))

    return np.vstack(samples)

def draw_EM(points,mean_0, cov_0, arr_log_likelihood, labels):

    labes = reassign_class_labels(labels)
    plot_iris_data(points,labels)

    x = arr_log_likelihood.size
    plt.title("log likelihood function")
    plt.xlabel("iterations")
    plt.ylabel("log-likelihood")
    plt.scatter(np.arange(x), arr_log_likelihood)
    plt.show()

    xmin = 4
    xmax = 8
    ymin = 0
    ymax = 8
    nr_points = 50


    for i in range(points.shape[0]):
        for j in range(mean_0.shape[1]):
            if labels[i] == j:
                plt.scatter(points[i, 0], points[i, 1], c='C{}'.format(j))
                break
        c = 0

    for k in range(0, mean_0.shape[1]):
        mu = mean_0[:,k]
        cov = cov_0[:,:,k]
        delta_x = float(xmax-xmin) / float(nr_points)
        delta_y = float(ymax-ymin) / float(nr_points)
        x = np.arange(xmin, xmax, delta_x)
        y = np.arange(ymin, ymax, delta_y)
        X, Y = np.meshgrid(x, y)
        Z = mlab.bivariate_normal(X,Y,np.sqrt(cov[0][0]),np.sqrt(cov[1][1]),mu[0], mu[1], cov[0][1])
        plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
        CS = plt.contour(X, Y, Z)
        plt.clabel(CS, inline=1, fontsize=10)

    plt.show()

def draw_kmeans(points, centers, labels, cumulative_distance):

    for i in range(points.shape[0]):
        for j in range(centers.shape[1]):
            if labels[i] == j:
                plt.scatter(points[i, 0], points[i, 1], c='C{}'.format(j))
                break
        c = 0

    for i in range(centers.shape[1]):
        plt.scatter(centers[0, i], centers[1, i], c='C{}'.format(i), marker='X', linewidths=1, edgecolors=(0,0,0))

    plt.show()

    plt.plot(cumulative_distance)
    plt.show()

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def init_EM(x_dim, dimension=2,nr_components=3, scenario=None):
    """ initializes the EM algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""

    alpha_0 = np.full((1, nr_components), 1/nr_components);

    mean_0 = np.zeros([dimension, nr_components])
    (r,_) = x_dim.shape

    for i in range(0, nr_components):
        idx = np.random.randint(r,size = nr_components)
        sample_m = x_dim[idx,:]
        mean_0[:,i] = np.mean(sample_m, axis=0)

    cov_1 = np.cov(x_dim[:,0],x_dim[:,1])
    cov_0 = np.zeros([dimension, dimension, nr_components])
    for i in range(0, nr_components):
        cov_0[:,:,i] = cov_1

    return (alpha_0, mean_0, cov_0)
#--------------------------------------------------------------------------------
def EM(X,K,alpha_0,mean_0,cov_0, max_iter, tol):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension
    nr_samples = X.shape[0]
    D = X.shape[1]
    assert D == mean_0.shape[0]
    arr_log_likelihood = np.zeros([max_iter,1])

    prob_a = np.zeros([nr_samples, K])

    for it in range(0,max_iter):

        #E-step compute Rmk
        x_prob = np.zeros([nr_samples, K])
        for k in range(0, K):
            x_prob[:,k] = multivariate_normal.pdf(X, mean_0[:,k], cov_0[:,:,k])
        prob_a = np.multiply(alpha_0, x_prob)

        sum_prob = prob_a.sum(axis=1, keepdims=True)
        arr_log_likelihood[it] = np.sum(np.log10(sum_prob))
        if it != 0 and abs(arr_log_likelihood[it] - arr_log_likelihood[it-1]) < tol :
            arr_log_likelihood = arr_log_likelihood[0:it]
            break;

        rmk = prob_a/sum_prob

        #M-step
        Nk = rmk.sum(axis=0)
        for k in range(0,K):
            mul = X * np.reshape(rmk[:,k],[nr_samples,1])
            mean_0[:,k] = np.sum(mul,axis = 0)/Nk[k]

        for k in range(0 ,K):
            xm = np.transpose(X)-np.reshape(mean_0[:,k],[D,1])
            new_cov = np.zeros([D,D])
            for m in range(0,nr_samples):
                xmm = np.reshape(xm[:,m],[D,1])
                mul = np.matmul(xmm,np.transpose(xmm))
                new_cov = new_cov + rmk[m,k]*(mul)
            new_cov = new_cov/Nk[k]
            cov_0[:,:,k] = new_cov[:,:]
        N = np.sum(Nk)
        alpha_0 = Nk/N

    class_label = np.argmax(prob_a, axis=1)
    return(alpha_0, mean_0, cov_0, arr_log_likelihood, class_label)
    #TODO: iteratively compute the posterior and update the parameters

    #TODO: classify all samples after convergence
    pass
#--------------------------------------------------------------------------------
def init_k_means(X, dimension=None, nr_clusters=None, scenario=None):
    """ initializes the k_means algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    # TODO chosse suitable inital values for each scenario

    potential_centers = np.copy(X)
    centers = []
    for i in range(nr_clusters):
        selection = int(np.floor(np.random.rand() * len(potential_centers)))
        centers.append( potential_centers[selection].reshape([2, 1]) )
        potential_centers = np.delete(potential_centers, selection, 0)

    return np.hstack(centers)

#--------------------------------------------------------------------------------
def k_means(X, K, centers_0, max_iter, tol):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    assert D == centers_0.shape[0]

    centers = centers_0
    cumulative_distance = np.zeros([max_iter, 1])

    for x in range(max_iter):
        nearestCenters = []
        for i in range(X.shape[0]):
            nearestCenters.append(getNearestCluster(X[i], centers))

        cum_dist = 0
        for i in range(centers.shape[1]):
            try:
                assigned_points = np.vstack( [row for index, row in enumerate(X) if nearestCenters[index] == i] )
                centers[:, i] = np.mean(assigned_points, axis=0)
                cum_dist += sum([np.linalg.norm(point - centers[:, i]) for point in assigned_points])
            except ValueError:
                print('No points assigned to cluster {}'.format(i))

        cumulative_distance[x, 0] = cum_dist


    labels = np.array( [getNearestCluster(X[i], centers) for i in range(X.shape[0])] )
    return centers, cumulative_distance, labels

def getNearestCluster(point, centers):
    distances = [np.linalg.norm(point - centers[:, i]) for i in range(centers.shape[1])]
    return distances.index(min(distances))

#--------------------------------------------------------------------------------
def PCA(data,nr_dimensions=None, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening

    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    # Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components
    means = np.mean(data, axis=0)
    da = data.copy()
    flag = True
    i = 0
    for sample in np.nditer(da, op_flags=['readwrite']):
        sample[...] = sample - means[i%4]
        i += 1

    cov = np.cov(da.T)
    originalVar = cov[0][0] + cov[1][1] + cov[2][2] + cov[3][3]
    evalues, evectors = np.linalg.eig(cov)

    # sort eigenvalues and vectors by size of eigenvalue
    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]
    evectors = evectors[:,idx]

    principalComponents = evectors[:, 0:dim]
    if whitening:
        principalComponents = np.matmul(np.sqrt(np.diag(evalues[:dim])), principalComponents.T).T

    dt = np.matmul(da, principalComponents)


    # Have a look at the associated eigenvalues and compute the amount of varianced explained
    print('===================')
    print('original variance', originalVar)
    print('sum of eigenvalues', np.sum(evalues))
    print('associated eigenvalues', evalues[:dim])
    print('explained', np.sum(evalues[:dim])/np.sum(evalues))
    print('neglected eigenvalues', evalues[dim:])

    return dt
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input:
    Returns:
        X... samples, 150x4
        Y... labels, 150x1"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100,2] =  iris.data[50:100,2]-0.25
    Y = iris.target
    return X,Y
#--------------------------------------------------------------------------------
def plot_iris_data(data,labels):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1"""
    plt.scatter(data[labels==0,0], data[labels==0,1], label='Iris-Setosa')
    plt.scatter(data[labels==1,0], data[labels==1,1], label='Iris-Versicolor')
    plt.scatter(data[labels==2,0], data[labels==2,1], label='Iris-Virgnica')

    plt.legend()
    plt.show()
#--------------------------------------------------------------------------------
def likelihood_multivariate_normal(X, mean, cov, log=False):
   """Returns the likelihood of X for multivariate (d-dimensional) Gaussian
   specified with mu and cov.

   X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
   mean ... mean -- [mu_1, mu_2,...,mu_d]
   cov ... covariance matrix -- np.array with (d x d)
   log ... False for likelihood, true for log-likelihood
   """

   dist = multivariate_normal(mean, cov)
   if log is False:
       P = dist.pdf(X)
   elif log is True:
       P = dist.logpdf(X)
   return P

#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,nr_points,title="Title"):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters

    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""

	#npts = 100
    delta_x = float(xmax-xmin) / float(nr_points)
    delta_y = float(ymax-ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)

    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X,Y,np.sqrt(cov[0][0]),np.sqrt(cov[1][1]),mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return
#--------------------------------------------------------------------------------
def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.

    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)

    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return np.random.permutation(y) # permutation of all samples
#--------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable.
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50]==0)   ,  np.sum(labels[0:50]==1)   , np.sum(labels[0:50]==2)   ],
                                  [np.sum(labels[50:100]==0) ,  np.sum(labels[50:100]==1) , np.sum(labels[50:100]==2) ],
                                  [np.sum(labels[100:150]==0),  np.sum(labels[100:150]==1), np.sum(labels[100:150]==2)]])
    new_labels = np.array([np.argmax(class_assignments[:,0]),
                           np.argmax(class_assignments[:,1]),
                           np.argmax(class_assignments[:,2])])
    return new_labels
#--------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)

    plot_gauss_contour(mu, cov, -2, 2, -2, 2,100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)

    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))

    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels =np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd==0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd==1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd==2] = new_labels[2]



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    sanity_checks()
    main()
