from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    f = np.load(filename)
    dc = f - np.mean(f, axis=0)
    return dc


def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1)


def get_eig(S, m):
    d = len(S)
    w, v = eigh(S, subset_by_index=[d-m, d-1]) # get largest m eigen-values/vectors
    eigh_dict = dict()
    for i in range(m):
        eigh_dict[w[i]] = v[:,i] # save columns of eigenvectors
    wsrt = np.sort(w)[::-1] # eigenvalues in decreasing order
    vsrt = v.copy()
    for i in range(m):
        vsrt[:,i] = eigh_dict[wsrt[i]] # rearrangement
    return np.diag(wsrt), vsrt


def get_eig_prop(S, perc):
    w, v = eigh(S)
    wsum = sum(w)
    # w / wsum = perc
    # w = perc * wsum
    # (perc*wsum, wsum]
    w, v = eigh(S, subset_by_value=[perc * wsum, wsum])
    m = len(w)
    eigh_dict = dict()
    for i in range(m):
        eigh_dict[w[i]] = v[:,i]
    wsrt = np.sort(w)[::-1] # eigenvalues in decreasing order
    vsrt = v.copy()
    for i in range(m):
        vsrt[:,i] = eigh_dict[wsrt[i]] # rearrangement
    return np.diag(wsrt), vsrt


def project_image(img, U):
    sum = np.zeros(img.shape[0])    # img.shape: (1024,)
    for i in range(U.shape[1]): # U.shape: (1024, 2)
        alpha = np.dot(U[:,i], img)
        sum += np.dot(alpha, U[:,i])
    return sum


def display_image(orig, proj):
    # reshape the images to be 32 x 32
    org = np.reshape(orig, (32,32), order = 'F')
    prj = np.reshape(proj, (32,32), order = 'F')
    # create a figure with one row of two subplots
    fig, ax = plt.subplots(1, 2)
    # title the subplots
    ax[0].set_title('Original')
    ax[1].set_title('Projection')
    # adjust aspect ratio
    ax0 = ax[0].imshow(org,aspect = 'equal')
    ax1 = ax[1].imshow(prj,aspect = 'equal')
    # create a colorbar for each image
    fig.colorbar(ax0, ax=ax[0])
    fig.colorbar(ax1, ax=ax[1])
    plt.show()

x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 512)
projection = project_image(x[0], U)
display_image(x[0], projection)