# %%
# import libraries
import os

import numpy as np
import PIL

import tensorflow as tf
from numpy import asarray, clip
from PIL import Image
from scipy import stats
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.impute import KNNImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import Sequential

from xgboost import XGBRegressor


# %%
# define functions
def processImage(file, grayscale = True, basewidth = 280):
    """Imports image, converts to grayscale if specified, adjusts height

    Args:
        file (str): image file name
        grayscale (bool, optional): Convert to grayscale. Defaults to True.
        basewidth (int, optional): base pixel width. Defaults to 280.

    Returns:
        PIL Image: A processed PIL Image of the original image
    """
    
    # open file and convert to grayscale if specified
    if grayscale == True:
        img = Image.open(file).convert('L')
    if grayscale == False:
        img = Image.open(file)
        
    # resize image
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    
    return img

def getIndexFeatures(img, poly_degree, include_y = True):
    """Obtains the positional indices/location of pixels as features.
       For example, the upper most pixel has feature values [0, 0].
       Includes polynomial expansions and interactions between dimensions
       if specified; includes pixel values as output vector if specified.
       Note: feature matrix is populated in row-major order (i.e., the first
       three records are [0,0], [0, 1], [0, 2]).

    Args:
        img (PIL Image): PIL Image as obtained using processImage()
        poly_degree (int): Polynomial degree with which to expand features
        include_y (bool, optional): Vector of pixel values. Defaults to True.
        

    Returns:
        Numpy array(s): Feature matrix and outcome vector, if specified
    """

    # convert image to array
    img_t = asarray(img)

    # create empty feature matrix
    X = np.empty((img_t.shape[0]*img_t.shape[1], 2))

    # create empty outcome vector, if specified
    if include_y == True:
        y = np.empty(img_t.shape[0]*img_t.shape[1])

    # loop through image in row-major order to populate feature matrix 
    # and outcome vector, if specified
    i = 0
    for r in range(img_t.shape[0]):
        for c in range(img_t.shape[1]):
            X[i, 0] = r
            X[i, 1] = c
            if include_y == True:
                y[i] = img_t[r, c]
            i = i + 1
    poly = PolynomialFeatures(poly_degree)
    X = poly.fit_transform(X)
    X = np.delete(X, 0, axis = 1)
    X = stats.zscore(X, axis = 0)

    if include_y == True:
        return X, y
    if include_y == False:
        return X

def convertIndexToImage(og_img_shape, pred_y):
    """Converts vector of pixel values (organized in row-major order, 
       as with getIndexFeatures()) back into an image. This is used to 
       reconstruct predicted pixel values into an image. This function
       is typically used internally in other functions.

    Args:
        og_img_shape (): Original shape of the target image
        pred_y (numpy array): vector of predicited pixel values, which
        must be organized in row-major order (i.e., the first
        three records are [0,0], [0, 1], [0, 2])

    Returns:
        PIL Image: PIL Image file
    """

    # initialize empty array for resulting image
    pred_img = np.empty(og_img_shape)
    
    # populate empty result array with predicted pixel values
    # in row-major order
    i = 0
    for r in range(og_img_shape[0]):
        for c in range(og_img_shape[1]):
            pred_img[r, c] = pred_y[i]
            i = i + 1
            
    return pred_img

def modelIndexWarp(X, y, prop_row, mod, og_image_shape, nbs = 1):
    """Models pixel values as a funciton of their position in
       the image using specified proportion of pixels as training
       data, algorith, and hyperparameters (for KNN)

    Args:
        X (Numpy array): Feature matrix generated using getIndexFeatures()
        y (Numpy array): Pixel values generated using getIndexFeatures()
        prop_row (float): Proportion of pixels to use during training
        mod (str): 'rf' for random forest; 'xgb' for gradient boosted decision
                   trees; 'knn' for k-nearest neighbor; 'et' for extra trees
                   regressor
        nbs (int): Number of neighbors hyperparameter for 'knn'
        og_image_shape (): shape of original image

    Returns:
        PIL Image: Predicted image
    """

    # initialize model object as specified
    if mod == "rf":
        model = RandomForestRegressor()
    if mod == "xgb":
        model = XGBRegressor(objective='reg:squarederror')
    if mod == "knn":
        model = KNeighborsRegressor(n_neighbors = nbs)
    if mod == "et":
        model = ExtraTreesRegressor()
        
    # sample pixels to use for training
    row_sample = np.random.choice(range(X.shape[0]), size=round(prop_row*X.shape[0]), replace = False)
    X_sample = X[row_sample, :]
    y_sample = y[row_sample]

    # fit model to training sample
    reg = model.fit(X_sample, y_sample)

    # predict pixel values, convert to integer and clip to 0-255
    pred_y = reg.predict(X)
    pred_y = np.round(pred_y)
    pred_y = np.clip(pred_y, 0, 255)

    # convert array of pixel values to PIL Image
    pred_img = convertIndexToImage(og_image_shape, pred_y)
    result = Image.fromarray(np.uint8(pred_img))

    return result

def lrWarp(img, prop_row, prop_col, iterations, verbose = False):
    """Uses linear regression to model to predict a column of pixel
       values as function of other columns of pixel values, using
       specified proportion of rows (i.e. observations) and columns
       (i.e. features). Process can be repeated using iterations > 1,
       for which the mean result is taken.

    Args:
        img (PIL Image): Original image
        prop_row (float): proportion of rows (i.e. observations) to use during training
        prop_col (float): proportion of columns (i.e. features) to use during training
        iterations (int): Number of iterations; more iterations creates smoother result
        verbose (bool, optional): Print iteration. Defaults to False.

    Returns:
        PIL Image: Predicted image
    """

    # convert image to array
    img_array = asarray(img)

    # initialize 3-D empty array for predicted pixel values for each iteration
    pred_img = np.empty(((img_array.shape[0], img_array.shape[1], iterations)))
            
    # loop through iterations        
    for i in range(iterations):
        if verbose == True:
            print("Processing Iteration #", i + 1)

        # loop through each column
        for col in range(img_array.shape[1]):

            # construct training X and y
            X = np.delete(img_array, col, axis=1)
            y = img_array[:, col]

            # subset row and column sample as specified
            row_sample = np.random.choice(range(X.shape[0]), size=round(prop_row*X.shape[0]), replace = False)
            col_sample = np.random.choice(range(X.shape[1]), size=round(prop_col*X.shape[1]), replace = False)
            X_rc_samp = X[row_sample, :]
            X_rc_samp = X_rc_samp[:, col_sample]
            y_r_samp = y[row_sample]

            # fit linear regression model
            reg = LinearRegression().fit(X_rc_samp, y_r_samp)

            # predict entire column of pixels
            X_c_samp = X[:, col_sample]
            pred_img[:, col, i] = reg.predict(X_c_samp)

    # take mean over all iterations        
    pred_img_flat = pred_img.mean(axis = 2)  

    # convert to integer, clip to 0-255, and convert to image
    pred_img_flat = np.round(pred_img_flat)
    pred_img_flat = np.clip(pred_img_flat, 0, 255)
    result = Image.fromarray(np.uint8(pred_img_flat))
    
    return result

def permutationWarp(img, iterations, max_block_prop):
    """Permute randomly sized blocks of columns and rows

    Args:
        img (PIL Image): Original image to be warped
        iterations ([type]): number of permutation iterations;
                             more iterations creates smoother result
        max_block_prop (float): maximum block size as proportion of columns or 
                                rows.

    Returns:
        PIL Image: Permuted image
    """

    # convert image to array
    img_array = asarray(img)

    # initialize 3-D array to hold results for all iterations
    pred_img = np.empty(((img_array.shape[0], img_array.shape[1], iterations)))

    # loop over iterations
    for i in range(iterations):

        #initialize empty array for single resulting image
        pred_img_single = np.empty(img_array.shape)

        # determine whether columns or rows will be permuted, and transpose image as appropriate
        d = np.random.binomial(1, 0.5)
        if d == 1:
            img_array = img_array.T
            pred_img_single = pred_img_single.T

        # randomly obtain block size to b permuted
        block_size = np.random.choice(range(1, round(max_block_prop*img_array.shape[0])))

        # permute through max. number of blocks with size block_size
        s = 0
        while s + block_size < img_array.shape[0]:

            # obtain permuted order for appropriate block
            p_order = np.random.choice(range(s, s+block_size), size = block_size, replace = False)

            # store permuted row or column vectors
            pred_img_single[s:s+block_size, :] = img_array[p_order, :]

            s = s + block_size

        # final block, which may be of size < block_size
        p_order = np.random.choice(range(s, img_array.shape[0]), size = (img_array.shape[0] - s), replace = False)
        pred_img_single[s:img_array.shape[0], :] = img_array[p_order, :]

        # convert image back to appropriate orientation
        if d == 1:
            img_array = img_array.T
            pred_img_single = pred_img_single.T

        # store single permuted image 
        pred_img[:, :, i] = pred_img_single

    # tak emean over all iterations and convert to PIL Image
    pred_img = pred_img.mean(axis = 2) 
    result = Image.fromarray(np.uint8(pred_img))
    
    return result

