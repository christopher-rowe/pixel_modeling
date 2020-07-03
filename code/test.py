# %%
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
from tensorflow.keras import layers

from xgboost import XGBRegressor



# %%
def processImage(file, grayscale = True, basewidth = 280):
    
    img = Image.open(file).convert('L')
        
    # resize image
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    
    return img

def LRWarp(img, prop_row, prop_col, iterations, verbose = False):
    
    # convert image to array
    img_array = asarray(img)
    pred_img = np.empty(((img_array.shape[0], img_array.shape[1], iterations)))
            
    for i in range(iterations):
        if verbose == True:
            print("Processing Iteration #", i + 1)
        for col in range(img_array.shape[1]):
            X = np.delete(img_array, col, axis=1)
            y = img_array[:, col]
            row_sample = np.random.choice(range(X.shape[0]), size=round(prop_row*X.shape[0]), replace = False)
            col_sample = np.random.choice(range(X.shape[1]), size=round(prop_col*X.shape[1]), replace = False)
            X_rc_samp = X[row_sample, :]
            X_rc_samp = X_rc_samp[:, col_sample]
            y_r_samp = y[row_sample]
            reg = LinearRegression().fit(X_rc_samp, y_r_samp)
            X_c_samp = X[:, col_sample]
            pred_img[:, col, i] = reg.predict(X_c_samp)
            
    pred_img_flat = pred_img.mean(axis = 2)     
    pred_img_flat = np.round(pred_img_flat)
    pred_img_flat = np.clip(pred_img_flat, 0, 255)
    result = Image.fromarray(np.uint8(pred_img_flat))
    
    return result

def LRWarp2(img1, img2, prop_row, prop_col, iterations, verbose = False):
    
    # convert image to array
    img1_array = asarray(img1)
    img2_array = asarray(img2)
    pred_img = np.empty(((img1_array.shape[0], img1_array.shape[1], iterations)))
            
    for i in range(iterations):
        if verbose == True:
            print("Processing Iteration #", i + 1)
        for col in range(img1_array.shape[1]):
            X1 = np.delete(img1_array, col, axis=1)
            X2 = np.delete(img2_array, col, axis=1)
            y1 = img1_array[:, col]
            row_sample = np.random.choice(range(X1.shape[0]), size=round(prop_row*X1.shape[0]), replace = False)
            col_sample = np.random.choice(range(X1.shape[1]), size=round(prop_col*X1.shape[1]), replace = False)
            X1_rc_samp = X1[row_sample, :]
            X1_rc_samp = X1_rc_samp[:, col_sample]
            y1_r_samp = y1[row_sample]
            reg = LinearRegression().fit(X1_rc_samp, y1_r_samp)
            X2_c_samp = X2[:, col_sample]
            pred_img[:, col, i] = reg.predict(X2_c_samp)
            
    pred_img_flat = pred_img.mean(axis = 2)     
    pred_img_flat = np.round(pred_img_flat)
    pred_img_flat = np.clip(pred_img_flat, 0, 255)
    result = Image.fromarray(np.uint8(pred_img_flat))
    
    return result

def getIndexData(img, poly_degree):
    
    img_t = asarray(img)
    X = np.empty((img_t.shape[0]*img_t.shape[1], 2))
    y = np.empty(img_t.shape[0]*img_t.shape[1])
    i = 0
    for r in range(img_t.shape[0]):
        for c in range(img_t.shape[1]):
            X[i, 0] = r
            X[i, 1] = c
            y[i] = img_t[r, c]
            i = i + 1
    poly = PolynomialFeatures(poly_degree)
    X = poly.fit_transform(X)
    X = np.delete(X, 0, axis = 1)
    X = stats.zscore(X, axis = 0)

    return X, y

def convertIndexToImage(og_img_shape, pred_y):
    
    pred_img = np.empty(og_img_shape)
    
    i = 0
    for r in range(og_img_shape[0]):
        for c in range(og_img_shape[1]):
            pred_img[r, c] = pred_y[i]
            i = i + 1
            
    return pred_img

def modelIndexWarp(X, y, prop_row, mod, nbs, og_image_shape):
    if mod == "rf":
        model = RandomForestRegressor()
    if mod == "xgb":
        model = XGBRegressor(objective='reg:squarederror')
    if mod == "knn":
        model = KNeighborsRegressor(n_neighbors = nbs)
    if mod == "et":
        model = ExtraTreesRegressor()
        
    row_sample = np.random.choice(range(X.shape[0]), size=round(prop_row*X.shape[0]), replace = False)
    X_sample = X[row_sample, :]
    y_sample = y[row_sample]
    reg = model.fit(X_sample, y_sample)
    pred_y = reg.predict(X)
    pred_y = np.round(pred_y)
    pred_y = np.clip(pred_y, 0, 255)
    pred_img = convertIndexToImage(og_image_shape, pred_y)
    result = Image.fromarray(np.uint8(pred_img))
    return result

def permutationWarp(img, iterations, max_block_prop):

    img_array = asarray(img)
    pred_img = np.empty(((img_array.shape[0], img_array.shape[1], iterations)))

    for i in range(iterations):
        pred_img_single = np.empty(img_array.shape)

        d = np.random.binomial(1, 0.5)
        if d == 1:
            img_array = img_array.T
            pred_img_single = pred_img_single.T

        block_size = np.random.choice(range(1, round(max_block_prop*img_array.shape[0])))
        s = 0
        while s + block_size < img_array.shape[0]:
            p_order = np.random.choice(range(s, s+block_size), size = block_size, replace = False)
            pred_img_single[s:s+block_size, :] = img_array[p_order, :]
            s = s + block_size

        p_order = np.random.choice(range(s, img_array.shape[0]), size = (img_array.shape[0] - s), replace = False)
        pred_img_single[s:img_array.shape[0], :] = img_array[p_order, :]

        if d == 1:
            img_array = img_array.T
            pred_img_single = pred_img_single.T

        pred_img[:, :, i] = pred_img_single

    pred_img = pred_img.mean(axis = 2) 
    result = Image.fromarray(np.uint8(pred_img))
    
    return result

# %%
# import photo
os.chdir('/Users/chrisrowe/Documents/personal_projects/pixel_modeling/data/')

# open and process images
img_cr = processImage('cr.jpg')

# %%
X, y = getIndexData(img_cr, poly_degree = 30)
knn = modelIndexWarp(X, y, prop_row = 0.003,
              mod = 'knn', nbs = 1, og_image_shape = asarray(img_cr).shape)
knn

# %%
LRWarp(img_cr, prop_row = 1, prop_col = 1, 
       iterations = 1, verbose = False)

# %%
permutationWarp(img_cr, 10, 0.2)

# %%
