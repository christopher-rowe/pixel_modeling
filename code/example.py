# %% import pixel_modeling library
import pixel_modeling as pm
import numpy as np
import os

# %% import and process images
path = os.getcwd() 
os.chdir(os.path.abspath(os.path.join(path, os.pardir)))
img = pm.processImage('data/cr.jpg', grayscale = True, 
                      basewidth = 800)

# %% Get index features with and without polynomial/interaction terms
X1, y = pm.getIndexFeatures(img, poly_degree = 1, include_y = True)
X2 = pm.getIndexFeatures(img, poly_degree = 2, include_y = False)
X5 = pm.getIndexFeatures(img, poly_degree = 5, include_y = False)

# %% Example - k-nearest neighbor
pm.modelIndexWarp(X1, y, prop_row = 0.0005,
                  mod = 'knn', nbs = 1, 
                  og_image_shape = np.asarray(img).shape)

# %% Example - xgboost
pm.modelIndexWarp(X5, y, prop_row = 0.005,
                  mod = 'xgb', nbs = 1, 
                  og_image_shape = np.asarray(img).shape)

# %% Example - random forest
pm.modelIndexWarp(X1, y, prop_row = 0.005,
                  mod = 'rf', nbs = 1, 
                  og_image_shape = np.asarray(img).shape)

# %% Example - linear regression warp
pm.lrWarp(img, prop_row = 0.25, prop_col = 0.01, 
          iterations = 20, verbose = False)

# %% permutation warp
pm.permutationWarp(img, iterations = 10, max_block_prop = 0.2)
