# Pixel Modeling
Fitting statistical and machine learning models to image pixels to obtain interesting effects

## Overview
Because digitized images are just arrays of numbers, there are a wide variety of ways to manipulate those numbers and thus manipulate images. Image effects, as in something like Photoshop or Instagram, really just involve targeted mathematical operations on the numeric arrays underlying any given image. I thought it would be fun to experiment with different approaches to manipulate image pixel values, with a focus on incorporating randomness and utilizing regression algorithms, which can create some suprising and fascinating results. 

I should note that this repository is intended more as a source of inspiration for further experimentation as opposed to providing nicely packaged functionality, though I have included a a small number of standalone functions. This is just a personal project that I try to tinker with when I find the time, and I hope that it can inspire others to utilize programming, statistics, and machine learning to create weird and novel art.

## Organization
- /data: you will want to create a /data directory to house the original images that you want to tinker with. 
- /code: contains some standalone functions in pixel_modeling.py, which I use in the demonstration.ipynb notebook in the /notebooks directory
- /notebooks: contains a simple demonstration of some functionalities in demonstration.ipynb as well as some further experimentation in experimentation.ipynb
- /reports: contains html reports corresponding to the demonstration.ipynb and experimentation.ipynb notebooks found in the /notebooks directory.
- /results: contains the results produced in the demonstration.ipynb and experimentation.ipynb notebooks found in the /notebooks directory.



