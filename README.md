# Using Coresets And Sketches To Maintain Accuracy And Improve Run-Time of Least-Mean-Squares Solvers 

## Abstract

In the early 1900s, Constantin Carathéodory and Ernst Steinitz developed and proved Carathéodory's theorem. This theorem states that we can write each point contained in a convex hull of n points in R<sup>d</sup>  as a convex combination of at most d+1 points. These details suggest that we can maintain a d<sup>2</sup> + 1 scaled set of points that we can use to calculate the covariance matrix for a design matrix X and, with some other calculations, solve the LMS equation. While this method avoids updating the linear combinations of all n points and reduces numerical errors, it is not popular for its run time of O(n<sup>2</sup> d<sup>2</sup>) or O(nd<sup>3</sup>).

In this paper, we implement the novel LMS corest algorithm from *Fast and Accurate Least-Mean-Squares Solvers*(Maalouf et al.). This method modifies Carathéodory's theorem(combines coresets and sketches) to improve the runtime and maintain accuracy. For the application, we show that this solver can boost the performance of existing solvers such as those in the scikit-learn library. 

# How to navigate this project:

## Pre-implementation steps

Steps 1-3 contain information about the functions called in the main implementation file(ENTER NAME HERE). We ordered the steps following the order they occur in the main implementation file. 

### 1. Data retrieval and preprocessing:

**Folder Name:** Code

**File name:** DataPreprocessing.py

**Contents:** This file contains code to import and standardize the data sets. The code removes features that are not useful for regression and provides an option to standardize the data. We call the functions in this file before we implement the LMS-Coreset algorithm.

*Related Files:* AdjustData.py and StandardizeSplit.py

*Contents:* DataPreprocessing.py imports and calls functions from these files. The AdjustData.py cleans the data and removes features(if necessary) that are not useful for regression. StandardizeSplit.py contains a standardizing and a data split function(test/train).


### 2. LMS-Coreset Algorithm Function:

**File names:** ENTER HERE

**Contents:**  These files contain the functions for the LMS-Coreset algorithm(algorithms 1,2, and 5). The function returns the method coreset.


### 3. Least-Mean-Square Solvers:

**File name:** LeastSquaresFunctions.py

**Contents:** This file contains functions to fit and test the LMS Solvers. This includes linear regression, ridge regression, lasso regresssion, and eleastic net regression.

## Implementation steps

### 4. Algorithm Implementation:

**File names:** ENTER HERE

**Contents:**  ENTER HERE



## Authors:

Zhongxuan Liu[bkbliu@ucdavis.edu]

Shanshan Chen[ssnchen@ucdavis.edu]

Joseph Gonzalez[joegonza@ucdavis.edu]

