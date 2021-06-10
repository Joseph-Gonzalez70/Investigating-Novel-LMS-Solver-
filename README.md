# Using Coresets And Sketches To Maintain Accuracy And Improve Run-Time of Least-Mean-Squares Solvers 

## Abstract

In the early 1900s, Constantin Carathéodory and Ernst Steinitz developed and proved Carathéodory's theorem. This theorem states that we can write each point contained in a convex hull of n points in R<sup>d</sup>  as a convex combination of at most d+1 points. These details suggest that we can maintain a d<sup>2</sup> + 1 scaled set of points that we can use to calculate the covariance matrix for a design matrix X and, with some other calculations, solve the LMS equation. While this method avoids updating the linear combinations of all n points and reduces numerical errors, it is not popular for its run time of O(n<sup>2</sup> d<sup>2</sup>) or O(nd<sup>3</sup>).

In this paper, we implement the novel LMS corest algorithm from *Fast and Accurate Least-Mean-Squares Solvers*(Maalouf et al.). This method modifies Carathéodory's theorem(combines coresets and sketches) to improve the runtime and maintain accuracy. For the application, we show that this solver can boost the performance of existing solvers such as those in the scikit-learn library. 

# How to navigate this project:

## Pre-implementation steps

### 1. Data retrieval and preprocessing:

**Folder Name:** Code

**File name:** DataPreprocessing.py

*Contents:* This file contains code to import and standardize the data sets. The code removes features that are not useful for regression and provides an option to standardize the data. We call the functions in this file before we implement the LMS-Coreset algorithm.

**Related Files:** AdjustData.py and StandardizeSplit.py

*Contents:* DataPreprocessing.py imports and calls functions from these files. The AdjustData.py cleans the data and removes features(if necessary) that are not useful for regression. StandardizeSplit.py contains a standardizing and a data split function(test/train).

### 2. Least-Mean-Square Solvers:

**Folder Name:** Code

**File name:** LeastSquaresFunctions.py

*Contents:* This file contains functions to fit and test the LMS Solvers. This includes linear regression, ridge regression, lasso regression, and elastic net regression.

## Implementation

### 3. Algorithm Implementation:

**Folder Name:** Code

**File names:** AlgorithmImplement.ipynb

*Contents:* This files contain the functions for the LMS-Coreset algorithm(algorithms 16, 1, 2, and 5). The functions return the LMS coreset.

### 4. Report And Results:

**File names:** STA208_Report.pdf

*Contents:* This file contains the report for our project. The report includes methodologies and implementation results.

### Optional: 

The notebook folder contains "brief" exploratory data analyses for the data sets in the data folder. Since the goal for the project is to implement the LMS coreset algorithm, we list this notebook as an optional read.

## Authors and Contributions:

Zhongxuan Liu[bkbliu@ucdavis.edu]: Algorithms implementation, experiments and conclusion.

Shanshan Chen[ssnchen@ucdavis.edu]: Algorithm 1,2 implementations and algorithm 16,1,2,5 debugging

Joseph Gonzalez[joegonza@ucdavis.edu] abstract, introduction, methodologies, appendices, and data summaries

