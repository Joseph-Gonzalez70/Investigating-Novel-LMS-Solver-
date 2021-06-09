# This file contains the least mean square solvers: Ridge, Lasso, ElasticNet, OLS

# Libraries:
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

def lin_reg(X_tr, y_tr, X_test = None, y_test = None):
    """
    This function performs ordinary least squares linear regression
    Parameters:
        X_tr -> full X data or training set
        y_tr -> full y observations or y training observations
        X_test -> X test set
        y_test -> y test set
    Output:
        1. If X_test is None, function returns the linear regression object(class)
         fitted on X_tr.
        2. If X_test is not None, function returns the linear regression object(class)
        and the mse prediction error on X_test.
    """
    reg = LinearRegression()
    if X_test == None:
        reg.fit(X_tr, y_tr)
        return(reg)
    reg.fit(X_tr, y_tr)
    predict_vals = reg.predict(X_test)
    return(reg, mean_squared_error(y_test, predict_vals))




def ridge_reg(X_tr, y_tr, alpha_values, X_test = None,
              y_test = None, m_folds = 10, fit_intercept = True, normalize = False):
    """
    This function performs Ridge Regression with alpha tuning(RidgeCV)
    Parameters:
        X_tr -> full X data or training set
        y_tr -> full y observations or y training observations
        alpha_values -> Array of alpha values to test.
        X_test -> X test set
        y_test -> y test set
        m_folds -> number of folds to tune alpha(default 10 folds)
        fit_intercept -> option to fit intercept in the model
        normalize -> Option to normalize the X regressors(subtract the mean and divide by the l2-norm).
    Output:
        1. If X_test is None, function returns the ridge regression object(class)
         fitted on X_tr.
        2. If X_test is not None, function returns the ridge regression object(class) and
        the mse prediction error on X_test.
    """
    fit_ridge = RidgeCV(alphas = alpha_values, fit_intercept = fit_intercept,
                        normalize = normalize, cv = m_folds)
    if X_test == None:
        fit_ridge.fit(X_tr, y_tr)
        return(fit_ridge)
    fit_ridge.fit(X_tr, y_tr)
    pred_ridge = fit_ridge.predict(X_test)
    test_error_ridge = mean_squared_error(y_test, pred_ridge)
    return(fit_ridge, test_error_ridge)



def lasso_reg(X_tr, y_tr, alpha_values, X_test = None,
              y_test = None, m_folds = 10, fit_intercept = True, normalize = False, max_i = 1000):
    """
    This function performs Lasso Regression with alpha tuning(LassoCV)
    Parameters:
        X_tr -> full X data or training set
        y_tr -> full y observations or y training observations
        alpha_values -> Array of alpha values to test.
        X_test -> X test set
        y_test -> y test set
        m_folds -> number of folds to tune alpha(default 10 folds)
        fit_intercept -> option to fit intercept in the model
        normalize -> Option to normalize the X regressors(subtract the mean and divide by the l2-norm).
    Output:
        1. If X_test is None, function returns the lasso regression object(class)
         fitted on X_tr.
        2. If X_test is not None, function returns the lasso regression object(class) and
        the mse prediction error on X_test.
    """
    fit_lasso = LassoCV(alphas = alpha_values, fit_intercept = fit_intercept,
                        normalize = normalize, max_iter=max_i, cv = m_folds)
    if X_test == None:
        fit_lasso.fit(X_tr, y_tr)
        return(fit_lasso)
    fit_lasso.fit(X_tr, y_tr)
    pred_lasso = fit_lasso.predict(X_test)
    test_error_lasso = mean_squared_error(y_test, pred_lasso)
    return (fit_lasso, test_error_lasso)



def elastic_reg(X_tr, y_tr, alpha_values, l1_ratio = [.1, .5, .7, .9, .95, .99, 1],
                X_test = None, y_test = None, m_folds = 10, fit_intercept = True , normalize = False, max_i = 1000):
    """
    This function performs elastic net Regression with alpha and l1 ratio tuning(ElasticNetCV)
    Parameters:
        X_tr -> full X data or training set
        y_tr -> full y observations or y training observations
        alpha_values -> Array of alpha values to test.
        l1_ratio -> float or list of floats between 0 and 1 passed to ElasticNet(scaling between l1 and l2 penalties).
                    If l1_ratio = 0 the penalty is an L2 penalty.
                    If l1_ratio = 1 it is an L1 penalty.
                    If 0 < l1_ratio < 1, the penalty is a combination of L1 and L2
        X_test -> X test set
        y_test -> y test set
        m_folds -> number of folds to tune alpha(default 10 folds)
        fit_intercept -> option to fit intercept in the model
        normalize -> Option to normalize the X regressors(subtract the mean and divide by the l2-norm).
    Output:
        1. If X_test is None, function returns the elastic net regression object(class)
         fitted on X_tr.
        2. If X_test is not None, function returns the elastic net regression object(class) and
        the mse prediction error on X_test.
    """
    fit_elastic = ElasticNetCV(l1_ratio = l1_ratio, alphas=alpha_values,
                               fit_intercept = fit_intercept, normalize = normalize, cv = m_folds, max_iter = max_i)
    if X_test == None:
        fit_elastic.fit(X_tr, y_tr)
        return (fit_elastic)
    fit_elastic.fit(X_tr, y_tr)
    pred_elastic = fit_elastic.predict(X_test)
    test_error_elastic = mean_squared_error(y_test, pred_elastic)
    return (fit_elastic,test_error_elastic)