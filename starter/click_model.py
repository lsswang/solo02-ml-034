"""Existing relevance-only click classifier, without examination probability."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def objective(w, rows, ridge):
    X=np.array([r['features'] for r in rows]);y=np.array([r['click'] for r in rows]);a=np.array([r['weight'] for r in rows]);z=X@w
    loss=np.sum(a*(np.logaddexp(0,z)-y*z))+ridge/2*np.dot(w,w)
    grad=X.T@(a*(expit(z)-y))+ridge*w
    return float(loss),grad

def fit(rows,dimension,ridge=.1):
    r=minimize(objective,np.zeros(dimension),args=(rows,ridge),jac=True,method='L-BFGS-B',bounds=[(-3,3)]*dimension)
    return {'w':r.x.tolist(),'objective':float(r.fun)}
