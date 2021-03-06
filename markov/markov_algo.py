import math, random, pylab
import os, numpy
from scipy.stats import norm
import algopy

cdf_a = -358./23.
cdf_b = 37./294.
def cdf(x):
    return (algopy.exp(cdf_a* x + 111. * algopy.arctan(cdf_b * x)) + 1) ** -1.


def get_f(r_t, h_i_tm1, lambda_i, gamma_i):
    return 1/algopy.sqrt(2*numpy.pi * h_i_tm1) * algopy.exp(-((r_t - (lambda_i + gamma_i * algopy.sqrt(h_i_tm1))) ** 2.) / (2 * h_i_tm1))

def get_f_i(delta_i_t, b_i, c_i):
    return algopy.sqrt(algopy.square((delta_i_t - b_i))) - c_i * (delta_i_t - b_i)

def get_h_i_t(w_i, alpha_i, beta_i, b_i, c_i, mu_i, v_i, h_tm1_agg_i, delta_t_agg_i):
    if mu_i > 0:
        sqrthtpowmu = algopy.sqrt(h_tm1_agg_i) ** mu_i
        return (w_i + alpha_i * sqrthtpowmu * get_f_i(delta_t_agg_i, b_i, c_i) ** v_i + beta_i * sqrthtpowmu) ** (2./mu_i)
    else:
        sqrtht = algopy.sqrt(h_tm1_agg_i)
        return (algopy.exp(w_i + alpha_i * get_f_i(delta_t_agg_i, b_i, c_i)  ** v_i + beta_i * algopy.log(sqrtht)) ** (2.))

def get_P_t(d_1, e_1, r_t):
    return cdf(d_1 + e_1 * r_t)

def get_Q_t(d_2, e_2, r_t):
    return cdf(d_2 + e_2 * r_t)

def get_p_1_t(P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1):
    '''
    note g_1_tm1 = f(r_t...|S_ti1-i and r_t-1)

    '''
    prob = g_1_tm1 * p_1_tm1 / (g_1_tm1 * p_1_tm1+ g_2_tm1 * (1 - p_1_tm1))
    return P_t * prob + (1 - Q_t) * (1 - prob)


def get_p_2_t(P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1):
    '''
    note g_1_tm1 = f(r_t...|S_ti1-i and r_t-1)

    '''
    prob = g_1_tm1 * p_1_tm1 / (g_1_tm1 * p_1_tm1+ g_2_tm1 * (1 - p_1_tm1))
    return Q_t * (1 - prob) +  (1 - P_t) * prob


def get_h_tm1_agg_i(p_1_tm1_agg_i, h_1_tm1, h_2_tm1, lambda1, gamma1, lambda2, gamma2):
    return (p_1_tm1_agg_i * h_1_tm1 + (1-p_1_tm1_agg_i) * h_2_tm1 +
        p_1_tm1_agg_i * (1-p_1_tm1_agg_i) * (lambda1 + algopy*numpy.sqrt(h_1_tm1) - (lambda2 + gamma1*algopy.sqrt(h_2_tm1))) **2)

def get_delta_t_agg_i(p_1_tm1_agg_i, r_t, lambda_1, gamma_1, lambda_2, gamma_2, h_1_tm1, h_2_tm1):
    p1 =       p_1_tm1_agg_i * ((r_t- (lambda_1 + gamma_1 * algopy.sqrt(h_1_tm1)))/algopy.sqrt(h_1_tm1))
    p2 = (1 - p_1_tm1_agg_i) * ((r_t- (lambda_2 + gamma_2 * algopy.sqrt(h_2_tm1)))/algopy.sqrt(h_2_tm1))
    return p1 + p2

def get_p_1_tm1_agg_1(d_1, e_1, lambda_1, gamma_1, h_1_tm1, p_1_tm1, d_2, e_2, lambda_2, gamma_2, h_2_tm1, p_2_tm1):
    p1term =      cdf(d_1+e_1*(lambda_1 + gamma_1 * algopy.sqrt(h_1_tm1)))*p_1_tm1
    p2term = (1 - cdf(d_2+e_2*(lambda_2 + gamma_2 * algopy.sqrt(h_2_tm1))))*p_2_tm1
    return p1term/(p1term + p2term)

def get_p_1_tm1_agg_2(d_1, e_1, lambda_1, gamma_1, h_1_tm1, p_1_tm1, d_2, e_2, lambda_2, gamma_2, h_2_tm1, p_2_tm1):
    p1term = (1 - cdf(d_1+e_1*(lambda_1 + gamma_1 * algopy.sqrt(h_1_tm1))))*p_1_tm1
    p2term =      cdf(d_2+e_2*(lambda_2 + gamma_2 * algopy.sqrt(h_2_tm1)))*p_2_tm1
    return p1term/(p1term + p2term)

class Params(object):
    def __init__(self, mu, v, b, c, lamb, gamma, alpha, beta, omega, d, e):
        #parameters controlling garch specification
        self.mu = mu
        self.v = v
        self.b = b
        self.c = c

        #garch-mean
        self.gamma = gamma # risk premium on the vol
        self.lamb = lamb # mean
        #garch other
        self.alpha = alpha # innovations
        self.beta = beta # lags of vol
        self.omega = omega # mean of vol

        #regime switching parameterization
        self.d = d # parameters controlling p/q
        self.e = e # parameters controlling p/q

def loglikilihood(r, params1, params2, ):

    running_sum = 0
    p_1_tm1 = .5
    p_2_tm1 = .5
    g_1_tm1 = 1
    g_2_tm1 = 1
    h_1_tm1 = .25 ** 2
    h_2_tm1 = .25 ** 2
    p_1_tm1_agg_1 = .5
    p_1_tm1_agg_2 = .5

    pm1 = params1
    pm2 = params2
    p_1, p_2, h_1, h_2 = [], [], [], []
    for t in xrange(1, len(r)):
        r_t = r[t]
        P_t = get_P_t(pm1.d, pm1.e, r_t)
        Q_t = get_Q_t(pm2.d, pm2.e, r_t)
        p_1_t = get_p_1_t(P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1)
        p_2_t = get_p_2_t(P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1)
        p_1_tm1_agg_1 = get_p_1_tm1_agg_1(pm1.d, pm1.e, pm1.lamb, pm1.gamma, h_1_tm1, p_1_tm1, pm2.d, pm2.e, pm2.lamb, pm2.gamma, h_2_tm1, p_2_tm1)
        p_1_tm1_agg_2 = get_p_1_tm1_agg_2(pm1.d, pm1.e, pm1.lamb, pm1.gamma, h_1_tm1, p_1_tm1, pm2.d, pm2.e, pm2.lamb, pm2.gamma, h_2_tm1, p_2_tm1)
        h_tm1_agg_1 = get_h_tm1_agg_i(p_1_tm1_agg_1, h_1_tm1, h_2_tm1, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma)
        h_tm1_agg_2 = get_h_tm1_agg_i(p_1_tm1_agg_2, h_1_tm1, h_2_tm1, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma)
        delta_t_agg_1 = get_delta_t_agg_i(p_1_tm1_agg_1, r_t, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma, h_1_tm1, h_2_tm1)
        delta_t_agg_2 = get_delta_t_agg_i(p_1_tm1_agg_2, r_t, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma, h_1_tm1, h_2_tm1)
        h_1_t = get_h_i_t(pm1.omega, pm1.alpha, pm1.beta, pm1.b, pm1.c, pm1.mu, pm1.v, h_tm1_agg_1, delta_t_agg_1)
        h_2_t = get_h_i_t(pm2.omega, pm2.alpha, pm2.beta, pm2.b, pm2.c, pm2.mu, pm2.v, h_tm1_agg_2, delta_t_agg_2)

        f_1_t = get_f(r_t, h_1_tm1, pm1.lamb, pm1.gamma)
        f_2_t = get_f(r_t, h_2_tm1, pm2.lamb, pm2.gamma)
        running_sum += algopy.log(f_1_t)
        running_sum += algopy.log(f_2_t)

        p_1_tm1 = p_1_t
        p_2_tm1 = p_2_t
        g_1_tm1 = f_1_t
        g_2_tm1 = f_2_t
        h_1_tm1 = h_1_t
        h_2_tm1 = h_2_t
        p_1.append(p_1_t)
        p_2.append(p_2_t)
        h_1.append(h_1_t)
        h_2.append(h_2_t)

    return running_sum, (p_1, p_2, h_1, h_2)

def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    frm = pd.read_csv('f.csv')

    adjClose = frm['Adj Close']
    r = numpy.log(adjClose/adjClose.shift(1))[1:]

    #def __init__(self, mu, v, b, c, lamb, gamma, alpha, beta, omega, d, e):
    params1 = Params(2, 2, 0, 0, .025, 0.00001, .06, .92, 0, 2.8, 2.1)
    params2 = Params(2, 2, 0, 0, .025, 0.00001, .12, .86, 0, 2.9, 6.9)

    ll, (p_1, p_2, h_1, h_2) = loglikilihood(r, params1, params2)
    print ll
    dat = pd.DataFrame(dict(p_1=p_1, p_2=p_2, vol_1=numpy.sqrt(h_1), vol_2=numpy.sqrt(h_2)))
    dat.to_csv('results.csv')




