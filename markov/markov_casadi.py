import random, pylab
import os, numpy
from scipy.stats import norm
import sys
sys.path.append('casadi')
import casadi
import time
STEPS = 252

def getCdf():
    x = casadi.SX.sym('x')
    sqrt2 = casadi.sqrt(2)
    return casadi.Function('cdf', [x], [.5 - .5 * casadi.erf(-x/sqrt2)])

cdf = getCdf()

def get_f():
    r_t, h_i_tm1, lambda_i, gamma_i = casadi.SX.sym('r_t'), casadi.SX.sym('h_i_tm1'), casadi.SX.sym('lambda_i'), casadi.SX.sym('gamma_i')
    z = 1/casadi.sqrt(2*numpy.pi * h_i_tm1) * casadi.exp(-((r_t - (lambda_i + gamma_i * casadi.sqrt(h_i_tm1))) ** 2.) / (2 * h_i_tm1))
    return casadi.Function('f', [r_t, h_i_tm1, lambda_i, gamma_i], [z])

f = get_f()

def get_f_i():
    delta_i_t, b_i, c_i = casadi.SX.sym('delta_i_t'), casadi.SX.sym('b_i'), casadi.SX.sym('c_i'),
    z = casadi.sqrt((delta_i_t - b_i)**2) - c_i * (delta_i_t - b_i)
    return casadi.Function('f_i', [delta_i_t, b_i, c_i], [z])

f_i = get_f_i()

def get_h_i_t():
    w_i, alpha_i, beta_i, b_i, c_i, mu_i, v_i, h_tm1_agg_i, delta_t_agg_i = casadi.SX.sym('w_i'), casadi.SX.sym('alpha_i'), casadi.SX.sym('beta_i'), casadi.SX.sym('b_i'), casadi.SX.sym('c_i'), casadi.SX.sym('mu_i'), casadi.SX.sym('v_i'), casadi.SX.sym('h_tm1_agg_i'), casadi.SX.sym('delta_t_agg_i'),
    boolM = mu_i > 0
    sqrthtpowmu = casadi.sqrt(h_tm1_agg_i) ** mu_i
    zTrue = (w_i + alpha_i * sqrthtpowmu * f_i(delta_t_agg_i, b_i, c_i) ** v_i + beta_i * sqrthtpowmu) ** (2./mu_i)
    sqrtht = casadi.sqrt(h_tm1_agg_i)
    zFalse= (casadi.exp(w_i + alpha_i * f_i(delta_t_agg_i, b_i, c_i)  ** v_i + beta_i * casadi.log(sqrtht)) ** (2.))
    z = casadi.if_else(boolM, zTrue, zFalse)
    return casadi.Function('h_i_t',
        [w_i, alpha_i, beta_i, b_i, c_i, mu_i, v_i, h_tm1_agg_i, delta_t_agg_i],
        [z])

h_i_t = get_h_i_t()

def get_P_t(d_1, e_1, r_t):
    return cdf(d_1 + e_1 * r_t)

def get_Q_t(d_2, e_2, r_t):
    return cdf(d_2 + e_2 * r_t)

def get_p_1_t():
    '''
    note g_1_tm1 = f(r_t...|S_ti1-i and r_t-1)

    '''
    P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1 = casadi.SX.sym('P_t'), casadi.SX.sym('Q_t'), casadi.SX.sym('g_1_tm1'), casadi.SX.sym('p_1_tm1'), casadi.SX.sym('g_2_tm1'),
    prob = g_1_tm1 * p_1_tm1 / (g_1_tm1 * p_1_tm1+ g_2_tm1 * (1 - p_1_tm1))
    z = P_t * prob + (1 - Q_t) * (1 - prob)
    return casadi.Function('p_1_t', [P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1], [z])

p_1_t_f = get_p_1_t()

def get_h_tm1_agg_i():
    p_1_tm1_agg_i, h_1_tm1, h_2_tm1, lambda1, gamma1, lambda2, gamma2 = casadi.SX.sym('p_1_tm1_agg_i'), casadi.SX.sym('h_1_tm1'), casadi.SX.sym('h_2_tm1'), casadi.SX.sym('lambda1'), casadi.SX.sym('gamma1'), casadi.SX.sym('lambda2'), casadi.SX.sym('gamma2'),
    z = (p_1_tm1_agg_i * h_1_tm1 + (1-p_1_tm1_agg_i) * h_2_tm1 +
        p_1_tm1_agg_i * (1-p_1_tm1_agg_i) * (lambda1 + gamma1*casadi.sqrt(h_1_tm1) - (lambda2 + gamma2*casadi.sqrt(h_2_tm1))) **2)
    return casadi.Function('h_tm1_agg_i', [p_1_tm1_agg_i, h_1_tm1, h_2_tm1, lambda1, gamma1, lambda2, gamma2], [z])

h_tm1_agg_i = get_h_tm1_agg_i()

def get_delta_t_agg_i():
    p_1_tm1_agg_i, r_t, lambda_1, gamma_1, lambda_2, gamma_2, h_1_tm1, h_2_tm1 = casadi.SX.sym('p_1_tm1_agg_i'), casadi.SX.sym('r_t'), casadi.SX.sym('lambda_1'), casadi.SX.sym('gamma_1'), casadi.SX.sym('lambda_2'), casadi.SX.sym('gamma_2'), casadi.SX.sym('h_1_tm1'), casadi.SX.sym('h_2_tm1'),
    p1 =       p_1_tm1_agg_i * ((r_t- (lambda_1 + gamma_1 * casadi.sqrt(h_1_tm1)))/casadi.sqrt(h_1_tm1))
    p2 = (1 - p_1_tm1_agg_i) * ((r_t- (lambda_2 + gamma_2 * casadi.sqrt(h_2_tm1)))/casadi.sqrt(h_2_tm1))
    z = p1 + p2
    return casadi.Function('delta_t_agg_i', [p_1_tm1_agg_i, r_t, lambda_1, gamma_1, lambda_2, gamma_2, h_1_tm1, h_2_tm1], [z])

delta_t_agg_i = get_delta_t_agg_i()

def get_p_1_tm1_agg_1():
    d_1, e_1, lambda_1, gamma_1, h_1_tm1, p_1_tm1, d_2, e_2, lambda_2, gamma_2, h_2_tm1, p_2_tm1 = casadi.SX.sym('d_1'), casadi.SX.sym('e_1'), casadi.SX.sym('lambda_1'), casadi.SX.sym('gamma_1'), casadi.SX.sym('h_1_tm1'), casadi.SX.sym('p_1_tm1'), casadi.SX.sym('d_2'), casadi.SX.sym('e_2'), casadi.SX.sym('lambda_2'), casadi.SX.sym('gamma_2'), casadi.SX.sym('h_2_tm1'), casadi.SX.sym('p_2_tm1'),
    p1term =      cdf(d_1+e_1*(lambda_1 + gamma_1 * casadi.sqrt(h_1_tm1)))*p_1_tm1
    p2term = (1 - cdf(d_2+e_2*(lambda_2 + gamma_2 * casadi.sqrt(h_2_tm1))))*p_2_tm1
    z = p1term/(p1term + p2term)
    return casadi.Function('p_1_tm1_agg_1', [d_1, e_1, lambda_1, gamma_1, h_1_tm1, p_1_tm1, d_2, e_2, lambda_2, gamma_2, h_2_tm1, p_2_tm1], [z])

p_1_tm1_agg_1_f = get_p_1_tm1_agg_1()

class Params(object):
    def __init__(self, modelType, paramNum, mu, v, b, c, gamma, lamb, alpha, beta ,omega, d, e, asVals=False):
        #parameters controlling garch specification
        self.modelType = modelType
        assert modelType in ['GARCH', 'EGARCH']
        a = casadi.SX.sym
        paramNum = str(paramNum)
        self.mu, self.mu_val = a('mu' + paramNum), mu
        self.v, self.v_val= a('v' + paramNum), v
        self.b, self.b_val= a('b' + paramNum), b
        self.c, self.c_val= a('c' + paramNum), c

        #garch-mean
        self.gamma, self.gamma_val = a('gamma' + paramNum), gamma # risk premium on the vol
        self.lamb, self.lamb_val = a('lamb' + paramNum), lamb # mean
        #garch other
        self.alpha, self.alpha_val = a('alpha' + paramNum), alpha # innovations
        self.beta, self.beta_val = a('beta' + paramNum), beta # lags of vol
        self.omega, self.omega_val = a('omega' + paramNum), omega # mean of vol

        #regime switching parameterization
        self.d, self.d_val = a('d' + paramNum), d # parameters controlling p/q
        self.e, self.e_val = a('e' + paramNum), e # parameters controlling p/q
        self.d, self.d_val = d, d # parameters controlling p/q
        self.e, self.e_val = e, e # parameters controlling p/q
        self.params = 'mu', 'v', 'b', 'c', 'gamma', 'lamb', 'alpha', 'beta', 'omega' #, 'd', 'e'
        if asVals:
            for p in self.params:
                setattr(self, p, getattr(self, p + '_val'))

    def _getSubs(self):
        subs = []
        if self.modelType=='EGARCH':
            subs.append(['mu', 0])
            subs.append(['v', 1])
            subs.append(['b', 0])
        elif self.modelType=='GARCH':
            subs.append(['mu', 2])
            subs.append(['v', 2])
            subs.append(['b', 0])
            subs.append(['c', 0])
        return subs

    def subsForModel(self, ll):
        subs = self._getSubs()
        params = casadi.vertcat(*[getattr(self, p) for p, _ in subs])
        paramVals = casadi.vertcat(*[val for _, val in subs])
        return casadi.substitute(ll, params, paramVals)

    @property
    def vertcat(self):
        modelSubs = [x[0] for x in self._getSubs()]
        return casadi.vertcat(*[getattr(self, p) for p in self.params if p not in modelSubs])

    @property
    def vertcatvals(self):
        modelSubs = [x[0] for x in self._getSubs()]
        return casadi.vertcat(*[getattr(self, p + '_val') for p in self.params if p not in modelSubs])

    @property
    def vals(self):
        modelSubs = [x[0] for x in self._getSubs()]
        return [getattr(self, p + '_val') for p in self.params if p not in modelSubs]

    def bound(self, var):
        if 'alpha' in var.name():
            return (0, 1)
        elif 'beta' in var.name():
            return (0, 1)
        elif 'omega' in var.name():
            return (0, numpy.inf)
        else:
            return (-numpy.inf, numpy.inf)


def loglikilihood(r, params1, params2, ):

    running_sum = 0
    p_1_tm1 = 1
    p_2_tm1 = 0
    g_1_tm1 = 1
    g_2_tm1 = 1
    h_1_tm1 = .25 ** 2 / STEPS
    h_2_tm1 = .25 ** 2 / STEPS
    p_1_tm1_agg_1 = .5
    p_1_tm1_agg_2 = .5

    pm1 = params1
    pm2 = params2
    p_1, p_2, h_1, h_2 = [], [], [], []
    for t in xrange(1, len(r)):
        r_t = r[t]
        P_t = get_P_t(pm1.d, pm1.e, r_t)
        Q_t = get_Q_t(pm2.d, pm2.e, r_t)
        p_1_t = p_1_t_f(P_t, Q_t, g_1_tm1, p_1_tm1, g_2_tm1)
        p_2_t = 1 - p_1_t
        #import pdb; pdb.set_trace()
        p_1_tm1_agg_1 = p_1_tm1_agg_1_f(pm1.d, pm1.e, pm1.lamb, pm1.gamma, h_1_tm1, p_1_tm1, pm2.d, pm2.e, pm2.lamb, pm2.gamma, h_2_tm1, p_2_tm1)
        p_1_tm1_agg_2 = 1- p_1_tm1_agg_1
        h_tm1_agg_1 = h_tm1_agg_i(p_1_tm1_agg_1, h_1_tm1, h_2_tm1, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma)
        h_tm1_agg_2 = h_tm1_agg_i(p_1_tm1_agg_2, h_1_tm1, h_2_tm1, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma)
        delta_t_agg_1 = delta_t_agg_i(p_1_tm1_agg_1, r_t, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma, h_1_tm1, h_2_tm1)
        delta_t_agg_2 = delta_t_agg_i(p_1_tm1_agg_2, r_t, pm1.lamb, pm1.gamma, pm2.lamb, pm2.gamma, h_1_tm1, h_2_tm1)

        h_1_t = h_i_t(pm1.omega, pm1.alpha, pm1.beta, pm1.b, pm1.c, pm1.mu, pm1.v, h_tm1_agg_1, delta_t_agg_1)
        h_2_t = h_i_t(pm2.omega, pm2.alpha, pm2.beta, pm2.b, pm2.c, pm2.mu, pm2.v, h_tm1_agg_2, delta_t_agg_2)
        #if t % 50 == 1:
        #    import pdb; pdb.set_trace()
        #    print (h_1_tm1 * STEPS)**.5, (h_1_t * STEPS)**.5

        f_1_t = f(r_t, h_1_tm1, pm1.lamb, pm1.gamma)
        f_2_t = f(r_t, h_2_tm1, pm2.lamb, pm2.gamma)
        #import pdb; pdb.set_trace()
        running_sum += casadi.log(f_1_t)
        running_sum += casadi.log(f_2_t)

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

    return -running_sum, (p_1, p_2, h_1, h_2)

class Timer(object):
    def __init__(self, msg, verbose=True):
        self.msg = msg
        self.verbose=verbose

    def __enter__(self, *args):
        self.now = time.clock()

    def __exit__(self, *args):
        elapse = (time.clock()- self.now)
        print "%s: %s" % (self.msg, elapse)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    if False:
        frm = pd.read_csv('f.csv')

        adjClose = frm['Adj Close']
        r = numpy.log(adjClose/adjClose.shift(1))[1:]
    else:
        frm = pd.read_csv('switching model.csv')
        r = numpy.array(frm['Mean return']) /STEPS**.5
    #r = numpy.hstack([r]*10)
    print len(r)

    a = casadi.SX.sym
    #def __init__(self, mu, v, b, c, lamb, gamma, alpha, beta, omega, d, e):
    asVals=False
    params1 = Params('GARCH', 1, 2, 2, 0, 0, .0, 0, .10, .8, .001/STEPS, 2.1, 2.1, asVals=asVals)
    params2 = Params('GARCH', 2, 2, 2, 0, 0, .0, 0, .06, .92, .001/STEPS, 2.1, 6.8, asVals=True)

    with Timer('ll calc'):
        ll, (p_1, p_2, h_1, h_2) = loglikilihood(r, params1, params2)
    if not asVals:
        with Timer('subs model'):
            ll = params1.subsForModel(ll)
            #ll = params2.subsForModel(ll)
        params = casadi.vertcat(params1.vertcat) #, params2.vertcat)
        param_vals = casadi.vertcat(params1.vertcatvals) #, params2.vertcatvals)
        with Timer('subs ll'):
            print casadi.substitute(ll, params, param_vals)
        #with Timer('calc hess'):
        #    hess = casadi.hessian(ll, params)
        #with Timer('subs hess'):
        #    hessret = casadi.substitute(hess[0], params, param_vals)
        #    hessret2 = casadi.substitute(hess[1], params, param_vals)
        with Timer('calc jacob'):
            jacob = casadi.jacobian(ll, params)
        with Timer('subs jacob'):
            jacobret = casadi.substitute(jacob, params, param_vals)
        #import pdb; pdb.set_trace()

        bounds = []
        paramsFlat = [params[x] for x in xrange(params.shape[0])]
        for count, value in enumerate(paramsFlat):
            bounds.append(params1.bound(value))

        def optim(x):
            #for x_i, value in zip(x, paramsFlat):
                #print '%s: %s' % (x_i, value)
            ret = casadi.substitute(ll, params, casadi.vertcat(*x))
            return float(ret)

        def fprime(x):
            ret= casadi.substitute(jacob, params, casadi.vertcat(*x))
            ret= numpy.array([float(ret[i]) for i in xrange(ret.shape[1])])
            return ret


        from scipy import optimize
        with Timer('optimize'):
            #opt, fx, its, _, smode = optimize.fmin_slsqp(optim, params1.vals + params2.vals, bounds=bounds, fprime=fprime, iprint=20, full_output=True)
            opt, fx, its, _, smode = optimize.fmin_slsqp(optim, params1.vals, bounds=bounds, fprime=fprime, iprint=20, full_output=True)
        print "x: %s" % opt
        print "ll(x): %s" % fx
        print "warn: %s" % smode
        print "n iter: %s" % its

        import itertools
        for val, value in itertools.izip(paramsFlat, opt):
            print "%s: %s" % (val, value)
    else:
        print ll
        print [(x*STEPS)**.5 for x in h_1[::50]]
        dat = pd.DataFrame(dict(p_1=p_1, p_2=p_2, vol_1=numpy.sqrt(h_1), vol_2=numpy.sqrt(h_2)))
        dat.to_csv('results.csv')


