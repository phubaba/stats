import numpy


def sampleFractionalPathDaviesHarte(n, L, hurst, seed=None, numSamples=1):
    '''
    samples a fractional brownian motion using the davies-harte method

    2**n steps will be sampled within L
    hurst is the hurst parameter in fractional brownian motion.
        .5 is gaussian
        >.5 long memory
        <.5 rough motion

    where N = 2**n, this algorithm has a complexity of O(N * log(N)) due to fft

    thanks go out to Tom Dieker for implementing in c
    please see his web page at www.columbia.edu/~ad3217/fbm/

    this algorithm could be improved if we had a way of simulating
    random variables without alocating memory
    can also avoid recomputing the eigenvalues over and over

    '''
    N = 2**n
    eig = generateLambdaEigenvalues(N, hurst)[0]
    ret = []
    scaling = (L/float(N)) ** hurst
    for count in xrange(numSamples):
        S, T = generateSandT(eig, N, seed=seed)

        #real part of the fft is the sample path
        S = numpy.real(numpy.fft.fft(S + 1j*T))[:N]
        ret.append(scaling * S)

    return ret


def gn_cov(k, hurst):
    '''
    calculates g(k) E(B_1^H(B_(k+1)^H - B_k^H))

    '''
    if k == 0:
        return 1
    twoHurst = hurst * 2
    return 0.5 * (k + 1) ** twoHurst + 0.5 * abs(k - 1) ** twoHurst - k** twoHurst


def generateLambdaEigenvalues(N, hurst):
    cValues = numpy.fromiter((gn_cov(k, hurst) for k in xrange(0, N)), numpy.float64)
    #note this is a bit tricky, there is a slight offset in the covariance parameters
    cValues = numpy.hstack([cValues, [gn_cov(N, hurst)], cValues[:0:-1]])
    return numpy.fft.fft(cValues), cValues


def generateSandT(eigenValues, N, seed=None):
    m = 2*N
    numpy.random.seed(seed)
    S = numpy.random.normal(size=(m, ))
    T = numpy.random.normal(size=(m, ))
    S = S * numpy.sqrt(eigenValues) / numpy.sqrt(2*m)
    T = T * numpy.sqrt(eigenValues) / numpy.sqrt(2*m)

    S[0] = S[0] * numpy.sqrt(2)
    S[-1] = S[-1] * numpy.sqrt(2)
    T[0] = 0
    T[-1] = 0
    S = numpy.hstack([S, S[::-1]])
    T = numpy.hstack([T, -T[::-1]])
    return S, T
