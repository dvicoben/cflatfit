import numpy as np

_SMALL_FLOAT_NONZERO = 1e-16

def log_or_zero_float(x: float) -> float:
    return np.log(x) if x > 0.0 else 0.0

def log_or_zero(x: np.ndarray) -> np.ndarray:
    """ Evaluate to log(x) for x > 0 and to 0 otherwise.

    Parameters
    ----------
    x : np.ndarray
        Argument.

    Returns
    -------
    logx : np.ndarray
        Elementwise contains log(x) for x > 0 and zero otherwise.
    """
    r = np.zeros_like(x)
    ma = x > 0
    r[ma] = np.log(x[ma])
    return r

def poisson_chi2(n: np.ndarray, mu: np.ndarray) -> float:
    """Compute asymptotically chi2-distributed cost for Poisson-distributed data.

    See Baker & Cousins, NIM 221 (1984) 437-442.

    Parameters
    ----------
    n : np.ndarray
        Observed counts.
    mu : np.ndarray
        Expected counts per bin.

    Returns
    -------
    nll : float
        Cost function value.
    """
    return 2*np.sum(n*(log_or_zero(n) - log_or_zero(mu)) + mu - n)

def poisson_chi2_W2(n: np.ndarray, mu: np.ndarray, scalefactorW2: np.ndarray) -> float:
    """Likelihood using scaling (error * error) / yield, implemented as in RooFit (https://github.com/root-project/root/blob/master/math/mathcore/src/FitUtil.cxx#L1434) L1566-1586
           
    This is to be used solely to compute the covariance matrix which should follow the SumW2Error approach (https://root.cern/doc/master/classRooAbsPdf.html#RooAbsPdf:fitTo)
        

    Parameters
    ----------
    n : np.ndarray
        Observed counts.
    mu : np.ndarray
        Expected counts per bin.
    scalefactorW2 : np.ndarray
        Bin-wise (error * error) / yield

    Returns
    -------
    nllw2 : float
        Cost function value
    """
    return 2*np.sum(scalefactorW2*(n*(log_or_zero(n) - log_or_zero(mu)) + mu - n))

def cash_statistic(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Cash statistic, as defined in in 10.1140/epjc/s10052-022-11019-z

    Parameters
    ----------
    k : np.ndarray
        Argument.
    lam : np.ndarray
        Argument.

    Returns
    -------
    c : np.ndarray
        Bin-wise cash statistic
    """
    c = 2*(lam - k - k * (log_or_zero(lam) - log_or_zero(k)))
    return c

def poisson_scaled_chi2(n: np.ndarray, nvar: np.ndarray, mu: np.ndarray) -> float:
    """Implements scaled poisson dsitribution, see https://arxiv.org/pdf/1309.1287 and discussion 
    in 10.1140/epjc/s10052-022-11019-z.
    Corresponds to scaling n and model predicution mu by factor t=n/Variance(n)

    Parameters
    ----------
    n : np.ndarray
        Observed counts - sum of weights
    nvar : np.ndarray
        Variance of observed counts (NOT error) - sum of weights square
    mu : np.ndarray
        Expected counts
    
    Returns
    -------
    q : float
        The figure of merit - should be chi2-like
    """
    # nvariance = np.ones_like(nvar)
    # ma = n > 0
    # nvariance[ma] = nvar[ma]
    # t = n/nvariance
    t = n/nvar
    return cash_statistic(t*n, t*mu)

def chi2_template_DA(n: np.ndarray, nvar: np.ndarray, mu: np.ndarray, muvar: np.ndarray) -> float:
    """Implements the figure of merit as per Eq. (24) in 10.1140/epjc/s10052-022-11019-z

    Parameters
    ----------
    n : np.ndarray
        Observed counts - sum of weights
    nvar : np.ndarray
        Variance of observed counts (NOT error) - sum of weights square
    mu : np.ndarray
        Expected counts
    muvar : np.ndarray
        Variance of expected counts (NOT error)

    Returns
    -------
    q : float
        The figure of merit - should be chi2-like
    """
    s = mu/muvar
    t = n/nvar
    beta = (t*n + s*mu)/(t*mu + s*mu + _SMALL_FLOAT_NONZERO)
    q = np.sum(cash_statistic(t*n, beta*t*mu) + cash_statistic(s*mu, beta*s*mu))
    return q

def chi2_template_JSC(n: np.ndarray, mu: np.ndarray, mu_var: np.ndarray) -> float:
    """
    Compute asymptotically chi2-distributed cost for a template fit.

    J.S. Conway, PHYSTAT 2011, https://doi.org/10.48550/arXiv.1103.0354

    Parameters
    ----------
    n : array-like
        Observed counts.
    mu : array-like
        Expected counts. This is the sum of the normalised templates scaled with
        the component yields. Must be positive everywhere.
    mu_var : array-like
        Expected variance of mu. Must be positive everywhere.

    Returns
    -------
    float
        Asymptotically chi-square-distributed test statistic.

    Notes
    -----
    The implementation deviates slightly from the paper by making the result
    asymptotically chi2-distributed, which helps to maximise the numerical
    accuracy for Minuit.
    """
    # n, mu, mu_var = np.atleast_1d(n, mu, mu_var)
    beta_var = mu_var / mu**2
    # Eq. 15 from https://doi.org/10.48550/arXiv.2206.12346
    p = 0.5 - 0.5 * mu * beta_var
    beta = p + np.sqrt(p**2 + n * beta_var)
    return poisson_chi2(n, mu * beta) + np.sum((beta - 1) ** 2 / beta_var)

def nll_gaussian(x: float, mu: float, sigma: float) -> float:
    """Returns -2ln(Gaussian). Meant for gaussian constraints in likelihoods.
    
    NOTE: The constant terms -0.5*ln(2pi) and -ln(sigma) are neglected

    Parameters
    ----------
    x : float
        Point to evaluate at
    mu : float
        Guassian mean
    sigma : float
        Standard deviation

    Returns
    -------
    float
        -2ln(Gaussian(x ; mu, sigma))
    """
    # lngauss = -log_or_zero(sigma) - 0.5*((x-mu)/sigma)**2
    lngauss = -0.5*((x-mu)/sigma)**2
    return -2*lngauss


def nll_poisson(n: float, mu: float) -> float:
    """Poisson term

    Parameters
    ----------
    n : np.ndarray
        Observed counts.
    mu : np.ndarray
        Expected counts per bin.

    Returns
    -------
    nll : float
        Cost function value.
    """
    return 2*(n*(log_or_zero_float(n) - log_or_zero_float(mu)) + mu - n)