"""
Vectorized PDF evaluation of the prediction densities.

Same code as the coordinator (coordinator_node/services/functions/density_pdf.py),
so local scores evaluate densities exactly like the live scoring.
"""
import numpy as np
import scipy.stats as st
import numpy as np
from math import gamma


def _asarray(x):
    return np.asarray(x, dtype=float)


# 1) Normal
def norm_pdf(x, loc=0.0, scale=1.0):
    if scale <= 0:
        raise ValueError("scale must be positive.")
    x = _asarray(x)
    z = (x - loc) / scale
    return (1.0 / (scale * np.sqrt(2*np.pi))) * np.exp(-0.5 * z**2)


# 2) Exponential
def expon_pdf(x, loc=0.0, scale=1.0):
    if scale <= 0:
        raise ValueError("scale must be positive.")
    x = _asarray(x)
    z = (x - loc) / scale
    out = np.zeros_like(x)
    mask = x >= loc
    out[mask] = (1.0 / scale) * np.exp(-z[mask])
    return out


# 3) Student t
def t_pdf(x, df, loc=0.0, scale=1.0):
    if df <= 0 or scale <= 0:
        raise ValueError("df and scale must be positive.")
    x = _asarray(x)
    y = (x - loc) / scale
    c = gamma((df+1)/2) / (np.sqrt(df*np.pi) * gamma(df/2))
    return (1.0/scale) * c * (1 + (y**2)/df)**(-(df+1)/2)


# 4) Weibull min
def weibull_min_pdf(x, c, loc=0.0, scale=1.0):
    if c <= 0 or scale <= 0:
        raise ValueError("c and scale must be positive.")
    x = _asarray(x)
    y = (x - loc) / scale
    out = np.zeros_like(x)
    mask = x >= loc
    out[mask] = (c/scale) * (y[mask]**(c-1)) * np.exp(-(y[mask]**c))
    return out


# 5) Gamma
def gamma_pdf(x, a, loc=0.0, scale=1.0):
    if a <= 0 or scale <= 0:
        raise ValueError("a and scale must be positive.")
    x = _asarray(x)
    t = x - loc
    out = np.zeros_like(x)
    mask = x >= loc
    out[mask] = (
        (t[mask]**(a-1)) *
        np.exp(-t[mask]/scale) /
        (gamma(a) * scale**a)
    )
    return out


# 6) Weibull max
def weibull_max_pdf(x, c, loc=0.0, scale=1.0):
    if c <= 0 or scale <= 0:
        raise ValueError("c and scale must be positive.")
    x = _asarray(x)
    y = (loc - x) / scale
    out = np.zeros_like(x)
    mask = x <= loc
    out[mask] = (c/scale) * (y[mask]**(c-1)) * np.exp(-(y[mask]**c))
    return out


# 7) Beta
def beta_pdf(x, a, b, loc=0.0, scale=1.0):
    if a <= 0 or b <= 0 or scale <= 0:
        raise ValueError("invalid parameters.")
    x = _asarray(x)
    y = (x - loc) / scale
    out = np.zeros_like(x)
    mask = (x >= loc) & (x <= loc+scale)
    B = gamma(a)*gamma(b)/gamma(a+b)
    out[mask] = (1/(scale*B)) * (y[mask]**(a-1)) * ((1-y[mask])**(b-1))
    return out


# 8) Lognormal
def lognorm_pdf(x, s, loc=0.0, scale=1.0):
    if s <= 0 or scale <= 0:
        raise ValueError("invalid parameters.")
    x = _asarray(x)
    out = np.zeros_like(x)
    mask = x > loc
    z = np.log((x[mask] - loc)/scale)
    out[mask] = (
        1 / ((x[mask]-loc)*s*np.sqrt(2*np.pi))
        * np.exp(-(z**2)/(2*s*s))
    )
    return out


# 9) Chi
def chi_pdf(x, df, loc=0.0, scale=1.0):
    if df <= 0 or scale <= 0:
        raise ValueError("invalid parameters.")
    x = _asarray(x)
    y = (x - loc) / scale
    out = np.zeros_like(x)
    mask = x >= loc
    c = (2**(1-df/2)) / gamma(df/2)
    out[mask] = (1/scale)*c*(y[mask]**(df-1))*np.exp(-(y[mask]**2)/2)
    return out


# 10) Chi2
def chi2_pdf(x, df, loc=0.0, scale=1.0):
    if df <= 0 or scale <= 0:
        raise ValueError("invalid parameters.")
    x = _asarray(x)
    y = (x - loc) / scale
    out = np.zeros_like(x)
    mask = x >= loc
    c = 1/(2**(df/2)*gamma(df/2))
    out[mask] = (1/scale)*c*(y[mask]**(df/2 - 1))*np.exp(-y[mask]/2)
    return out


# 11) Rayleigh
def rayleigh_pdf(x, loc=0.0, scale=1.0):
    if scale <= 0:
        raise ValueError("scale must be positive.")
    x = _asarray(x)
    z = x - loc
    out = np.zeros_like(x)
    mask = x >= loc
    out[mask] = (z[mask]/scale**2) * np.exp(-0.5*(z[mask]**2)/scale**2)
    return out


# 12) Pareto
def pareto_pdf(x, b, loc=0.0, scale=1.0):
    if b <= 0 or scale <= 0:
        raise ValueError("invalid parameters.")
    x = _asarray(x)
    z = (x - loc) / scale
    out = np.zeros_like(x)
    mask = x >= loc + scale
    out[mask] = (1/scale) * b * (z[mask]**(-b-1))
    return out


# 13) Cauchy
def cauchy_pdf(x, loc=0.0, scale=1.0):
    if scale <= 0:
        raise ValueError("scale must be positive.")
    x = _asarray(x)
    z = (x - loc) / scale
    return 1/(np.pi*scale*(1+z**2))


# 14) Laplace
def laplace_pdf(x, loc=0.0, scale=1.0):
    if scale <= 0:
        raise ValueError("scale must be positive.")
    x = _asarray(x)
    return (1/(2*scale))*np.exp(-np.abs(x-loc)/scale)


# 15) F
def f_pdf(x, dfn, dfd, loc=0.0, scale=1.0):
    if dfn <= 0 or dfd <= 0 or scale <= 0:
        raise ValueError("invalid parameters.")
    x = _asarray(x)
    y = (x - loc) / scale
    out = np.zeros_like(x)
    mask = x >= loc

    beta_val = gamma(dfn/2)*gamma(dfd/2)/gamma((dfn+dfd)/2)
    c = (1/beta_val)*(dfn/dfd)**(dfn/2)

    out[mask] = (1/scale)*c*(y[mask]**(dfn/2 - 1)) * \
                (1 + (dfn/dfd)*y[mask])**(-(dfn+dfd)/2)

    return out


def builtin_density_pdf(density_dict, x):
    """
    Calculate the PDF of a builtin distribution.

    Dispatches to the built-in replacements for scipy.stats functions.
    This version does NOT provide defaults:
    if a required param is missing in density_dict["params"],
    a KeyError will be raised immediately.

    Example of density_dict:
        {
          "name": "norm",
          "params": {
             "mu": 0.0,
             "sigma": 1.0
          }
        }
    """

    bname = density_dict["name"]
    bparams = density_dict["params"]

    if bname == "norm":
        mu = bparams['loc']         # one must exist
        sigma = bparams['scale']  # one must exist
        return norm_pdf(x, loc=mu, scale=sigma)

    elif bname == "lognorm":
        s = bparams["s"]  # must exist
        loc = bparams["loc"]  # must exist
        scale = bparams["scale"]  # must exist
        return lognorm_pdf(x, s, loc, scale)

    elif bname == "expon":
        loc = bparams["loc"]
        scale = bparams["scale"]
        return expon_pdf(x, loc, scale)

    elif bname == "t":
        df = bparams["df"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return t_pdf(x, df, loc, scale)

    elif bname == "weibull_min":
        c = bparams["c"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return weibull_min_pdf(x, c, loc, scale)

    elif bname == "gamma":
        a = bparams["a"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return gamma_pdf(x, a, loc, scale)

    elif bname == "weibull_max":
        c = bparams["c"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return weibull_max_pdf(x, c, loc, scale)

    elif bname == "beta":
        a = bparams["a"]
        b_ = bparams["b"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return beta_pdf(x, a, b_, loc, scale)

    elif bname == "chi":
        df = bparams["df"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return chi_pdf(x, df, loc, scale)

    elif bname == "chi2":
        df = bparams["df"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return chi2_pdf(x, df, loc, scale)

    elif bname == "rayleigh":
        loc = bparams["loc"]
        scale = bparams["scale"]
        return rayleigh_pdf(x, loc, scale)

    elif bname == "pareto":
        b_ = bparams["b"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return pareto_pdf(x, b_, loc, scale)

    elif bname == "cauchy":
        loc_ = bparams["loc"]
        scale_ = bparams["scale"]
        return cauchy_pdf(x, loc_, scale_)

    elif bname == "laplace":
        loc_ = bparams["loc"]
        scale_ = bparams["scale"]
        return laplace_pdf(x, loc_, scale_)

    elif bname == "f":
        dfn = bparams["dfn"]
        dfd = bparams["dfd"]
        loc = bparams["loc"]
        scale = bparams["scale"]
        return f_pdf(x, dfn, dfd, loc, scale)

    else:
        raise NotImplementedError(f"Unsupported builtin distribution '{bname}'")



def density_pdf_vectorized(density_dict, xs):
    xs = np.asarray(xs, dtype=float)
    dist_type = density_dict.get("type")

    # 1) Mixture
    if dist_type == "mixture":
        components = density_dict["components"]
        total_pdf = np.zeros_like(xs, dtype=float)
        weights_sum = 0.0

        for comp in components:
            weight = abs(comp["weight"])
            weights_sum += weight
            comp_pdf = density_pdf_vectorized(comp["density"], xs)
            total_pdf += weight * comp_pdf

        if weights_sum == 0:
            return total_pdf
        return total_pdf / weights_sum

    # 2) Scipy distribution
    elif dist_type == "scipy":
        dist_name = density_dict["name"]
        params = density_dict["params"]

        dist_class = getattr(st, dist_name)
        dist_obj = dist_class(**params)

        return dist_obj.pdf(xs)

    # 3) Statistics normal
    elif dist_type == "statistics":
        bname = density_dict["name"]
        bparams = density_dict["params"]

        if bname == "normal":
            mu = bparams.get("mu", bparams.get("loc", 0.0))
            sigma = bparams.get("sigma", bparams.get("scale", 1.0))

            # manual vectorized normal pdf (faster than NormalDist loop)
            coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
            return coeff * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)

        raise NotImplementedError(f"Unsupported statistics dist '{bname}'")

    # 4) Builtin
    elif dist_type == "builtin":
        return builtin_density_pdf(density_dict, xs)

    else:
        raise ValueError(f"Unknown density type: {dist_type}")