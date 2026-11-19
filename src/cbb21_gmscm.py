"""Stationary and non-stationary ground-motion SCM formulations and diagnostics."""

# modules
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from cbb21_cossim import smooth

# helpers
def _fisher_z_from_corr(p, eps=1e-12):
    """
    Compute Fisher-Z transform while avoiding infinities at p = -1 or p = 1.
    """
    p_safe = np.clip(np.asarray(p, dtype=float), -1.0 + eps, 1.0 - eps)
    return 0.5 * np.log((1.0 + p_safe) / (1.0 - p_safe))


def _make_plot_distances(dh, n=200):
    """
    Create plotting distances from either a scalar maximum distance or an
    array of distances.

    If an array is supplied, finite values are sorted before plotting.
    """
    dh_arr = np.asarray(dh, dtype=float)

    if dh_arr.ndim == 0:
        xmax = float(dh_arr)
        x = np.linspace(0.0, xmax, n)
    else:
        x = dh_arr[np.isfinite(dh_arr)]
        if x.size == 0:
            raise ValueError("dh must contain at least one finite distance.")
        x = np.sort(x)
        xmax = float(np.nanmax(x))

    return x, xmax

def _summarize_corr_deviation(
    e,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Summarize correlation deviation values.

    Parameters
    ----------
    e : array-like
        Correlation deviation values, usually computed as

            e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

        where z_p_hat is the Fisher-transformed empirical correlation,
        z_p_model is the Fisher-transformed model correlation, and n_j is
        the number of common records used to estimate the empirical
        correlation for each station pair.

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan. This preserves the existing
        return structure while avoiding unnecessary or unjustified bootstrap
        estimates.

    confidence_level : float, default 0.90
        Bootstrap confidence level. The default value of 0.90 corresponds
        approximately to a 5th to 95th percentile confidence interval.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    e_mean : float
        Mean of the finite correlation deviation values.

    e_std : float
        Standard deviation of the finite correlation deviation values.

    e_5 : float
        Lower confidence interval bound for the mean correlation deviation.
        Returned as np.nan when return_ci=False.

    e_95 : float
        Upper confidence interval bound for the mean correlation deviation.
        Returned as np.nan when return_ci=False.

    estd_5 : float
        Lower confidence interval bound for the standard deviation of the
        correlation deviation. Returned as np.nan when return_ci=False.

    estd_95 : float
        Upper confidence interval bound for the standard deviation of the
        correlation deviation. Returned as np.nan when return_ci=False.
    """

    e = np.asarray(e, dtype=float)
    e_valid = e[np.isfinite(e)]

    if e_valid.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    e_mean = np.nanmean(e_valid)
    e_std = np.nanstd(e_valid)

    if (not return_ci) or (e_valid.size < 2):
        return e_mean, e_std, np.nan, np.nan, np.nan, np.nan

    data = (e_valid,)

    res_mean = bootstrap(
        data,
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
        method="percentile",
    )

    res_std = bootstrap(
        data,
        np.std,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
        method="percentile",
    )

    e_5 = res_mean.confidence_interval.low
    e_95 = res_mean.confidence_interval.high
    estd_5 = res_std.confidence_interval.low
    estd_95 = res_std.confidence_interval.high

    return e_mean, e_std, e_5, e_95, estd_5, estd_95

# Goda and Hong 2008 GM SCM
def GMSCM_GH08(
    T,
    dh,
    n_rec,
    cbb2021corr,
    plot=False,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Evaluate the Goda and Hong (2008) ground-motion spatial correlation model.

    This function computes model-predicted spatial correlation coefficients
    for the separation distances in `cbb2021corr`, compares them against
    empirical correlation estimates, and summarizes the resulting correlation
    deviation values.

    The correlation deviation is computed in Fisher-Z space as

        e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

    where z_p_hat is the Fisher-transformed empirical correlation estimate,
    z_p_model is the Fisher-transformed model prediction, and n_j is the
    number of common records used to estimate the empirical correlation for
    each station pair.

    Parameters
    ----------
    T : float
        Intensity-measure period in seconds. The Goda and Hong (2008) model
        is available for T = 0.01 s and for 0.1 <= T <= 3.0 s.

    dh : float or array-like
        Maximum separation distance, in kilometres, used for plotting the
        reference correlation curve. This is only used when plot=True.

    n_rec : int
        Number of records used to define the plotting uncertainty band around
        the model correlation curve. This is only used when plot=True.

    cbb2021corr : object or pandas.DataFrame-like
        Empirical correlation data. Must contain the following attributes or
        columns:

        h : array-like
            Separation distance between station pairs, in kilometres.

        p_hat : array-like
            Empirical correlation estimate for each station pair.

        z_p_hat : array-like
            Fisher-Z transform of the empirical correlation estimate.

        n_j : array-like
            Number of common records used to estimate each empirical
            correlation value.

    plot : bool, default False
        If True, plot empirical correlations against separation distance,
        coloured by correlation deviation, with the Goda and Hong (2008)
        model curve shown for reference.

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan. This is useful when
        bootstrapping is not warranted or when only the point summaries are
        required.

    confidence_level : float, default 0.90
        Bootstrap confidence level used when return_ci=True. The default
        value of 0.90 corresponds approximately to a 5th to 95th percentile
        confidence interval.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    p : ndarray
        Model-predicted correlation coefficient for each separation distance
        in `cbb2021corr.h`.

    e : ndarray
        Correlation deviation values in Fisher-Z space.

    e_mean : float
        Mean correlation deviation.

    e_std : float
        Standard deviation of the correlation deviation.

    e_5 : float
        Lower confidence interval bound for the mean correlation deviation.
        Returned as np.nan when return_ci=False.

    e_95 : float
        Upper confidence interval bound for the mean correlation deviation.
        Returned as np.nan when return_ci=False.

    estd_5 : float
        Lower confidence interval bound for the standard deviation of the
        correlation deviation. Returned as np.nan when return_ci=False.

    estd_95 : float
        Upper confidence interval bound for the standard deviation of the
        correlation deviation. Returned as np.nan when return_ci=False.
    """

    if 0.1 <= T <= 3:
        alpha = -0.16 * np.log(T) + 0.68
        beta = 0.44

    elif T == 0.01:
        alpha = 0.93
        beta = 0.49

    else:
        print("Goda and Hong (2008) not available for pSA = %0.2fs" % T)

        p = np.full_like(cbb2021corr.h, np.nan, dtype=float)
        e = np.full_like(cbb2021corr.z_p_hat, np.nan, dtype=float)

        return p, e, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # model correlation at empirical pair distances
    p = np.exp(-alpha * cbb2021corr.h**beta)

    # Fisher-Z transform of model correlation
    z_p = _fisher_z_from_corr(p)

    # correlation deviation
    e = (cbb2021corr.z_p_hat - z_p) * np.sqrt(cbb2021corr.n_j - 2)

    # summary statistics, with optional bootstrap CIs
    e_mean, e_std, e_5, e_95, estd_5, estd_95 = _summarize_corr_deviation(
        e,
        return_ci=return_ci,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    if T == 0:
        plot_str = "PGA"
    else:
        plot_str = "pSA = %0.1fs" % T

    if plot:
        x, xmax = _make_plot_distances(dh, n=100)
        p_dh = np.exp(-alpha * x ** beta)
        p_std = np.square(1.0 - p_dh) / np.sqrt(n_rec)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150, constrained_layout=True)

        im = ax.scatter(
            cbb2021corr.h,
            cbb2021corr.p_hat,
            c=e,
            vmin=-4,
            vmax=4,
            cmap="bwr",
            edgecolors="black",
            s=60,
        )

        ax.plot(x, p_dh, "-k")
        ax.plot(x, p_dh - p_std, "--k")
        ax.plot(x, p_dh + p_std, "--k")

        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)
        ax.set_title("Goda and Hong (2008): %s" % plot_str)
        ax.set_ylabel(r"Correlation, $\hat{p}$ (%s)" % plot_str)
        ax.set_xlabel("Separation Distance, h (km)")

        textstr = "\n".join(
            (
                rf"$\mu_e={e_mean:.2f}$",
                rf"$\sigma_e={e_std:.2f}$",
            )
        )

        ax.text(
            0.5,
            -0.95,
            textstr,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax.legend([r'$\hat{p}_{j,k}$', r'$p(h)$', r'$p(h) \pm \sigma$'], loc='upper right')
        fig.colorbar(im, ax=ax, label="Correlation Deviation, e", aspect=20)
        plt.show()

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95


# Jayaram and Baker (2009) GM SCM
def GMSCM_JB09(
    case,
    T,
    dh,
    n_rec,
    cbb2021corr,
    plot=False,
    plot_trend = False,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Evaluate the Jayaram and Baker (2009) ground-motion spatial correlation model.

    This function computes model-predicted spatial correlation coefficients
    for the separation distances in `cbb2021corr`, compares them with
    empirical correlation estimates, and summarizes the resulting correlation
    deviation values.

    The correlation deviation is computed in Fisher-Z space as

        e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

    where z_p_hat is the Fisher-transformed empirical correlation estimate,
    z_p_model is the Fisher-transformed model prediction, and n_j is the
    number of common records used to estimate the empirical correlation for
    each station pair.

    Parameters
    ----------
    case : int
        Jayaram and Baker (2009) model case.

        case = 1
            Spatial correlation model for ground motions without Vs30-based
            site correction.

        case = 2
            Spatial correlation model for ground motions with Vs30-based
            site correction.

    T : float
        Intensity-measure period in seconds. Use T = 0 for PGA if that is how
        PGA is represented in the surrounding workflow.

    dh : float or array-like
        Maximum separation distance, in kilometres, or array of separation
        distances used for plotting the reference correlation curve. This is
        only used when plot=True.

    n_rec : int
        Number of records used to define the plotting uncertainty band around
        the model correlation curve. This is only used when plot=True.

    cbb2021corr : object or pandas.DataFrame-like
        Empirical correlation data. Must contain the following attributes or
        columns:

        h : array-like
            Separation distance between station pairs, in kilometres.

        p_hat : array-like
            Empirical correlation estimate for each station pair.

        z_p_hat : array-like
            Fisher-Z transform of the empirical correlation estimate.

        n_j : array-like
            Number of common records used to estimate each empirical
            correlation value.

    plot : bool, default False
        If True, plot empirical correlations against separation distance,
        coloured by correlation deviation, with the Jayaram and Baker (2009)
        model curve shown for reference.

    plot_trend : bool, default False
        If True, plot LOWESS of correlation coefficients and +/- stdev

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan.

    confidence_level : float, default 0.90
        Bootstrap confidence level used when return_ci=True. The default
        value of 0.90 corresponds approximately to a 5th to 95th percentile
        confidence interval.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    p : ndarray
        Model-predicted correlation coefficient for each separation distance
        in `cbb2021corr.h`.

    e : ndarray
        Correlation deviation values in Fisher-Z space.

    e_mean : float
        Mean correlation deviation.

    e_std : float
        Standard deviation of the correlation deviation.

    e_5, e_95 : float
        Lower and upper confidence interval bounds for the mean correlation
        deviation. Returned as np.nan when return_ci=False.

    estd_5, estd_95 : float
        Lower and upper confidence interval bounds for the standard deviation
        of the correlation deviation. Returned as np.nan when return_ci=False.
    """

    if case not in (1, 2):
        raise ValueError("case must be 1 or 2 for Jayaram and Baker (2009).")

    if (T < 1) and (case == 1):
        b = 8.5 + 17.2 * T
    elif (T < 1) and (case == 2):
        b = 20.0
    else:
        b = 22.0 + 3.7 * T

    p = np.exp(-3.0 * cbb2021corr.h / b)

    z_p = _fisher_z_from_corr(p)

    e = (cbb2021corr.z_p_hat - z_p) * np.sqrt(cbb2021corr.n_j - 2)

    e_mean, e_std, e_5, e_95, estd_5, estd_95 = _summarize_corr_deviation(
        e,
        return_ci=return_ci,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    plot_str = "PGA" if T == 0 else "pSA = %0.2fs" % T

    if plot:
        x, xmax = _make_plot_distances(dh)
        p_dh = np.exp(-3.0 * x / b)
        p_std = np.square(1.0 - p_dh) / np.sqrt(n_rec)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200, constrained_layout=True)

        im = ax.scatter(
            cbb2021corr.h,
            cbb2021corr.p_hat,
            c=e,
            vmin=-4,
            vmax=4,
            cmap="bwr",
            edgecolors="black",
            s=60,
        )

        ax.plot(x, p_dh, "-k")
        ax.plot(x, p_dh - p_std, "--k")
        ax.plot(x, p_dh + p_std, "--k")

        if plot_trend:
            xgrid, _, mean, std = smooth(cbb2021corr.h, cbb2021corr.p_hat, K=200,show_progress=False)
            ax.plot(xgrid, mean, color="forestgreen", label="LOWESS mean")
            ax.fill_between(xgrid, mean - std,mean + std, color="forestgreen", label="LOWESS bootstrap ±1 SD",alpha=0.2)

        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(r"Correlation, $\hat{p}$ (%s)" % plot_str)
        ax.set_xlabel("Separation Distance, h (km)")

        textstr = "\n".join(
            (
                rf"$\mu_e={e_mean:.2f}$",
                rf"$\sigma_e={e_std:.2f}$",
            )
        )

        ax.text(
            0.5,
            -0.95,
            textstr,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax.legend([r'$\hat{p}_{j,k}$', r'$p(h)$', r'$p(h) \pm \sigma$'], loc='upper right')
        fig.colorbar(im, ax=ax, label="Correlation Deviation, e", aspect=20, pad = 0.02)
        plt.show()

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95

# Goda and Atkinson 2010
def GMSCM_GA10(
    T,
    dh,
    n_rec,
    cbb2021corr,
    plot=False,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Evaluate the Goda and Atkinson (2010) ground-motion spatial correlation model.

    This function computes model-predicted spatial correlation coefficients
    for the separation distances in `cbb2021corr`, compares them with
    empirical correlation estimates, and summarizes the resulting correlation
    deviation values.

    The correlation deviation is computed in Fisher-Z space as

        e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

    Parameters
    ----------
    T : float
        Intensity-measure period in seconds. The Goda and Atkinson (2010)
        model is available only for the tabulated periods:

            0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, and 5.0 s.

    dh : float or array-like
        Maximum separation distance, in kilometres, or array of separation
        distances used for plotting the reference correlation curve. This is
        only used when plot=True.

    n_rec : int
        Number of records used to define the plotting uncertainty band around
        the model correlation curve. This is only used when plot=True.

    cbb2021corr : object or pandas.DataFrame-like
        Empirical correlation data. Must contain h, p_hat, z_p_hat, and n_j.

    plot : bool, default False
        If True, plot empirical correlations against separation distance,
        coloured by correlation deviation, with the Goda and Atkinson (2010)
        model curve shown for reference.

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan.

    confidence_level : float, default 0.90
        Bootstrap confidence level used when return_ci=True.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    p : ndarray
        Model-predicted correlation coefficient for each separation distance
        in `cbb2021corr.h`.

    e : ndarray
        Correlation deviation values in Fisher-Z space.

    e_mean, e_std : float
        Mean and standard deviation of the correlation deviation.

    e_5, e_95 : float
        Lower and upper confidence interval bounds for the mean correlation
        deviation. Returned as np.nan when return_ci=False.

    estd_5, estd_95 : float
        Lower and upper confidence interval bounds for the standard deviation
        of the correlation deviation. Returned as np.nan when return_ci=False.
    """

    period = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    alpha = np.array([0.060, 0.062, 0.073, 0.086, 0.073, 0.051, 0.061, 0.092, 0.071])
    beta = np.array([0.283, 0.276, 0.248, 0.219, 0.248, 0.329, 0.421, 0.671, 0.741])
    gamma = np.array([5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 3.035, 1.189, 1.201])

    locs = np.flatnonzero(np.isclose(T, period))

    if locs.size == 0:
        print("Goda and Atkinson (2010) not available for pSA = %0.2fs" % T)

        p = np.full_like(cbb2021corr.h, np.nan, dtype=float)
        e = np.full_like(cbb2021corr.z_p_hat, np.nan, dtype=float)

        return p, e, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    loc = locs[0]

    p = gamma[loc] * np.exp(-alpha[loc] * cbb2021corr.h ** beta[loc]) - gamma[loc] + 1.0

    z_p = _fisher_z_from_corr(p)

    e = (cbb2021corr.z_p_hat - z_p) * np.sqrt(cbb2021corr.n_j - 2)

    e_mean, e_std, e_5, e_95, estd_5, estd_95 = _summarize_corr_deviation(
        e,
        return_ci=return_ci,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    plot_str = "PGA" if T == 0 else "pSA = %0.1fs" % T

    if plot:
        x, xmax = _make_plot_distances(dh)
        p_dh = gamma[loc] * np.exp(-alpha[loc] * x ** beta[loc]) - gamma[loc] + 1.0
        p_std = np.square(1.0 - p_dh) / np.sqrt(n_rec)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150, constrained_layout=True)

        im = ax.scatter(
            cbb2021corr.h,
            cbb2021corr.p_hat,
            c=e,
            vmin=-4,
            vmax=4,
            cmap="bwr",
            edgecolors="black",
            s=60,
        )

        ax.plot(x, p_dh, "-k")
        ax.plot(x, p_dh - p_std, "--k")
        ax.plot(x, p_dh + p_std, "--k")

        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)
        ax.set_title("Goda and Atkinson (2010): %s" % plot_str)
        ax.set_ylabel(r"Correlation, $\hat{p}$ (%s)" % plot_str)
        ax.set_xlabel("Separation Distance, h (km)")

        textstr = "\n".join(
            (
                rf"$\mu_e={e_mean:.2f}$",
                rf"$\sigma_e={e_std:.2f}$",
            )
        )

        ax.text(
            0.5,
            -0.95,
            textstr,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax.legend([r'$\hat{p}_{j,k}$', r'$p(h)$', r'$p(h) \pm \sigma$'], loc='upper right')
        fig.colorbar(im, ax=ax, label="Correlation Deviation, e", aspect=20)
        plt.show()

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95

# Esposito and Iervolino 2012
def GMSCM_EI12(
    T,
    dh,
    n_rec,
    cbb2021corr,
    plot=False,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Evaluate the Esposito and Iervolino (2012) ground-motion spatial correlation model.

    This function computes model-predicted spatial correlation coefficients
    for the separation distances in `cbb2021corr`, compares them with
    empirical correlation estimates, and summarizes the resulting correlation
    deviation values.

    The correlation deviation is computed in Fisher-Z space as

        e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

    Parameters
    ----------
    T : float
        Intensity-measure period in seconds. Use T = 0 for PGA if that is how
        PGA is represented in the surrounding workflow.

    dh : float or array-like
        Maximum separation distance, in kilometres, or array of separation
        distances used for plotting the reference correlation curve. This is
        only used when plot=True.

    n_rec : int
        Number of records used to define the plotting uncertainty band around
        the model correlation curve. This is only used when plot=True.

    cbb2021corr : object or pandas.DataFrame-like
        Empirical correlation data. Must contain h, p_hat, z_p_hat, and n_j.

    plot : bool, default False
        If True, plot empirical correlations against separation distance,
        coloured by correlation deviation, with the Esposito and Iervolino
        (2012) model curve shown for reference.

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan.

    confidence_level : float, default 0.90
        Bootstrap confidence level used when return_ci=True.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    p : ndarray
        Model-predicted correlation coefficient for each separation distance
        in `cbb2021corr.h`.

    e : ndarray
        Correlation deviation values in Fisher-Z space.

    e_mean, e_std : float
        Mean and standard deviation of the correlation deviation.

    e_5, e_95 : float
        Lower and upper confidence interval bounds for the mean correlation
        deviation. Returned as np.nan when return_ci=False.

    estd_5, estd_95 : float
        Lower and upper confidence interval bounds for the standard deviation
        of the correlation deviation. Returned as np.nan when return_ci=False.
    """

    b = 8.6 + 11.6 * T

    p = np.exp(-3.0 * cbb2021corr.h / b)

    z_p = _fisher_z_from_corr(p)

    e = (cbb2021corr.z_p_hat - z_p) * np.sqrt(cbb2021corr.n_j - 2)

    e_mean, e_std, e_5, e_95, estd_5, estd_95 = _summarize_corr_deviation(
        e,
        return_ci=return_ci,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    plot_str = "PGA" if T == 0 else "pSA = %0.2fs" % T

    if plot:
        x, xmax = _make_plot_distances(dh)
        p_dh = np.exp(-3.0 * x / b)
        p_std = np.square(1.0 - p_dh) / np.sqrt(n_rec)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150, constrained_layout=True)

        im = ax.scatter(
            cbb2021corr.h,
            cbb2021corr.p_hat,
            c=e,
            vmin=-4,
            vmax=4,
            cmap="bwr",
            edgecolors="black",
            s=60,
        )

        ax.plot(x, p_dh, "-k")
        ax.plot(x, p_dh - p_std, "--k")
        ax.plot(x, p_dh + p_std, "--k")

        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)
        ax.set_title("Esposito and Iervolino (2012): %s" % plot_str)
        ax.set_ylabel(r"Correlation, $\hat{p}$ (%s)" % plot_str)
        ax.set_xlabel("Separation Distance, h (km)")

        textstr = "\n".join(
            (
                rf"$\mu_e={e_mean:.2f}$",
                rf"$\sigma_e={e_std:.2f}$",
            )
        )

        ax.text(
            0.5,
            -0.95,
            textstr,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax.legend([r'$\hat{p}_{j,k}$', r'$p(h)$', r'$p(h) \pm \sigma$'], loc='upper right')
        fig.colorbar(im, ax=ax, label="Correlation Deviation, e", aspect=20)
        plt.show()

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95

# Loth and Baker 2013
# Short range coregionalization matrix, B1
B1 = np.array(
    [
        [0.29, 0.25, 0.23, 0.23, 0.18, 0.1, 0.06, 0.06, 0.06],
        [0.25, 0.30, 0.2, 0.16, 0.1, 0.04, 0.03, 0.04, 0.05],
        [0.23, 0.20, 0.27, 0.18, 0.1, 0.03, 0, 0.01, 0.02],
        [0.23, 0.16, 0.18, 0.31, 0.22, 0.14, 0.08, 0.07, 0.07],
        [0.18, 0.10, 0.1, 0.22, 0.33, 0.24, 0.16, 0.13, 0.12],
        [0.10, 0.04, 0.03, 0.14, 0.24, 0.33, 0.26, 0.21, 0.19],
        [0.06, 0.03, 0, 0.08, 0.16, 0.26, 0.37, 0.3, 0.26],
        [0.06, 0.04, 0.01, 0.07, 0.13, 0.21, 0.3, 0.28, 0.24],
        [0.06, 0.05, 0.02, 0.07, 0.12, 0.19, 0.26, 0.24, 0.23],
    ]
)

# Long range coregionalization matrix, B2
B2 = np.array(
    [
        [0.47, 0.4, 0.43, 0.35, 0.27, 0.15, 0.13, 0.09, 0.12],
        [0.4, 0.42, 0.37, 0.25, 0.15, 0.03, 0.04, 0, 0.03],
        [0.43, 0.37, 0.45, 0.36, 0.26, 0.15, 0.09, 0.05, 0.08],
        [0.35, 0.25, 0.36, 0.42, 0.37, 0.29, 0.2, 0.16, 0.16],
        [0.27, 0.15, 0.26, 0.37, 0.48, 0.41, 0.26, 0.21, 0.21],
        [0.15, 0.03, 0.15, 0.29, 0.41, 0.55, 0.37, 0.33, 0.32],
        [0.13, 0.04, 0.09, 0.2, 0.26, 0.37, 0.51, 0.49, 0.49],
        [0.09, 0.00, 0.05, 0.16, 0.21, 0.33, 0.49, 0.62, 0.6],
        [0.12, 0.03, 0.08, 0.16, 0.21, 0.32, 0.49, 0.6, 0.68],
    ]
)

# Nugget effect coregionalization matrix, B3
B3 = np.array(
    [
        [
            0.240000000000000,
            0.219983028675722,
            0.209991239369580,
            0.0899940658151642,
            -0.0199982490874490,
            0.0100004273375877,
            0.0299729607606612,
            0.0200291990885140,
            0.00995702711846606,
        ],
        [
            0.219983028675722,
            0.280000000000000,
            0.199999710563431,
            0.0400020556476041,
            -0.0500003168664929,
            -5.02841885169300e-07,
            0.0100141900421009,
            0.00994747690890486,
            -0.00996663511790072,
        ],
        [
            0.209991239369580,
            0.199999710563431,
            0.280000000000000,
            0.0500007637487926,
            -0.0600002196848805,
            -1.80663938055364e-07,
            0.0399992445541918,
            0.0299487635912810,
            0.0100035235224244,
        ],
        [
            0.0899940658151642,
            0.0400020556476041,
            0.0500007637487926,
            0.270000000000000,
            0.139999321454879,
            0.0499996979574019,
            0.0499981807238188,
            0.0499227563681531,
            0.0399858842409999,
        ],
        [
            -0.0199982490874490,
            -0.0500003168664929,
            -0.0600002196848805,
            0.139999321454879,
            0.190000000000000,
            0.0700000354290215,
            0.0499897414826383,
            0.0499443288879162,
            0.0499652709189241,
        ],
        [
            0.0100004273375877,
            -5.02841885169300e-07,
            -1.80663938055364e-07,
            0.0499996979574019,
            0.0700000354290215,
            0.120000000000000,
            0.0799859494118349,
            0.0699172702775962,
            0.0599608152312721,
        ],
        [
            0.0299729607606612,
            0.0100141900421009,
            0.0399992445541918,
            0.0499981807238188,
            0.0499897414826383,
            0.0799859494118349,
            0.120000000000000,
            0.0997643834727755,
            0.0800031285024676,
        ],
        [
            0.0200291990885140,
            0.00994747690890486,
            0.0299487635912810,
            0.0499227563681531,
            0.0499443288879162,
            0.0699172702775962,
            0.0997643834727755,
            0.100000000000000,
            0.0896690207890228,
        ],
        [
            0.00995702711846606,
            -0.00996663511790072,
            0.0100035235224244,
            0.0399858842409999,
            0.0499652709189241,
            0.0599608152312721,
            0.0800031285024676,
            0.0896690207890228,
            0.0900000000000000,
        ],
    ]
)

T_list = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0001])

B_coeff1_interpolator = RegularGridInterpolator((T_list, T_list), B1, bounds_error=True, method="linear")
B_coeff2_interpolator = RegularGridInterpolator((T_list, T_list), B2, bounds_error=True, method="linear")
B_coeff3_interpolator = RegularGridInterpolator((T_list, T_list), B3, bounds_error=True, method="linear")

def _interpolate_B_coefficients(T1: float, T2: float, index_1: int, index_2: int, B: np.ndarray):
    """Diagonal-preserving interpolation for Loth & Baker matrices."""
    T1_vals = T_list[index_1:index_1 + 2]
    T2_vals = T_list[index_2:index_2 + 2]
    B_vals = B[index_1:index_1 + 2, index_2:index_2 + 2]

    T_avg_targ = np.mean([T1, T2])
    T_avg_vals = np.mean([T1_vals, T2_vals], axis=0)
    B_avg_vals = np.diagonal(B_vals)
    B_diag_val = np.interp(T_avg_targ, T_avg_vals, B_avg_vals)

    T_diff_targ = np.abs(T1 - T2)
    T_diff_vals = [0, np.abs(np.min(T1_vals) - np.max(T2_vals))]
    B_diff_vals = [B_diag_val, B_vals[0, 1]]
    return np.interp(T_diff_targ, T_diff_vals, B_diff_vals)

def GMSCM_LB13(
    T1,
    T2,
    dh,
    n_rec,
    corrdata,
    plot=False,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Evaluate the Loth and Baker (2013) ground-motion spatial correlation model.

    This function computes model-predicted spatial correlation coefficients
    for the separation distances in `corrdata`, compares them with empirical
    correlation estimates, and summarizes the resulting correlation deviation
    values.

    The model supports both same-period correlations, where T1 == T2, and
    cross-period correlations, where T1 != T2.

    The correlation deviation is computed in Fisher-Z space as

        e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

    Parameters
    ----------
    T1, T2 : float
        Intensity-measure periods in seconds. Periods must be between 0.01
        and 10.0 s.

    dh : float or array-like
        Maximum separation distance, in kilometres, or array of separation
        distances used for plotting the reference correlation curve. This is
        only used when plot=True.

    n_rec : int
        Number of records used to define the plotting uncertainty band around
        the model correlation curve. This is only used when plot=True.

    corrdata : object or pandas.DataFrame-like
        Empirical correlation data. Must contain h, p_hat, z_p_hat, and n_j.

    plot : bool, default False
        If True, plot empirical correlations against separation distance,
        coloured by correlation deviation, with the Loth and Baker (2013)
        model curve shown for reference.

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan.

    confidence_level : float, default 0.90
        Bootstrap confidence level used when return_ci=True.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    p : ndarray
        Model-predicted correlation coefficient for each separation distance
        in `corrdata.h`.

    e : ndarray
        Correlation deviation values in Fisher-Z space.

    e_mean, e_std : float
        Mean and standard deviation of the correlation deviation.

    e_5, e_95 : float
        Lower and upper confidence interval bounds for the mean correlation
        deviation. Returned as np.nan when return_ci=False.

    estd_5, estd_95 : float
        Lower and upper confidence interval bounds for the standard deviation
        of the correlation deviation. Returned as np.nan when return_ci=False.
    """

    if min(T1, T2) < 0.01 or max(T1, T2) > 10.0:
        raise ValueError("Periods must be between 0.01 and 10.0 seconds")

    index_1 = np.flatnonzero(T_list <= T1)[-1]
    index_2 = np.flatnonzero(T_list <= T2)[-1]

    if index_1 == index_2:
        B1c = _interpolate_B_coefficients(T1, T2, index_1, index_2, B1)
        B2c = _interpolate_B_coefficients(T1, T2, index_1, index_2, B2)
        B3c = _interpolate_B_coefficients(T1, T2, index_1, index_2, B3)
    else:
        B1c = float(np.asarray(B_coeff1_interpolator([T1, T2])).squeeze())
        B2c = float(np.asarray(B_coeff2_interpolator([T1, T2])).squeeze())
        B3c = float(np.asarray(B_coeff3_interpolator([T1, T2])).squeeze())

    p = (
        B1c * np.exp(-3.0 * corrdata.h / 20.0)
        + B2c * np.exp(-3.0 * corrdata.h / 70.0)
        + np.where(np.isclose(corrdata.h, 0.0), B3c, 0.0)
    )

    z_p = _fisher_z_from_corr(p)

    e = (corrdata.z_p_hat - z_p) * np.sqrt(corrdata.n_j - 2)

    e_mean, e_std, e_5, e_95, estd_5, estd_95 = _summarize_corr_deviation(
        e,
        return_ci=return_ci,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    if plot:
        if np.isclose(T1, T2):
            plot_str = "PGA" if T1 == 0 else f"pSA = {T1:.2f}s"
        else:
            plot_str = f"pSA = {T1:.2f}s vs {T2:.2f}s"

        x, xmax = _make_plot_distances(dh)

        p_dh = (
                B1c * np.exp(-3.0 * x / 20.0)
                + B2c * np.exp(-3.0 * x / 70.0)
                + np.where(np.isclose(x, 0.0), B3c, 0.0)
        )

        p_std = np.square(1.0 - p_dh) / np.sqrt(n_rec)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150, constrained_layout=True)

        im = ax.scatter(
            corrdata.h,
            corrdata.p_hat,
            c=e,
            vmin=-4,
            vmax=4,
            cmap="bwr",
            edgecolors="black",
            s=60,
        )

        ax.plot(x, p_dh, "-k", label="LB13 mean corr")
        ax.plot(x, p_dh - p_std, "--k", label="-1σ")
        ax.plot(x, p_dh + p_std, "--k", label="+1σ")

        ax.set_title(f"Loth and Baker (2013): {plot_str}")
        ax.set_ylabel(rf"Correlation, $\hat{{p}}$ ({plot_str})")
        ax.set_xlabel("Separation Distance, h (km)")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)

        textstr = "\n".join(
            (
                rf"$\mu_e={e_mean:.2f}$",
                rf"$\sigma_e={e_std:.2f}$",
            )
        )

        ax.text(
            0.5,
            -0.95,
            textstr,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax.legend([r'$\hat{p}_{j,k}$', r'$p(h)$', r'$p(h) \pm \sigma$'], loc='upper right')
        fig.colorbar(im, ax=ax, label="Correlation Deviation, e", aspect=20)
        plt.show(fig)

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95

# Heresi and Miranda 2019
def GMSCM_HM19(
    T,
    dh,
    n_rec,
    cbb2021corr,
    plot=False,
    return_ci=False,
    confidence_level=0.90,
    n_resamples=9999,
    random_state=None,
):
    """
    Evaluate the Heresi and Miranda (2019) ground-motion spatial correlation model.

    This function computes model-predicted spatial correlation coefficients
    for the separation distances in `cbb2021corr`, compares them with
    empirical correlation estimates, and summarizes the resulting correlation
    deviation values.

    The correlation deviation is computed in Fisher-Z space as

        e = (z_p_hat - z_p_model) * sqrt(n_j - 2)

    Parameters
    ----------
    T : float
        Intensity-measure period in seconds. Use T = 0 for PGA if that is how
        PGA is represented in the surrounding workflow.

    dh : float or array-like
        Maximum separation distance, in kilometres, or array of separation
        distances used for plotting the reference correlation curve. This is
        only used when plot=True.

    n_rec : int
        Number of records used to define the plotting uncertainty band around
        the model correlation curve. This is only used when plot=True.

    cbb2021corr : object or pandas.DataFrame-like
        Empirical correlation data. Must contain h, p_hat, z_p_hat, and n_j.

    plot : bool, default False
        If True, plot empirical correlations against separation distance,
        coloured by correlation deviation, with the Heresi and Miranda (2019)
        model curve shown for reference.

    return_ci : bool, default False
        If True, bootstrap confidence intervals are computed for the mean
        and standard deviation of the correlation deviation values.

        If False, no bootstrap calculation is performed and the confidence
        interval outputs are returned as np.nan.

    confidence_level : float, default 0.90
        Bootstrap confidence level used when return_ci=True.

    n_resamples : int, default 9999
        Number of bootstrap resamples used when return_ci=True.

    random_state : int, numpy.random.Generator, or None, default None
        Random seed or random generator passed to scipy.stats.bootstrap.

    Returns
    -------
    p : ndarray
        Model-predicted correlation coefficient for each separation distance
        in `cbb2021corr.h`.

    e : ndarray
        Correlation deviation values in Fisher-Z space.

    e_mean, e_std : float
        Mean and standard deviation of the correlation deviation.

    e_5, e_95 : float
        Lower and upper confidence interval bounds for the mean correlation
        deviation. Returned as np.nan when return_ci=False.

    estd_5, estd_95 : float
        Lower and upper confidence interval bounds for the standard deviation
        of the correlation deviation. Returned as np.nan when return_ci=False.
    """

    if T < 1.37:
        beta = 4.231 * T**2 - 5.180 * T + 13.392
    else:
        beta = 0.140 * T**2 - 2.249 * T + 17.050

    p = np.exp(-((cbb2021corr.h / beta) ** 0.55))

    z_p = _fisher_z_from_corr(p)

    e = (cbb2021corr.z_p_hat - z_p) * np.sqrt(cbb2021corr.n_j - 2)

    e_mean, e_std, e_5, e_95, estd_5, estd_95 = _summarize_corr_deviation(
        e,
        return_ci=return_ci,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    plot_str = "PGA" if T == 0 else "pSA = %0.2fs" % T

    if plot:
        x, xmax = _make_plot_distances(dh)
        p_dh = np.exp(-((x / beta) ** 0.55))
        p_std = np.square(1.0 - p_dh) / np.sqrt(n_rec)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150, constrained_layout=True)

        im = ax.scatter(
            cbb2021corr.h,
            cbb2021corr.p_hat,
            c=e,
            vmin=-4,
            vmax=4,
            cmap="bwr",
            edgecolors="black",
            s=60,
        )

        ax.plot(x, p_dh, "-k")
        ax.plot(x, p_dh - p_std, "--k")
        ax.plot(x, p_dh + p_std, "--k")

        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)
        ax.set_title("Heresi and Miranda (2019): %s" % plot_str)
        ax.set_ylabel(r"Correlation, $\hat{p}$ (%s)" % plot_str)
        ax.set_xlabel("Separation Distance, h (km)")

        textstr = "\n".join(
            (
                rf"$\mu_e={e_mean:.2f}$",
                rf"$\sigma_e={e_std:.2f}$",
            )
        )

        ax.text(
            0.5,
            -0.95,
            textstr,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax.legend([r'$\hat{p}_{j,k}$', r'$p(h)$', r'$p(h) \pm \sigma$'], loc='upper right')
        fig.colorbar(im, ax=ax, label="Correlation Deviation, e", aspect=20)
        plt.show(fig)

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95

# Bodenmann et al., (2023)
# https://github.com/bodlukas/ground-motion-correlation-bayes
period_BM23 = [0.01,0.03,0.06,0.1,0.3,0.6,1,3,6]
modelE_LE_BM23 = [10.5, 10.8, 10.4, 10.7, 10.3, 14.9, 16, 21.4, 37.3]
modelE_gE_BM23 = [0.43, 0.43, 0.43, 0.42, 0.39, 0.39, 0.4, 0.44, 0.45]

modelEA_LE_BM23 = [13.1, 13.5, 13.2, 13.8, 12.3, 19.1, 21.3, 29, 54.8]
modelEA_gE_BM23 = [0.35, 0.35, 0.34, 0.33, 0.33, 0.33, 0.35, 0.38, 0.39]
modelEA_LA_BM23 = [27.3, 28.2, 26.8, 24.8, 28.8, 27.6, 23.6, 25.1, 26.2]

modelEAS_LE_BM23 = [16.4, 16.9, 16.6, 16.3, 15.1, 25.6, 29.8, 42.1, 70.2]
modelEAS_gE_BM23 = [0.36, 0.36, 0.35, 0.34, 0.34, 0.37, 0.41, 0.46, 0.49]
modelEAS_LA_BM23 = [24.9, 25.6, 24.4, 23.3, 26.1, 24.2, 20.5, 18.5, 17.3]
modelEAS_LS_BM23 = [171.2, 185.6, 190.2, 189.8, 199.9, 222.8, 169.2, 358, 372.2]
modelEAS_w_BM23 = [0.84, 0.84, 0.84, 0.88, 0.85, 0.73, 0.7, 0.5, 0.54]

BM23_parameters = pd.DataFrame()
BM23_parameters['period'] = period_BM23
BM23_parameters['modelE_LE'] = modelE_LE_BM23
BM23_parameters['modelE_gammaE'] = modelE_gE_BM23
BM23_parameters['modelEA_LE'] = modelEA_LE_BM23
BM23_parameters['modelEA_gammaE'] = modelEA_gE_BM23
BM23_parameters['modelEA_LA'] = modelEA_LA_BM23
BM23_parameters['modelEAS_LE'] = modelEAS_LE_BM23
BM23_parameters['modelEAS_gammaE'] = modelEAS_gE_BM23
BM23_parameters['modelEAS_LA'] = modelEAS_LA_BM23
BM23_parameters['modelEAS_LS'] = modelEAS_LS_BM23
BM23_parameters['modelEAS_w'] = modelEAS_w_BM23

# Euclidean Distance Correlation Model
def getpE(dE,lE,gamma_E):
    pE = np.zeros((len(dE),1))
    pE[:,0] = np.exp(-(dE/lE)**gamma_E)
    return pE

# Angular Distance Correlation Model
def getpA(dA,lA):
    pA = (1+dA/lA)*(1-dA/180)**(180/lA)
    return pA

# Soil Dissimilarity Correlation Model
def getpS(dS,lS):
    pS = np.exp(-dS/lS)
    return pS

# combine and call model: EA
def MpEA(pE,pA):
    pEA = pE*pA
    return pEA

# combine and call model: EAS
def MpEAS(pE,pA,pS,w):
    pEAS = pE*(w*pA+(1-w)*pS)
    return pEAS

# main correlation function
def GMSCM_BM23(cbb21_pSAobs_T, period,parameters=None):

    if parameters is None:
        parameters = BM23_parameters

    results = {
        "E": {"e": [], "mu_e": [], "sig_e": [], "mu_e5": [], "mu_e95": [], "sig_e5": [], "sig_e95": []},
        "EA": {"e": [], "mu_e": [], "sig_e": [], "mu_e5": [], "mu_e95": [], "sig_e5": [], "sig_e95": []},
        "EAS": {"e": [], "mu_e": [], "sig_e": [], "mu_e5": [], "mu_e95": [], "sig_e5": [], "sig_e95": []}
    }

    for i, T in enumerate(period):
        if T in parameters["period"].values:

            # Model E
            pEE = getpE(cbb21_pSAobs_T[i].h,
                        parameters.modelE_LE[i],
                        parameters.modelE_gammaE[i]).reshape(-1)

            # Model EA
            pEAA = getpE(cbb21_pSAobs_T[i].h,
                         parameters.modelEA_LE[i],
                         parameters.modelEA_gammaE[i]).reshape(-1)
            pAA = getpA(cbb21_pSAobs_T[i].dA, parameters.modelEA_LA[i])
            pEA = MpEA(pEAA, pAA)

            # Model EAS
            pE = getpE(cbb21_pSAobs_T[i].h,
                       parameters.modelEAS_LE[i],
                       parameters.modelEAS_gammaE[i]).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, parameters.modelEAS_LA[i])
            pS = getpS(cbb21_pSAobs_T[i].dS, parameters.modelEAS_LS[i])
            pEAS = MpEAS(pE, pA, pS, parameters.modelEAS_w[i])

            # Fisher z-transform residuals
            def fisher_residual(obs, pred):
                p = np.clip(pred, -0.9999, 0.9999)
                z_pred = 0.5 * np.log((1 + p) / (1 - p))
                return (obs - z_pred) * np.sqrt(cbb21_pSAobs_T[i].n_j - 2)

            eE   = fisher_residual(cbb21_pSAobs_T[i].z_p_hat, pEE)
            eEA  = fisher_residual(cbb21_pSAobs_T[i].z_p_hat, pEA)
            eEAS = fisher_residual(cbb21_pSAobs_T[i].z_p_hat, pEAS)

            # Helper to compute stats + CI
            def compute_stats(arr):
                mu_e  = np.nanmean(arr)
                sig_e = np.nanstd(arr)
                try:
                    res_mu  = bootstrap((arr,), np.nanmean)
                    res_sig = bootstrap((arr,), np.nanstd)
                    mu_e5, mu_e95   = res_mu.confidence_interval
                    sig_e5, sig_e95 = res_sig.confidence_interval
                except Exception:
                    mu_e5 = mu_e95 = np.nan
                    sig_e5 = sig_e95 = np.nan
                return mu_e, sig_e, mu_e5, mu_e95, sig_e5, sig_e95

            # Save results
            for model, arr in zip(["E", "EA", "EAS"], [eE, eEA, eEAS]):
                results[model]["e"].append(arr)
                mu_e, sig_e, mu_e5, mu_e95, sig_e5, sig_e95 = compute_stats(arr)
                results[model]["mu_e"].append(mu_e)
                results[model]["sig_e"].append(sig_e)
                results[model]["mu_e5"].append(mu_e5)
                results[model]["mu_e95"].append(mu_e95)
                results[model]["sig_e5"].append(sig_e5)
                results[model]["sig_e95"].append(sig_e95)

        else:
            print(f'Bodenmann et al., 2023 not available for pSA = {T:.2f}s')
            for model in results.keys():
                for key in results[model]:
                    results[model][key].append(np.nan)

    return results

# Rampersad et al., (2026)
# Basin Dissimilarity Correlation Model
def getpZ(dZ,lZ):
    pZ = np.exp(-dZ/lZ)
    return pZ

# Topographic Dissimilarity Correlation Model (Exponential)
def getpT(dT,lT):
    pT = np.exp(-dT/lT)
    return pT

# combine and call model: EASZ
def MpEASZ(pE,pA,pS,pZ,w,wS):
    pEASZ = pE*(w*pA+(1-w)*(pS*wS + pZ*(1-wS)))
    return pEASZ

# combine and call model: EAST
def MpEAST(pE,pA,pS,pT,w,wS):
    pEAST = pE*(w*pA+(1-w)*(pS*wS + pT*(1-wS)))
    return pEAST

# combine and call model: EASZT
def MpEASZT(pE, pA, pS, pZ, pT, w, wS, wZ, wT):
    wS = np.clip(wS, 0.0, 1.0)
    wZ = np.clip(wZ, 0.0, 1.0)
    wT = np.clip(wT, 0.0, 1.0)

    w_sum = wS + wZ + wT
    if w_sum > 1.0:
        wS = wS / w_sum
        wZ = wZ / w_sum
        wT = wT / w_sum

    pEASZT = pE * (w * pA + (1.0 - w) * (pS*wS + pZ*wZ + pT*wT))
    return np.clip(pEASZT, -0.9999, 0.9999)

# Default SCM parameter locations; tables are loaded lazily to permit imports without data.
param_directory = Path(__file__).resolve().parent.parent / "data" / "nzwide_scm"
dwbar_files = "NZGMDB_v4p3_dwbar"


def _load_ra26_parameters(name):
    """Load a fitted SCM parameter table when a model is evaluated."""
    files = {
        "nz": param_directory / f"{dwbar_files}_param_summary.csv",
        "crus": param_directory / f"{dwbar_files}_crus_param_summary.csv",
        "Wellington": param_directory / "group1" / f"{dwbar_files}_grp1_param_summary.csv",
        "Canterbury": param_directory / "group2" / f"{dwbar_files}_grp2_param_summary.csv",
        "South Island": param_directory / "groups" / f"{dwbar_files}_grps_param_summary.csv",
        "North Island": param_directory / "groupn" / f"{dwbar_files}_grpn_param_summary.csv",
        "ln": param_directory / f"{dwbar_files}_T6_lnparam_summary.csv",
    }
    if name not in files:
        raise ValueError(f"Unknown parameter set: {name}")
    path = files[name]
    if not path.is_file():
        raise FileNotFoundError(f"SCM parameter file not found: {path}. Place the CSV in data/nzwide_scm/ or pass parameters explicitly.")
    return pd.read_csv(path)

# linear/default
def GMSCM_RA26(cbb21_pSAobs_T, period, parameters = None):

    if parameters is None:
        parameters = _load_ra26_parameters("nz")
    elif isinstance(parameters, str):
        parameters = _load_ra26_parameters(parameters)

    # Fisher z residuals with clipping
    def fisher_residual(p, obs, n):
        p = np.clip(p, -0.9999, 0.9999)
        z_p = 0.5 * np.log((1 + p) / (1 - p))
        return (obs - z_p) * np.sqrt(n - 2)

    models = ["E","EA","EAS","EASZ","EAST","EASZT"]

    # Storage for summary stats
    results = {
        m: {
            "e": [], "e_5": [], "e_95": [],
            "mu_e": [], "mu_e5": [], "mu_e95": [],
            "sig_e": [], "sig_e5": [], "sig_e95": []
        }
        for m in models
    }

    # Storage for raw p-fields (Option B)
    # pstore[model][suffix] = list of arrays (one per period)
    pstore = {
        m: {"": [], "_5": [], "_95": []}
        for m in models
    }

    # Precompute period lookup
    period_map = {
        (0.01 if s.upper() == "PGA" else float(s.replace("pSA_", ""))): idx
        for idx, s in enumerate(parameters["period"].values)
    }

    for i, T in enumerate(period):

        if T not in period_map:
            print(f'Rampersad et al., 2026 not available for pSA = {T:.2f}s')
            for model in results:
                for key in results[model]:
                    results[model][key].append(np.nan)
                for suffix in ["", "_5", "_95"]:
                    pstore[model][suffix].append(np.nan)
            continue

        idx = period_map[T]

        obs = cbb21_pSAobs_T[i].z_p_hat
        n   = cbb21_pSAobs_T[i].n_j

        # Helper to append stats
        def append_stats(model_key, arr, suffix_tag):
            mu = np.nanmean(arr)
            sig = np.nanstd(arr)

            if suffix_tag == "":
                results[model_key]["e"].append(arr)
                results[model_key]["mu_e"].append(mu)
                results[model_key]["sig_e"].append(sig)

            elif suffix_tag == "_5":
                results[model_key]["e_5"].append(arr)
                results[model_key]["mu_e5"].append(mu)
                results[model_key]["sig_e5"].append(sig)

            elif suffix_tag == "_95":
                results[model_key]["e_95"].append(arr)
                results[model_key]["mu_e95"].append(mu)
                results[model_key]["sig_e95"].append(sig)

        # --- Model E ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelE_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelE_gammaE{suffix}"]
            pE     = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)

            pstore["E"][suffix].append(pE)  # store raw p-field

            eE = fisher_residual(pE, obs, n)
            append_stats("E", eE, suffix)

        # --- Model EA ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEA_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEA_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEA_LA{suffix}"]

            pE  = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA  = getpA(cbb21_pSAobs_T[i].dA, LA)
            pEA = MpEA(pE, pA)

            pstore["EA"][suffix].append(pEA)

            eEA = fisher_residual(pEA, obs, n)
            append_stats("EA", eEA, suffix)

        # --- Model EAS ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEAS_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEAS_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEAS_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEAS_LS{suffix}"]
            w      = parameters.loc[idx, f"modelEAS_w{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].dS, LS)
            pEAS = MpEAS(pE, pA, pS, w)

            pstore["EAS"][suffix].append(pEAS)

            eEAS = fisher_residual(pEAS, obs, n)
            append_stats("EAS", eEAS, suffix)

        # --- Model EASZ ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEASZ_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEASZ_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEASZ_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEASZ_LS{suffix}"]
            LZ     = parameters.loc[idx, f"modelEASZ_LZ{suffix}"]
            w      = parameters.loc[idx, f"modelEASZ_w{suffix}"]
            wS     = parameters.loc[idx, f"modelEASZ_wS{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].dS, LS)
            pZ = getpZ(cbb21_pSAobs_T[i].dZ1, LZ)
            pEASZ = MpEASZ(pE, pA, pS, pZ, w, wS)

            pstore["EASZ"][suffix].append(pEASZ)

            eEASZ = fisher_residual(pEASZ, obs, n)
            append_stats("EASZ", eEASZ, suffix)

        # --- Model EAST ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEAST_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEAST_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEAST_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEAST_LS{suffix}"]
            LT     = parameters.loc[idx, f"modelEAST_LT{suffix}"]
            w      = parameters.loc[idx, f"modelEAST_w{suffix}"]
            wS     = parameters.loc[idx, f"modelEAST_wS{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].dS, LS)
            pT = getpT(cbb21_pSAobs_T[i].dTRI, LT)
            pEAST = MpEAST(pE, pA, pS, pT, w, wS)

            pstore["EAST"][suffix].append(pEAST)

            eEAST = fisher_residual(pEAST, obs, n)
            append_stats("EAST", eEAST, suffix)

        # --- Model EASZT ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEASZT_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEASZT_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEASZT_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEASZT_LS{suffix}"]
            LZ     = parameters.loc[idx, f"modelEASZT_LZ{suffix}"]
            LT     = parameters.loc[idx, f"modelEASZT_LT{suffix}"]
            w      = parameters.loc[idx, f"modelEASZT_w{suffix}"]
            wS     = parameters.loc[idx, f"modelEASZT_wS{suffix}"]
            wZ     = parameters.loc[idx, f"modelEASZT_wZ{suffix}"]
            wT     = parameters.loc[idx, f"modelEASZT_wT{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].dS, LS)
            pZ = getpZ(cbb21_pSAobs_T[i].dZ1, LZ)
            pT = getpT(cbb21_pSAobs_T[i].dTRI, LT)
            pEASZT = MpEASZT(pE, pA, pS, pZ, pT, w, wS, wZ, wT)

            pstore["EASZT"][suffix].append(pEASZT)

            eEASZT = fisher_residual(pEASZT, obs, n)
            append_stats("EASZT", eEASZT, suffix)

    return results, pstore

# log version
def GMSCM_RA26ln(cbb21_pSAobs_T, period, parameters = None):
    if parameters is None:
        parameters = _load_ra26_parameters("ln")

    # Fisher z residuals with clipping
    def fisher_residual(p, obs, n):
        p = np.clip(p, -0.9999, 0.9999)
        z_p = 0.5 * np.log((1 + p) / (1 - p))
        return (obs - z_p) * np.sqrt(n - 2)

    models = ["E","EA","EAlnS","EAlnSZ","EAlnST","EAlnSZT"]

    # Storage for summary stats
    results = {
        m: {
            "e": [], "e_5": [], "e_95": [],
            "mu_e": [], "mu_e5": [], "mu_e95": [],
            "sig_e": [], "sig_e5": [], "sig_e95": []
        }
        for m in models
    }

    # Storage for raw p-fields (Option B)
    # pstore[model][suffix] = list of arrays (one per period)
    pstore = {
        m: {"": [], "_5": [], "_95": []}
        for m in models
    }

    # Precompute period lookup
    period_map = {
        (0.01 if s.upper() == "PGA" else float(s.replace("pSA_", ""))): idx
        for idx, s in enumerate(parameters["period"].values)
    }

    for i, T in enumerate(period):

        if T not in period_map:
            print(f'Rampersad et al., 2026 not available for pSA = {T:.2f}s')
            for model in results:
                for key in results[model]:
                    results[model][key].append(np.nan)
                for suffix in ["", "_5", "_95"]:
                    pstore[model][suffix].append(np.nan)
            continue

        idx = period_map[T]

        obs = cbb21_pSAobs_T[i].z_p_hat.to_numpy(dtype=float)
        n = cbb21_pSAobs_T[i].n_j.to_numpy(dtype=float)

        # Helper to append stats
        def append_stats(model_key, arr, suffix_tag):
            mu = np.nanmean(arr)
            sig = np.nanstd(arr)

            if suffix_tag == "":
                results[model_key]["e"].append(arr)
                results[model_key]["mu_e"].append(mu)
                results[model_key]["sig_e"].append(sig)

            elif suffix_tag == "_5":
                results[model_key]["e_5"].append(arr)
                results[model_key]["mu_e5"].append(mu)
                results[model_key]["sig_e5"].append(sig)

            elif suffix_tag == "_95":
                results[model_key]["e_95"].append(arr)
                results[model_key]["mu_e95"].append(mu)
                results[model_key]["sig_e95"].append(sig)

        # --- Model E ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelE_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelE_gammaE{suffix}"]
            pE     = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)

            pstore["E"][suffix].append(pE)  # store raw p-field

            eE = fisher_residual(pE, obs, n)
            append_stats("E", eE, suffix)

        # --- Model EA ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEA_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEA_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEA_LA{suffix}"]

            pE  = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA  = getpA(cbb21_pSAobs_T[i].dA, LA)
            pEA = MpEA(pE, pA)

            pstore["EA"][suffix].append(pEA)

            eEA = fisher_residual(pEA, obs, n)
            append_stats("EA", eEA, suffix)

        # --- Model EAS ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEAlnS_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEAlnS_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEAlnS_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEAlnS_LS{suffix}"]
            w      = parameters.loc[idx, f"modelEAlnS_w{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].lndS, LS)
            pEAS = MpEAS(pE, pA, pS, w)

            pstore["EAlnS"][suffix].append(pEAS)

            eEAS = fisher_residual(pEAS, obs, n)
            append_stats("EAlnS", eEAS, suffix)

        # --- Model EASZ ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEAlnSZ_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEAlnSZ_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEAlnSZ_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEAlnSZ_LS{suffix}"]
            LZ     = parameters.loc[idx, f"modelEAlnSZ_LZ{suffix}"]
            w      = parameters.loc[idx, f"modelEAlnSZ_w{suffix}"]
            wS     = parameters.loc[idx, f"modelEAlnSZ_wS{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].lndS, LS)
            pZ = getpZ(cbb21_pSAobs_T[i].lndZ1, LZ)
            pEASZ = MpEASZ(pE, pA, pS, pZ, w, wS)

            pstore["EAlnSZ"][suffix].append(pEASZ)

            eEASZ = fisher_residual(pEASZ, obs, n)
            append_stats("EAlnSZ", eEASZ, suffix)

        # --- Model EAST ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEAlnST_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEAlnST_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEAlnST_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEAlnST_LS{suffix}"]
            LT     = parameters.loc[idx, f"modelEAlnST_LT{suffix}"]
            w      = parameters.loc[idx, f"modelEAlnST_w{suffix}"]
            wS     = parameters.loc[idx, f"modelEAlnST_wS{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].lndS, LS)
            pT = getpT(cbb21_pSAobs_T[i].lndTRI, LT)
            pEAST = MpEAST(pE, pA, pS, pT, w, wS)

            pstore["EAlnST"][suffix].append(pEAST)

            eEAST = fisher_residual(pEAST, obs, n)
            append_stats("EAlnST", eEAST, suffix)

        # --- Model EASZT ---
        for suffix in ["", "_5", "_95"]:
            LE     = parameters.loc[idx, f"modelEAlnSZT_LE{suffix}"]
            gammaE = parameters.loc[idx, f"modelEAlnSZT_gammaE{suffix}"]
            LA     = parameters.loc[idx, f"modelEAlnSZT_LA{suffix}"]
            LS     = parameters.loc[idx, f"modelEAlnSZT_LS{suffix}"]
            LZ     = parameters.loc[idx, f"modelEAlnSZT_LZ{suffix}"]
            LT     = parameters.loc[idx, f"modelEAlnSZT_LT{suffix}"]
            w      = parameters.loc[idx, f"modelEAlnSZT_w{suffix}"]
            wS     = parameters.loc[idx, f"modelEAlnSZT_wS{suffix}"]
            wZ     = parameters.loc[idx, f"modelEAlnSZT_wZ{suffix}"]
            wT     = parameters.loc[idx, f"modelEAlnSZT_wT{suffix}"]

            pE = getpE(cbb21_pSAobs_T[i].h, LE, gammaE).reshape(-1)
            pA = getpA(cbb21_pSAobs_T[i].dA, LA)
            pS = getpS(cbb21_pSAobs_T[i].lndS, LS)
            pZ = getpZ(cbb21_pSAobs_T[i].lndZ1, LZ)
            pT = getpT(cbb21_pSAobs_T[i].lndTRI, LT)
            pEASZT = MpEASZT(pE, pA, pS, pZ, pT, w, wS, wZ, wT)

            pstore["EAlnSZT"][suffix].append(pEASZT)

            eEASZT = fisher_residual(pEASZT, obs, n)
            append_stats("EAlnSZT", eEASZT, suffix)

    return results, pstore

# warp kernels (trial)
KumarWarp_T  = np.array([0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
KumarWarp_LE = np.array([26.46, 25.19, 22.37, 20.59, 28.31, 29.35, 23.11, 25.28, 23.33, 25.39, 27.32, 27.17])
KumarWarp_gE = np.array([0.43, 0.43, 0.42, 0.42, 0.40, 0.36, 0.34, 0.42, 0.44, 0.44, 0.42, 0.41])
KumarWarp_ax = np.array([0.99, 1.06, 1.05, 0.94, 1.97, 1.83, 3.37, 1.08, 0.95, 1.17, 1.16, 1.28])
KumarWarp_ay = np.array([1.61, 1.65, 1.65, 1.41, 2.18, 2.24, 3.00, 1.06, 0.59, 0.70, 1.28, 1.69])
KumarWarp_bx = np.array([1.26, 1.48, 1.51, 1.35, 1.12, 0.90, 1.68, 0.63, 0.41, 0.47, 0.59, 0.69])
KumarWarp_by = np.array([2.65, 2.84, 2.93, 2.70, 2.76, 2.40, 2.99, 1.18, 0.80, 0.72, 0.82, 1.03])

def get_kumar_params(T):
    return dict(
        LE  = float(np.interp(T, KumarWarp_T, KumarWarp_LE)),
        gE  = float(np.interp(T, KumarWarp_T, KumarWarp_gE)),
        a_x = float(np.interp(T, KumarWarp_T, KumarWarp_ax)),
        a_y = float(np.interp(T, KumarWarp_T, KumarWarp_ay)),
        b_x = float(np.interp(T, KumarWarp_T, KumarWarp_bx)),
        b_y = float(np.interp(T, KumarWarp_T, KumarWarp_by)),
    )

def kumar_cdf(x, a, b, eps=1e-12):
    """
    Kumaraswamy CDF: F(x) = 1 - (1 - x^a)^b, x in [0,1]
    """
    x = np.clip(x, eps, 1.0 - eps)
    return 1.0 - np.power(1.0 - np.power(x, a), b)

def warp_xy_cdf(X, a_x, b_x, a_y, b_y, rescale_back=True):
    """
    X : (n,2) coordinates in meters
    bounds : dict with fixed training-domain bounds in meters
             {'x_min','x_max','y_min','y_max'}
    """
    x = X[:, 0]
    y = X[:, 1]

    x_min = 1108366.9155328241
    x_max = 2089358.9051267859
    y_min = 4801322.9663666375
    y_max = 5985498.36062547

    dx = max(x_max - x_min, 1e-12)
    dy = max(y_max - y_min, 1e-12)

    xu = (x - x_min) / dx
    yu = (y - y_min) / dy

    # clip to [0,1] in case of tiny numerical overrun
    xu = np.clip(xu, 0.0, 1.0)
    yu = np.clip(yu, 0.0, 1.0)

    xw_u = kumar_cdf(xu, a_x, b_x)
    yw_u = kumar_cdf(yu, a_y, b_y)

    if rescale_back:
        xw = x_min + dx * xw_u
        yw = y_min + dy * yw_u
    else:
        xw = xw_u
        yw = yw_u

    return np.column_stack([xw, yw])

def GMSCM_KumarWarp(cbb21, T, n_rec, plot=False):
    """
    Compute correlation coefficients and deviations using the Kumar-warped
    spatial kernel with period-dependent point estimates.

    cbb21 must contain per-pair coordinates:
      'sta_j_xm', 'sta_j_ym', 'sta_k_xm', 'sta_k_ym'
    """

    # Get period-specific parameters ---
    pars = get_kumar_params(T)
    LE   = pars["LE"]
    gE   = pars["gE"]
    a_x  = pars["a_x"]
    a_y  = pars["a_y"]
    b_x  = pars["b_x"]
    b_y  = pars["b_y"]

    # Build a combined coordinate array for warping
    Xj = cbb21[['sta_j_xm', 'sta_j_ym']].values
    Xk = cbb21[['sta_k_xm', 'sta_k_ym']].values
    X_all = np.vstack([Xj, Xk])          # shape (2N, 2)

    # Warp all coordinates using global bounds
    Xw_all = warp_xy_cdf(
        X_all,
        a_x=a_x, b_x=b_x,
        a_y=a_y, b_y=b_y,
        rescale_back=True
    )

    N = Xj.shape[0]
    Xw_j = Xw_all[:N]
    Xw_k = Xw_all[N:]

    # Pairwise distances and model correlations (one per pair)
    d_m   = np.sqrt(np.sum((Xw_j - Xw_k)**2, axis=1))
    d     = d_m / 1000.0          # km

    LEt = LE**gE
    p   = np.exp(-(d**gE) / LEt)


    # Empirical correlations
    p_hat   = cbb21.p_hat
    z_p_hat = cbb21.z_p_hat

    # Fisher z-transform of model correlations
    z_p = 0.5 * np.log((1 + p) / (1 - p))

    # Deviation
    e = (z_p_hat - z_p) * np.sqrt(cbb21.n_j - 2)

    # Summary statistics + bootstrap
    e_mean = np.nanmean(e)
    e_std  = np.nanstd(e)

    data    = (e,)
    res     = bootstrap(data, np.mean)
    res_std = bootstrap(data, np.std)

    e_5     = res.confidence_interval.low
    e_95    = res.confidence_interval.high
    estd_5  = res_std.confidence_interval.low
    estd_95 = res_std.confidence_interval.high

    # Optional plotting
    if plot:
        # Distances in original and warped space (metres → km)
        d_raw_m  = np.sqrt(np.sum((Xj - Xk)**2, axis=1))
        d_warp_m = np.sqrt(np.sum((Xw_j - Xw_k)**2, axis=1))

        d_raw  = d_raw_m / 1000.0   # km
        d_warp = d_warp_m / 1000.0  # km

        p_warp = p
        p_raw  = np.exp(-(d_raw**gE) / LEt)

        dh = cbb21.h   # should already be in km
        plot_str = 'PGA' if T == 0 else f'pSA = {T:0.2f}s'

        fig, ax = plt.subplots(figsize=(8,5), dpi=150, constrained_layout=True)

        im = ax.scatter(
            dh, p_hat, c=e, vmin=-4, vmax=4,
            cmap='bwr', edgecolors='black', s=60,
            label='Empirical'
        )

        idx_raw  = np.argsort(d_raw)
        idx_warp = np.argsort(d_warp)

        ax.plot(d_raw[idx_raw],  p_raw[idx_raw],  '-k',  lw=2, label='Model (raw)')
        ax.plot(d_warp[idx_warp], p_warp[idx_warp], '--r', lw=2, label='Model (warped)')

        ax.set_xlim(0, np.max(dh))
        ax.set_ylim(-1, 1)
        ax.set_title(f'KumarWarp: {plot_str}')
        ax.set_ylabel(r'Correlation, $\hat{p}$ (%s)' % plot_str)
        ax.set_xlabel('Separation Distance, h (km)')

        textstr = '\n'.join((
            rf'$\mu_e={e_mean:.2f}$',
            rf'$\sigma_e={e_std:.2f}$'
        ))
        ax.text(
            0.99, -0.95, textstr, fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
        )

        fig.colorbar(im, ax=ax, label='Correlation Deviation, e', aspect=20)
        plt.show()

    return p, e, e_mean, e_std, e_5, e_95, estd_5, estd_95