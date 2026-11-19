"""Ground-motion residual partitioning and mixed-effects utilities."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm.auto import tqdm
import statsmodels.formula.api as smf
import json
import shapely
from shapely.geometry import shape

# get backarc stations
def get_backarc_mask(backarc_json_ffp, locs):
    """
    Return a boolean mask indicating which locations fall inside a backarc polygon.

    Parameters
    ----------
    backarc_json_ffp : str or path-like
        GeoJSON file containing a single polygon or multipolygon geometry.
    locs : ndarray, shape (n, 2)
        Point coordinates in the same CRS as the GeoJSON geometry.
        The required order is [x, y], i.e. [longitude, latitude] for lon/lat data.

    Returns
    -------
    mask : ndarray, shape (n,)
        True for points contained by the polygon. Points exactly on the boundary
        are False when using shapely.contains.
    """
    with open(backarc_json_ffp) as f:
        gj = json.load(f)

    geom = shape(gj["geometry"])

    # Build a vectorized array of Points
    pts = shapely.points(locs[:, 0], locs[:, 1])

    # Vectorized contains (no Python loop, no GEOS mismatch)
    return shapely.contains(geom, pts)

# Within-event residual approximation
def getwithineventres_baker(im_obs, im_sim, events, tau, phi):
    """
    Compute event and within-event residuals using the Baker/NGA-W2-style
    event-partition approximation.

    This function is intentionally kept close to the Baker MATLAB implementation:
    https://github.com/bakerjw/NGAW2_correlations/blob/master/src/residuals/GetInterIntraEventResiduals.m

    Model
    -----
    For each event e and observation j,

        epsilon_ej = ln(Y_obs,ej) - ln(Y_med,ej)

    is partitioned into an event term and a within-event term using the supplied
    tau and phi values.

    Inputs
    ------
    im_obs : array-like or pandas.Series
        Observed positive IM/FAS values.

    im_sim : array-like or pandas.Series
        Median-predicted positive IM/FAS values.

    events : pandas.Series
        Event IDs, one per observation. The function uses `events.unique()` and
        loops over events in first-appearance order.

    tau : array-like or pandas.Series
        Between-event standard deviation values. This implementation permits
        row-wise tau values.

    phi : array-like or pandas.Series
        Within-event standard deviation values. This implementation permits
        row-wise phi values.

    Returns
    -------
    dw_bar : pandas.DataFrame
        Normalized within-event residuals, approximately

            (epsilon_ej - dbe_ej) / phi_ej.

    dbe_bar : pandas.DataFrame
        Normalized event term, approximately

            dbe_ej / tau_ej.

    dw : pandas.DataFrame
        Raw within-event residuals,

            epsilon_ej - dbe_ej.

    dbe : pandas.DataFrame
        Raw event term.

    Notes
    -----
    This implementation permits row-wise tau and phi values. This is useful when
    the source GMM/OpenQuake formulation returns aleatory components that vary
    by site, source, path, or tectonic condition. In that case, `dbe` may become
    row-wise rather than one scalar per event. This is intentional here for
    compatibility with the validated Baker-style workflow.

    The returned DataFrames are appended event-by-event. Therefore, the output
    order is event-grouped according to `events.unique()`. If the input rows are
    not already grouped by event, the returned rows will not necessarily match
    the original dataframe row order.
    """
    evid_unique = events.unique()
    dw_bar = []
    dbe_bar = []
    dbe = []
    dw = []

    for i in range(len(evid_unique)):
        evid = evid_unique[i]
        filt = events == evid

        epsilon = np.log(im_obs[filt]) - np.log(im_sim[filt])
        tau_temp = tau[filt]
        phi_temp = phi[filt]

        oneN = np.ones((len(epsilon), 1))
        ccsub = np.diag(np.square(phi_temp))

        numerator = np.matmul(
            np.transpose(oneN),
            np.matmul(np.linalg.inv(ccsub), epsilon)
        )

        denom = np.matmul(
            np.transpose(oneN),
            np.matmul(np.linalg.inv(ccsub), oneN)
        )

        denom = 1 / np.square(tau_temp) + denom[0]

        dbe_temp = numerator / denom
        dw_temp = (epsilon - dbe_temp) / phi_temp
        dbe_norm = dbe_temp / tau_temp

        dw_bar.extend(dw_temp)
        dw.extend(epsilon - dbe_temp)
        dbe_bar.extend(dbe_norm)
        dbe.extend(dbe_temp)

    dw = pd.DataFrame(dw).rename(columns={0: "dw"})
    dbe = pd.DataFrame(dbe).rename(columns={0: "dbe"})

    dw_bar = pd.DataFrame(dw_bar).rename(columns={0: "dw_bar"})
    dbe_bar = pd.DataFrame(dbe_bar).rename(columns={0: "dbe_bar"})

    return dw_bar, dbe_bar, dw, dbe

# compute within-event residuals: Abrahamson and Youngs 1992
def ay92(
    im_obs_col,
    im_sim_col,
    evid_col,
    SA_tau_col,
    SA_phi_col,
    *,
    center=False,
    tol=1e-10,
    maxiter=200,
):
    """
    Estimate event and within-event residuals using a traditional
    Abrahamson-Youngs-style random-intercept likelihood for one IM/frequency
    column.

    Model
    -----
    For observation j in event e, define the total residual

        Delta_ej = ln(Y_obs,ej) - ln(Y_med,ej).

    The event-only residual decomposition is

        Delta_ej = deltaB_e + deltaW_ej,

    where

        deltaB_e  ~ N(0, tau^2),
        deltaW_ej ~ N(0, phi^2).

    For each event, the marginal covariance is

        Sigma_e = phi^2 I + tau^2 11',

    and the function estimates one global tau and one global phi for the
    supplied IM/frequency column by maximum likelihood.

    Inputs
    ------
    im_obs_col : array-like
        Observed positive IM/FAS values for one period or frequency.

    im_sim_col : array-like
        Median-predicted positive IM/FAS values for the same rows.

    evid_col : array-like
        Event IDs, one per observation.

    SA_tau_col : array-like
        GMM/source tau values. These are used only to initialize the optimizer;
        the function estimates its own tau from the residuals.

    SA_phi_col : array-like
        GMM/source phi values. These are used only to initialize the optimizer;
        the function estimates its own phi from the residuals.

    center : bool, default False
        If True, subtract the grand mean of Delta before fitting. If False, the
        decomposition assumes the median prediction is already unbiased on
        average.

    tol : float, default 1e-10
        Optimizer tolerance passed to L-BFGS-B.

    maxiter : int, default 200
        Maximum optimizer iterations.

    Returns
    -------
    out : dict
        Dictionary with the following entries:

        tau : float
            Estimated between-event standard deviation.

        phi : float
            Estimated within-event standard deviation.

        converged : bool
            Optimizer success flag.

        nll : float
            Negative log-likelihood at the returned solution.

        dbe_bar : pandas.Series
            Row-aligned normalized event term,

                deltaB_e / tau.

        dw_bar : pandas.Series
            Row-aligned normalized within-event residual,

                deltaW_ej / phi.

        deltaB : pandas.Series
            Row-aligned unstandardized event term deltaB_e.

        dbe_se : pandas.Series
            Row-aligned approximate standard error of deltaB_e, conditional on
            the estimated tau and phi.

        eps : pandas.Series
            Row-aligned unstandardized within-event residual,

                deltaW_ej = Delta_ej - deltaB_e.

        z_strict : pandas.Series
            Optional diagnostic residual standardized within each event to have
            sample mean zero and sample standard deviation one.

    Notes
    -----
    This is an event-only decomposition. It does not estimate a site-to-site
    term. For a Stafford-style event/site/within decomposition, use the crossed
    event-site mixed model instead.

    `z_strict` is only a diagnostic quantity. It should not be used as the
    production within-event residual for covariance modelling because it forces
    each event to have zero mean and unit variance, thereby changing the
    residual covariance structure.
    """
    im_obs_col = np.asarray(im_obs_col, dtype=float)
    im_sim_col = np.asarray(im_sim_col, dtype=float)
    SA_tau_col = np.asarray(SA_tau_col, dtype=float)
    SA_phi_col = np.asarray(SA_phi_col, dtype=float)
    evid_col = np.asarray(evid_col)

    mask = (
        np.isfinite(im_obs_col)
        & np.isfinite(im_sim_col)
        & np.isfinite(SA_tau_col)
        & np.isfinite(SA_phi_col)
        & (im_obs_col > 0.0)
        & (im_sim_col > 0.0)
        & (SA_tau_col > 0.0)
        & (SA_phi_col > 0.0)
        & pd.notna(evid_col)
    )

    y = im_obs_col[mask]
    mu = im_sim_col[mask]
    phi_gmm = SA_phi_col[mask]
    tau_gmm = SA_tau_col[mask]
    ev = pd.Series(evid_col[mask])

    if len(y) == 0:
        raise ValueError("No valid rows remain after masking.")

    Delta = np.log(y) - np.log(mu)

    if center:
        Delta = Delta - Delta.mean()

    df = pd.DataFrame({"evid": ev, "r": Delta})
    grp = df.groupby("evid", sort=False)

    n = grp.size()
    sum_r = grp["r"].sum()
    sum_r2 = grp["r"].apply(lambda x: float(np.square(x).sum()))

    N = int(n.sum())

    phi2_0 = float(np.mean(phi_gmm) ** 2)
    tau2_0 = float(np.mean(tau_gmm) ** 2)

    phi2_0 = max(phi2_0, 1e-12)
    tau2_0 = max(tau2_0, 1e-12)

    def nll(log_phi2, log_tau2):
        phi2 = np.exp(log_phi2)
        tau2 = np.exp(log_tau2)

        ld = ((n - 1) * np.log(phi2) + np.log(phi2 + n * tau2)).sum()

        quad = (
            (sum_r2 / phi2)
            - (tau2 / (phi2 * (phi2 + n * tau2))) * (sum_r ** 2)
        ).sum()

        return 0.5 * (ld + quad) + 0.5 * N * np.log(2.0 * np.pi)

    res = minimize(
        lambda x: nll(x[0], x[1]),
        x0=np.log([phi2_0, tau2_0]),
        method="L-BFGS-B",
        bounds=[
            (np.log(1e-12), np.log(1e6)),
            (np.log(1e-12), np.log(1e6)),
        ],
        options=dict(maxiter=maxiter, ftol=tol),
    )

    if res.success:
        phi2_hat, tau2_hat = np.exp(res.x[0]), np.exp(res.x[1])
    else:
        phi2_hat, tau2_hat = phi2_0, tau2_0

    phi2_hat = float(max(phi2_hat, 1e-10))
    tau2_hat = float(max(tau2_hat, 1e-12))

    phi_hat = float(np.sqrt(phi2_hat))
    tau_hat = float(np.sqrt(tau2_hat))

    # AY-style event term and within-event residual.
    deltaB_e = (tau2_hat * sum_r) / (n * tau2_hat + phi2_hat)
    deltaB_row = df["evid"].map(deltaB_e)

    eps = df["r"] - deltaB_row

    dbebar = deltaB_row / tau_hat
    dwbar = eps / phi_hat

    se_dbe_e = np.sqrt(
        (tau2_hat * phi2_hat) / (phi2_hat + n * tau2_hat)
    )
    dbe_se_row = df["evid"].map(se_dbe_e)

    z_strict = eps.groupby(df["evid"]).transform(
        lambda s: (
            (s - s.mean()) / s.std(ddof=1)
            if s.std(ddof=1) > 0.0
            else np.nan
        )
    )

    out = {
        "tau": tau_hat,
        "phi": phi_hat,
        "converged": bool(res.success),
        "nll": float(res.fun),
        "dbe_bar": pd.Series(np.nan, index=np.arange(len(im_obs_col))),
        "dw_bar": pd.Series(np.nan, index=np.arange(len(im_obs_col))),
        "deltaB": pd.Series(np.nan, index=np.arange(len(im_obs_col))),
        "dbe_se": pd.Series(np.nan, index=np.arange(len(im_obs_col))),
        "eps": pd.Series(np.nan, index=np.arange(len(im_obs_col))),
        "z_strict": pd.Series(np.nan, index=np.arange(len(im_obs_col))),
    }

    out["dbe_bar"].loc[mask] = dbebar.to_numpy()
    out["dw_bar"].loc[mask] = dwbar.to_numpy()
    out["deltaB"].loc[mask] = deltaB_row.to_numpy()
    out["dbe_se"].loc[mask] = dbe_se_row.to_numpy()
    out["eps"].loc[mask] = eps.to_numpy()
    out["z_strict"].loc[mask] = z_strict.to_numpy()

    return out

def ay92_per_event_tau_phi(im_obs, im_sim, events, *, center=False, eps=1e-12):
    """
    Diagnostic Abrahamson-Youngs-style partition with event-specific
    tau_i and phi_i for one IM/frequency column.

    Model
    -----
    For each event e,

        Delta_e = deltaB_e 1 + deltaW_e

    with marginal covariance

        Sigma_e = phi_i^2 I + tau_i^2 11'

    where tau_i and phi_i are estimated separately for each event.

    Important
    ---------
    This is not the theoretically preferred production model for Stafford-style
    decomposition, because tau is normally a between-event variance component
    estimated across events. Estimating tau_i from a single event is unstable
    and should be treated as a diagnostic or sensitivity calculation.

    Parameters
    ----------
    im_obs, im_sim : array-like
        Observed and predicted positive IM/FAS values.
    events : array-like
        Event IDs.
    center : bool, default False
        If True, subtract the grand mean residual before event-level fitting.
    eps : float
        Numerical floor for variances.

    Returns
    -------
    dw_bar : Series
        Standardized within-event residuals, (Delta - deltaB_i) / phi_i.
    dbe_bar : Series
        Standardized event term, deltaB_i / tau_i.
    eta_hat : Series
        Unstandardized event term deltaB_i, broadcast to rows.
    eps_out : Series
        Unstandardized within-event residuals, Delta - deltaB_i.
    ev_tbl : DataFrame
        Event-level summary with columns:
        ['evid', 'n', 'tau_i', 'phi_i', 'se_dbe', 'ok'].
    """
    ev = pd.Series(events)
    y = pd.Series(np.asarray(im_obs, dtype=float), index=ev.index)
    mu = pd.Series(np.asarray(im_sim, dtype=float), index=ev.index)

    valid = (
        np.isfinite(y)
        & np.isfinite(mu)
        & (y > 0.0)
        & (mu > 0.0)
        & ev.notna()
    )

    r = pd.Series(np.nan, index=ev.index, dtype=float)
    r.loc[valid] = np.log(y.loc[valid]) - np.log(mu.loc[valid])

    if center:
        r.loc[valid] = r.loc[valid] - r.loc[valid].mean()

    dw_bar = pd.Series(np.nan, index=r.index, dtype=float)
    dbe_bar = pd.Series(np.nan, index=r.index, dtype=float)
    eta_hat = pd.Series(np.nan, index=r.index, dtype=float)
    eps_out = pd.Series(np.nan, index=r.index, dtype=float)

    rows = []

    for evid, idx in ev.loc[valid].groupby(ev.loc[valid], sort=False).groups.items():
        idx = pd.Index(idx)
        r_i = r.loc[idx].to_numpy(dtype=float)
        n_i = r_i.size

        ok = True

        if n_i >= 2:
            sum_r = float(r_i.sum())
            sum_r2 = float(np.square(r_i).sum())

            def nll(theta):
                phi2 = np.exp(theta[0])
                tau2 = np.exp(theta[1])

                ld = (n_i - 1) * np.log(phi2) + np.log(phi2 + n_i * tau2)
                quad = (
                    sum_r2 / phi2
                    - (tau2 / (phi2 * (phi2 + n_i * tau2))) * sum_r**2
                )

                return 0.5 * (ld + quad)

            m = sum_r / n_i
            s2 = float(np.var(r_i - m, ddof=1)) if n_i > 1 else eps

            phi2_0 = max(s2, eps)
            tau2_0 = max(m * m, eps)

            res = minimize(
                nll,
                x0=np.log([phi2_0, tau2_0]),
                method="L-BFGS-B",
                bounds=[
                    (np.log(eps), np.log(1e6)),
                    (np.log(eps), np.log(1e6)),
                ],
            )

            if res.success:
                phi2_i, tau2_i = np.exp(res.x[0]), np.exp(res.x[1])
            else:
                phi2_i, tau2_i = phi2_0, tau2_0
                ok = False

        else:
            # n_i == 1: tau_i and phi_i are not identifiable.
            sum_r = float(r_i.sum())
            phi2_i = eps
            tau2_i = eps
            ok = True

        phi2_i = float(max(phi2_i, eps))
        tau2_i = float(max(tau2_i, eps))

        phi_i = float(np.sqrt(phi2_i))
        tau_i = float(np.sqrt(tau2_i))

        se_dbe_i = float(np.sqrt((tau2_i * phi2_i) / (phi2_i + n_i * tau2_i)))

        deltaB_i = float((tau2_i * sum_r) / (n_i * tau2_i + phi2_i))
        eps_i = r_i - deltaB_i

        eta_hat.loc[idx] = deltaB_i
        eps_out.loc[idx] = eps_i
        dbe_bar.loc[idx] = deltaB_i / tau_i
        dw_bar.loc[idx] = eps_i / phi_i

        rows.append((evid, n_i, tau_i, phi_i, se_dbe_i, ok))

    ev_tbl = pd.DataFrame(
        rows,
        columns=["evid", "n", "tau_i", "phi_i", "se_dbe", "ok"],
    )

    return dw_bar, dbe_bar, eta_hat, eps_out, ev_tbl

def lmerfit_dw(
    im_obs,
    im_sim,
    pSAobs_T,
    show_progress: bool = True,
    progress_desc: str = "LMER residuals",
    use_conditional_re_se: bool = True,
    check_reconstruction: bool = True,
):
    """
    Estimate event and within-event residual components using an event-only
    random-intercept mixed model.

    For each IM/frequency column, define the total residual

        Delta_es = ln(Y_obs,es) - ln(Y_med,es)

    and fit

        Delta_es = beta0 + deltaB_e + deltaW_es,

    where

        beta0    : fixed intercept / model bias,
        deltaB_e : event random effect, Var(deltaB_e) = tau^2,
        deltaW_es: within-event residual, Var(deltaW_es) = phi^2.

    Inputs
    ------
    im_obs : pandas.DataFrame
        Observed IM/FAS table. Must contain:
        - one event-ID column named 'evid'
        - all IM/FAS columns listed in `pSAobs_T`

    im_sim : pandas.DataFrame
        Median-predicted IM/FAS table. Must have the same row order as `im_obs`
        and must contain:
        - one event-ID column named 'evid'
        - all IM/FAS columns listed in `pSAobs_T`

        The prediction columns are interpreted as positive median predictions,
        not logarithmic predictions.

    pSAobs_T : list of str
        Names of the IM/FAS columns to process, for example:
        ['pSA_0.01', 'pSA_0.02', ...] or frequency-domain ordinates.

    show_progress : bool, default True
        If True, display a tqdm progress bar.

    progress_desc : str, default "LMER residuals"
        Progress-bar label.

    use_conditional_re_se : bool, default True
        If True, attempt to use statsmodels' conditional random-effect
        covariance, `random_effects_cov`, to estimate event-effect standard
        errors. If unavailable, fall back to the usual shrinkage approximation.

    check_reconstruction : bool, default True
        If True, verify that

            Delta_es = (beta0 + deltaB_e) + deltaW_es

        for rows used in each fit.

    Returns
    -------
    im_dwbar_sm : pandas.DataFrame
        Same shape/index structure as `im_sim`, with processed IM/FAS columns
        replaced by normalized within-event residuals:

            deltaW_es / phi.

    im_dbebar_sm : pandas.DataFrame
        Same shape/index structure as `im_sim`, with processed IM/FAS columns
        replaced by normalized event total terms:

            (beta0 + deltaB_e) / tau.

        Note that this includes the fixed intercept beta0. For the zero-mean
        random event effect alone, use beta0 from `phi_tau_df` and subtract it
        from `im_dbe_sm`.

    im_dw_sm : pandas.DataFrame
        Raw within-event residuals:

            deltaW_es.

    im_dbe_sm : pandas.DataFrame
        Event total term broadcast to rows:

            beta0 + deltaB_e.

    im_dbe_se_sm : pandas.DataFrame
        Approximate standard error of beta0 + deltaB_e, broadcast to rows.

    phi_tau_df : pandas.DataFrame
        One row per processed column with:
        - T_label
        - phi
        - tau
        - beta0
        - var_beta0
        - se_source_dbe

    Notes
    -----
    This function is appropriate for an event-only residual decomposition.
    For Stafford-style decomposition including repeatable site effects, use
    `lmerfit_ds2s()` instead.

    The stochastic between-event component is deltaB_e, not beta0 + deltaB_e.
    The current output convention keeps beta0 + deltaB_e for backward
    compatibility with the previous workflow.
    """

    required_cols = {"evid", *pSAobs_T}
    missing_obs = sorted(required_cols.difference(im_obs.columns))
    missing_sim = sorted(required_cols.difference(im_sim.columns))

    if missing_obs:
        raise ValueError(f"im_obs is missing required columns: {missing_obs}")
    if missing_sim:
        raise ValueError(f"im_sim is missing required columns: {missing_sim}")

    if len(im_obs) != len(im_sim):
        raise ValueError("im_obs and im_sim must have the same number of rows.")

    im_dwbar_sm = im_sim.copy()
    im_dbebar_sm = im_sim.copy()
    im_dw_sm = im_sim.copy()
    im_dbe_sm = im_sim.copy()
    im_dbe_se_sm = im_sim.copy()

    for df in (im_dwbar_sm, im_dbebar_sm, im_dw_sm, im_dbe_sm, im_dbe_se_sm):
        df[pSAobs_T] = df[pSAobs_T].astype(np.float64)

    phi_sm = []
    tau_sm = []
    beta0_sm = []
    varbeta0_sm = []
    se_source_dbe_sm = []

    def get_event_re_se(result):
        re_cov = getattr(result, "random_effects_cov", None)

        if re_cov is None or not isinstance(re_cov, dict) or len(re_cov) == 0:
            return None

        se_vals = {}

        for key, cov in re_cov.items():
            if isinstance(cov, pd.DataFrame):
                diag = np.diag(cov.to_numpy(dtype=float))
            else:
                diag = np.diag(np.asarray(cov, dtype=float))

            se_vals[str(key)] = float(np.sqrt(np.maximum(diag[0], 0.0)))

        return pd.Series(se_vals, dtype=float)

    iterator = range(len(pSAobs_T))
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc)

    for i in iterator:
        col = pSAobs_T[i]

        y_obs = np.asarray(im_obs[col], dtype=float)
        y_sim = np.asarray(im_sim[col], dtype=float)

        valid = (
            np.isfinite(y_obs)
            & np.isfinite(y_sim)
            & (y_obs > 0.0)
            & (y_sim > 0.0)
            & im_obs["evid"].notna().to_numpy()
        )

        fit_df = pd.DataFrame({
            "evid": im_obs.loc[valid, "evid"].astype(str).to_numpy(),
            "res": np.log(y_obs[valid]) - np.log(y_sim[valid]),
        }, index=im_obs.index[valid])

        psa_mdf = smf.mixedlm(
            "res ~ 1",
            fit_df,
            groups=fit_df["evid"],
        ).fit(reml=True)

        beta0 = float(psa_mdf.fe_params["Intercept"])
        beta0_sm.append(beta0)

        phi_hat = float(np.sqrt(psa_mdf.scale))
        tau_hat = float(np.sqrt(np.asarray(psa_mdf.cov_re)[0, 0]))

        phi_sm.append(phi_hat)
        tau_sm.append(tau_hat)

        var_beta0 = float(psa_mdf.cov_params().loc["Intercept", "Intercept"])
        varbeta0_sm.append(var_beta0)

        idx = psa_mdf.resid.index
        Delta = fit_df.loc[idx, "res"].astype(float)
        evid = fit_df.loc[idx, "evid"].astype(str)

        dW_row = psa_mdf.resid.astype(float)

        re_dict = {
            str(k): float(np.asarray(v, dtype=float).ravel()[0])
            for k, v in psa_mdf.random_effects.items()
        }
        dbe_re = pd.Series(re_dict, dtype=float)

        dBe_total_row = beta0 + evid.map(dbe_re)

        if dBe_total_row.isna().any():
            missing = evid[dBe_total_row.isna()].unique()
            raise ValueError(f"{col}: missing event random effects for events: {missing[:10]}")

        n_i = evid.value_counts().astype(float)

        se_dbe_cond = None
        if use_conditional_re_se:
            se_dbe_cond = get_event_re_se(psa_mdf)

        if (
            se_dbe_cond is not None
            and set(n_i.index).issubset(set(se_dbe_cond.index.astype(str)))
        ):
            se_dbe_cond.index = se_dbe_cond.index.astype(str)
            se_dBe = np.sqrt(se_dbe_cond.reindex(n_i.index) ** 2 + var_beta0)
            se_source_dbe = "conditional_re_cov"
        else:
            se_eta = np.sqrt(
                (tau_hat**2 * phi_hat**2) / (phi_hat**2 + n_i * tau_hat**2)
            )
            se_dBe = np.sqrt(se_eta**2 + var_beta0)
            se_source_dbe = "approx_shrinkage"

        se_source_dbe_sm.append(se_source_dbe)

        if check_reconstruction:
            recon = dBe_total_row + dW_row
            max_abs_diff = float(np.max(np.abs(Delta.to_numpy() - recon.to_numpy())))

            if max_abs_diff > 1e-8:
                raise ValueError(
                    f"{col}: event-only residual decomposition does not reconstruct Delta. "
                    f"max_abs_diff={max_abs_diff:.3e}"
                )

        original_idx = fit_df.loc[idx].index

        im_dw_sm.loc[original_idx, col] = dW_row.to_numpy()
        im_dwbar_sm.loc[original_idx, col] = dW_row.to_numpy() / phi_hat

        im_dbe_sm.loc[original_idx, col] = dBe_total_row.to_numpy()
        im_dbebar_sm.loc[original_idx, col] = dBe_total_row.to_numpy() / tau_hat

        im_dbe_se_sm.loc[original_idx, col] = evid.map(se_dBe).to_numpy()

    phi_tau_df = pd.DataFrame({
        "T_label": pSAobs_T,
        "phi": phi_sm,
        "tau": tau_sm,
        "beta0": beta0_sm,
        "var_beta0": varbeta0_sm,
        "se_source_dbe": se_source_dbe_sm,
    })

    return (
        im_dwbar_sm,
        im_dbebar_sm,
        im_dw_sm,
        im_dbe_sm,
        im_dbe_se_sm,
        phi_tau_df,
    )

def lmerfit_ds2s(
    im_obs,
    im_sim,
    pSAobs_T,
    show_progress: bool = True,
    progress_desc: str = "3-component mixed model",
    use_conditional_re_se: bool = True,
    check_reconstruction: bool = True,
):
    """
    Estimate event, site-to-site, and remaining within-event residual components
    using a crossed random-intercept mixed model.

    For each IM/frequency column, define the total residual

        Delta_es = ln(Y_obs,es) - ln(Y_med,es)

    and fit

        Delta_es = beta0 + deltaB_e + deltaS2S_s + deltaW0_es,

    where

        beta0       : fixed intercept / model bias,
        deltaB_e    : event random effect,       Var(deltaB_e)    = tau^2,
        deltaS2S_s  : site random effect,        Var(deltaS2S_s)  = phi_s2s^2,
        deltaW0_es  : remaining residual term,   Var(deltaW0_es)  = phi_w0^2.

    Inputs
    ------
    im_obs : pandas.DataFrame
        Observed IM/FAS table. Must contain:
        - event-ID column named 'evid'
        - station-ID column named 'sta'
        - all IM/FAS columns listed in `pSAobs_T`

    im_sim : pandas.DataFrame
        Median-predicted IM/FAS table. Must have the same row order as `im_obs`
        and must contain:
        - event-ID column named 'evid'
        - station-ID column named 'sta'
        - all IM/FAS columns listed in `pSAobs_T`

        The prediction columns are interpreted as positive median predictions,
        not logarithmic predictions.

    pSAobs_T : list of str
        Names of the IM/FAS columns to process.

    show_progress : bool, default True
        If True, display a tqdm progress bar.

    progress_desc : str, default "3-component mixed model"
        Progress-bar label.

    use_conditional_re_se : bool, default True
        If True, attempt to use statsmodels' conditional random-effect
        covariance, `random_effects_cov`, for event/site random-effect SEs.
        If unavailable, fall back to approximate shrinkage-style SEs.

    check_reconstruction : bool, default True
        If True, check that

            Delta_es = (beta0 + deltaB_e) + deltaS2S_s + deltaW0_es

        and that the manually computed deltaW0_es equals the conditional
        residual returned by MixedLM.

    Returns
    -------
    im_dw0_sm : pandas.DataFrame
        Raw remaining within-event residuals:

            deltaW0_es.

    im_ds2s_sm : pandas.DataFrame
        Site-to-site random effects broadcast to rows:

            deltaS2S_s.

    im_dbe_sm : pandas.DataFrame
        Event total terms broadcast to rows:

            beta0 + deltaB_e.

        This includes the fixed intercept beta0. For the zero-mean stochastic
        event term alone, subtract beta0 using `var_tbl`.

    im_dbe_se_sm : pandas.DataFrame
        Standard error of beta0 + deltaB_e, broadcast to rows.

    im_ds2s_se_sm : pandas.DataFrame
        Standard error of deltaS2S_s, broadcast to rows.

    im_dw0_se_sm : pandas.DataFrame
        Residual standard deviation phi_w0, broadcast to rows.

    event_tbl_dbe : pandas.DataFrame
        Event-level beta0 + deltaB_e by period/frequency.

    event_tbl_dbe_se : pandas.DataFrame
        Event-level standard errors for beta0 + deltaB_e.

    site_tbl_ds2s : pandas.DataFrame
        Site-level deltaS2S_s by period/frequency.

    site_tbl_ds2s_se : pandas.DataFrame
        Site-level standard errors for deltaS2S_s.

    var_tbl : pandas.DataFrame
        One row per processed column with:
        - T_label
        - tau
        - phi_s2s
        - phi_w0
        - beta0
        - var_beta0
        - se_source_dbe
        - se_source_ds2s

    Notes
    -----
    This is the preferred residual decomposition for Stafford-style
    event/site/within modelling:

        deltaB_e(f)     : event term, use im_dbe_sm - beta0
        deltaS2S_s(f)   : repeatable site term, use im_ds2s_sm
        deltaW0_es(f)   : remaining within-event term, use im_dw0_sm

    The current output convention stores beta0 + deltaB_e in `im_dbe_sm` and
    `event_tbl_dbe` for compatibility with the previous workflow.
    """

    required_cols = {"evid", "sta", *pSAobs_T}
    missing_obs = sorted(required_cols.difference(im_obs.columns))
    missing_sim = sorted(required_cols.difference(im_sim.columns))

    if missing_obs:
        raise ValueError(f"im_obs is missing required columns: {missing_obs}")
    if missing_sim:
        raise ValueError(f"im_sim is missing required columns: {missing_sim}")

    if len(im_obs) != len(im_sim):
        raise ValueError("im_obs and im_sim must have the same number of rows.")

    def extract_randeffects(s: pd.Series, kind: str) -> pd.Series:
        pat = rf"^{kind}\[C\({kind}\)\[(.+?)\]\]$"
        mask = s.index.to_series().str.match(pat)
        ss = s[mask].copy()
        ss.index = ss.index.to_series().str.replace(pat, r"\1", regex=True)
        return ss.astype(float)

    def get_vcomp_map(result, fallback_names):
        names = getattr(result.model.exog_vc, "names", None)
        names = list(names) if names is not None else list(fallback_names)

        vals = np.asarray(result.vcomp, dtype=float).ravel()

        if len(names) != len(vals):
            names = list(fallback_names)

        return dict(zip(names, vals))

    def get_conditional_re_se(result, kind):
        re_cov = getattr(result, "random_effects_cov", None)

        if re_cov is None or not isinstance(re_cov, dict) or len(re_cov) == 0:
            return None

        group_key = next(iter(re_cov))
        cov = re_cov[group_key]
        re_series = result.random_effects[group_key]

        if isinstance(cov, pd.DataFrame):
            diag = pd.Series(
                np.sqrt(np.maximum(np.diag(cov.to_numpy(dtype=float)), 0.0)),
                index=cov.index,
            )
        else:
            cov = np.asarray(cov, dtype=float)
            diag = pd.Series(
                np.sqrt(np.maximum(np.diag(cov), 0.0)),
                index=re_series.index,
            )

        return extract_randeffects(diag, kind)

    im_dw0_sm = im_sim.copy()
    im_ds2s_sm = im_sim.copy()
    im_dbe_sm = im_sim.copy()
    im_dbe_se_sm = im_sim.copy()
    im_ds2s_se_sm = im_sim.copy()
    im_dw0_se_sm = im_sim.copy()

    # Residual/RE quantities are undefined for rows excluded from a fit.
    # Preserve the metadata columns from im_sim but initialize all processed
    # IM columns to NaN so invalid rows can never retain simulated IM values.
    for df in (
            im_dw0_sm,
            im_ds2s_sm,
            im_dbe_sm,
            im_dbe_se_sm,
            im_ds2s_se_sm,
            im_dw0_se_sm,
    ):
        df.loc[:, pSAobs_T] = np.nan

    phis2s_sm = []
    phiw0_sm = []
    tau_sm = []
    beta0_sm = []
    varbeta0_sm = []
    se_source_dbe_sm = []
    se_source_ds2s_sm = []

    event_tbl_dbe = pd.DataFrame()
    event_tbl_dbe_se = pd.DataFrame()
    site_tbl_ds2s = pd.DataFrame()
    site_tbl_ds2s_se = pd.DataFrame()

    iterator = range(len(pSAobs_T))
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc)

    for i in iterator:
        period_col = pSAobs_T[i]

        y_obs = np.asarray(im_obs[period_col], dtype=float)
        y_sim = np.asarray(im_sim[period_col], dtype=float)

        valid = (
            np.isfinite(y_obs)
            & np.isfinite(y_sim)
            & (y_obs > 0.0)
            & (y_sim > 0.0)
            & im_obs["evid"].notna().to_numpy()
            & im_obs["sta"].notna().to_numpy()
        )

        fit_df = pd.DataFrame({
            "evid": im_obs.loc[valid, "evid"].astype(str).to_numpy(),
            "sta": im_obs.loc[valid, "sta"].astype(str).to_numpy(),
            "res": np.log(y_obs[valid]) - np.log(y_sim[valid]),
            "groups": 1,
        }, index=im_obs.index[valid])

        vcf = {"evid": "0 + C(evid)", "sta": "0 + C(sta)"}

        psa_mdf = smf.mixedlm(
            "res ~ 1",
            fit_df,
            re_formula="0",
            vc_formula=vcf,
            groups=fit_df["groups"],
        ).fit(reml=True)

        idx = psa_mdf.resid.index

        Delta = fit_df.loc[idx, "res"].astype(float)
        evid = fit_df.loc[idx, "evid"].astype(str)
        sta = fit_df.loc[idx, "sta"].astype(str)

        beta0 = float(psa_mdf.fe_params["Intercept"])
        beta0_sm.append(beta0)

        var_beta0 = float(psa_mdf.cov_params().loc["Intercept", "Intercept"])
        varbeta0_sm.append(var_beta0)

        group_key = next(iter(psa_mdf.random_effects))
        re_series = psa_mdf.random_effects[group_key]

        dbe_re = extract_randeffects(re_series, "evid")
        ds2s_re = extract_randeffects(re_series, "sta")

        dBe_row = beta0 + evid.map(dbe_re)
        dS2S_row = sta.map(ds2s_re)
        dW0_row = Delta - dBe_row - dS2S_row

        if dBe_row.isna().any():
            missing = evid[dBe_row.isna()].unique()
            raise ValueError(f"{period_col}: missing event random effects: {missing[:10]}")

        if dS2S_row.isna().any():
            missing = sta[dS2S_row.isna()].unique()
            raise ValueError(f"{period_col}: missing site random effects: {missing[:10]}")

        vc_map = get_vcomp_map(psa_mdf, vcf.keys())

        tau2_dbe = float(vc_map.get("evid", np.nan))
        phi2_s2s = float(vc_map.get("sta", np.nan))
        phi2_dw0 = float(psa_mdf.scale)

        tau_dbe = np.sqrt(tau2_dbe) if np.isfinite(tau2_dbe) else np.nan
        phi_s2s = np.sqrt(phi2_s2s) if np.isfinite(phi2_s2s) else np.nan
        phi_dw0 = np.sqrt(phi2_dw0) if np.isfinite(phi2_dw0) else np.nan

        tau_sm.append(tau_dbe)
        phis2s_sm.append(phi_s2s)
        phiw0_sm.append(phi_dw0)

        n_i = evid.value_counts().astype(float)
        m_j = sta.value_counts().astype(float)

        se_dbe_cond = None
        se_ds2s_cond = None

        if use_conditional_re_se:
            se_dbe_cond = get_conditional_re_se(psa_mdf, "evid")
            se_ds2s_cond = get_conditional_re_se(psa_mdf, "sta")

        if (
            se_dbe_cond is not None
            and set(n_i.index).issubset(set(se_dbe_cond.index.astype(str)))
        ):
            se_dbe_cond.index = se_dbe_cond.index.astype(str)
            se_dBe = np.sqrt(se_dbe_cond.reindex(n_i.index) ** 2 + var_beta0)
            se_source_dbe = "conditional_re_cov"
        else:
            se_eta = np.sqrt(
                (tau2_dbe * phi2_dw0) / (phi2_dw0 + n_i * tau2_dbe)
            )
            se_dBe = np.sqrt(se_eta**2 + var_beta0)
            se_source_dbe = "approx_shrinkage"

        if (
            se_ds2s_cond is not None
            and set(m_j.index).issubset(set(se_ds2s_cond.index.astype(str)))
        ):
            se_ds2s_cond.index = se_ds2s_cond.index.astype(str)
            se_s2s = se_ds2s_cond.reindex(m_j.index)
            se_source_ds2s = "conditional_re_cov"
        else:
            se_s2s = np.sqrt(
                (phi2_s2s * phi2_dw0) / (phi2_dw0 + m_j * phi2_s2s)
            )
            se_source_ds2s = "approx_shrinkage"

        se_source_dbe_sm.append(se_source_dbe)
        se_source_ds2s_sm.append(se_source_ds2s)

        if check_reconstruction:
            recon = dBe_row + dS2S_row + dW0_row
            max_abs_recon = float(np.max(np.abs(Delta.to_numpy() - recon.to_numpy())))

            if max_abs_recon > 1e-8:
                raise ValueError(
                    f"{period_col}: 3-component decomposition does not reconstruct Delta. "
                    f"max_abs_diff={max_abs_recon:.3e}"
                )

            fitted_resid = psa_mdf.resid.astype(float).reindex(idx)
            max_abs_resid = float(
                np.max(np.abs(dW0_row.to_numpy() - fitted_resid.to_numpy()))
            )

            if max_abs_resid > 1e-8:
                raise ValueError(
                    f"{period_col}: dW0 does not match MixedLM residuals. "
                    f"max_abs_diff={max_abs_resid:.3e}"
                )

        original_idx = fit_df.loc[idx].index

        im_dbe_sm.loc[original_idx, period_col] = dBe_row.to_numpy()
        im_ds2s_sm.loc[original_idx, period_col] = dS2S_row.to_numpy()
        im_dw0_sm.loc[original_idx, period_col] = dW0_row.to_numpy()

        im_dbe_se_sm.loc[original_idx, period_col] = evid.map(se_dBe).to_numpy()
        im_ds2s_se_sm.loc[original_idx, period_col] = sta.map(se_s2s).to_numpy()
        im_dw0_se_sm.loc[original_idx, period_col] = phi_dw0

        event_period = pd.DataFrame({
            "evid": n_i.index,
            "n": n_i.reindex(n_i.index).to_numpy(),
            period_col: (dbe_re + beta0).reindex(n_i.index).to_numpy(),
        })

        event_period_se = pd.DataFrame({
            "evid": n_i.index,
            "n": n_i.reindex(n_i.index).to_numpy(),
            period_col: se_dBe.reindex(n_i.index).to_numpy(),
        })

        if event_tbl_dbe.empty:
            event_tbl_dbe = event_period
            event_tbl_dbe_se = event_period_se
        else:
            event_tbl_dbe = pd.merge(
                event_tbl_dbe,
                event_period.drop(columns=["n"]),
                on="evid",
                how="outer",
            )
            event_tbl_dbe_se = pd.merge(
                event_tbl_dbe_se,
                event_period_se.drop(columns=["n"]),
                on="evid",
                how="outer",
            )

        site_period = pd.DataFrame({
            "sta": m_j.index,
            "m": m_j.reindex(m_j.index).to_numpy(),
            period_col: ds2s_re.reindex(m_j.index).to_numpy(),
        })

        site_period_se = pd.DataFrame({
            "sta": m_j.index,
            "m": m_j.reindex(m_j.index).to_numpy(),
            period_col: se_s2s.reindex(m_j.index).to_numpy(),
        })

        if site_tbl_ds2s.empty:
            site_tbl_ds2s = site_period
            site_tbl_ds2s_se = site_period_se
        else:
            site_tbl_ds2s = pd.merge(
                site_tbl_ds2s,
                site_period.drop(columns=["m"]),
                on="sta",
                how="outer",
            )
            site_tbl_ds2s_se = pd.merge(
                site_tbl_ds2s_se,
                site_period_se.drop(columns=["m"]),
                on="sta",
                how="outer",
            )

    var_tbl = pd.DataFrame({
        "T_label": pSAobs_T,
        "tau": tau_sm,
        "phi_s2s": phis2s_sm,
        "phi_w0": phiw0_sm,
        "beta0": beta0_sm,
        "var_beta0": varbeta0_sm,
        "se_source_dbe": se_source_dbe_sm,
        "se_source_ds2s": se_source_ds2s_sm,
    })

    return (
        im_dw0_sm,
        im_ds2s_sm,
        im_dbe_sm,
        im_dbe_se_sm,
        im_ds2s_se_sm,
        im_dw0_se_sm,
        event_tbl_dbe,
        event_tbl_dbe_se,
        site_tbl_ds2s,
        site_tbl_ds2s_se,
        var_tbl,
    )

# def lmerfit_ds2s(
#         im_obs,
#         im_sim,
#         pSAobs_T,
#         show_progress: bool = True,
#         progress_desc: str = "3-component mixed model",
# ):
#     """
#     Decompose residuals into:
#       dBe  (between-event)
#       dS2S (site-to-site)
#       dW0  (remaining within-event)
#
#     Returns
#     -------
#     im_dw0_sm, im_ds2s_sm, im_dbe_sm,
#     im_dbe_se_sm, im_ds2s_se_sm, im_dw0_se_sm,
#     event_tbl_dbe, event_tbl_dbe_se,
#     site_tbl_ds2s, site_tbl_ds2s_se,
#     var_tbl : DataFrame with columns
#         ['T_label', 'tau', 'phi_s2s', 'phi_w0', 'beta0', 'var_beta0']
#     """
#
#     def extract_randeffects(s: pd.Series, kind: str) -> pd.Series:
#         pat = rf"^{kind}\[C\({kind}\)\[(.+?)\]\]$"
#         mask = s.index.to_series().str.match(pat)
#         ss = s[mask]
#         # set index to the captured ID (event id or station code)
#         ss.index = ss.index.to_series().str.replace(pat, r"\1", regex=True)
#         return ss.astype(float)
#
#     # dataframes for storage
#     im_dw0_sm = im_sim.copy()
#     im_ds2s_sm = im_sim.copy()
#     im_dbe_sm = im_sim.copy()
#     im_dbe_se_sm = im_sim.copy()
#     im_ds2s_se_sm = im_sim.copy()
#     im_dw0_se_sm = im_sim.copy()
#     im_residual = im_sim.copy()
#
#     phis2s_sm = []
#     phiw0_sm = []
#     tau_sm = []
#     beta0_sm = []  # bias/intercept
#     varbeta0_sm = []  # var(Intercept)
#
#     # summary dataframes
#     event_tbl_dbe = pd.DataFrame()
#     event_tbl_dbe_se = pd.DataFrame()
#     site_tbl_ds2s = pd.DataFrame()
#     site_tbl_ds2s_se = pd.DataFrame()
#
#     # prepare iterator
#     iterator = range(len(pSAobs_T))
#     if show_progress:
#         iterator = tqdm(iterator, desc=progress_desc)
#
#     for i in iterator:
#         period_col = pSAobs_T[i]
#
#         # residuals: log(obs) - log(pred)
#         im_residual[f"res_{i:d}"] = np.log(im_obs[period_col]) - np.log(im_sim[period_col])
#         im_residual["groups"] = 1  # single grouping factor; random effects via vc_formula
#
#         # variance components formula
#         vcf = {"evid": "0 + C(evid)", "sta": "0 + C(sta)"}
#
#         # mixed model: random intercepts for evid & sta
#         psa_mdf = smf.mixedlm(
#             f"res_{i:d} ~ 1",
#             im_residual,
#             re_formula="0",
#             vc_formula=vcf,
#             groups=im_residual["groups"],
#         ).fit(reml=True)
#
#         # rows actually used in fit (important if NaNs)
#         idx = psa_mdf.resid.index
#         Delta = im_residual.loc[idx, f"res_{i:d}"]
#         evid = im_residual.loc[idx, "evid"]
#         sta = im_residual.loc[idx, "sta"]
#
#         # bias
#         beta0 = float(psa_mdf.fe_params["Intercept"])
#         beta0_sm.append(beta0)
#
#         # random effects -> dBe (event) and dS2S (site)
#         re_series = psa_mdf.random_effects[next(iter(psa_mdf.random_effects))]
#         dbe_sm = extract_randeffects(re_series, "evid")
#         ds2s_sm = extract_randeffects(re_series, "sta")
#         dw0_sm = psa_mdf.resid.astype(float)
#
#         dBe_row = beta0 + evid.map(dbe_sm)  # between-event (with bias)
#         dS2S_row = sta.map(ds2s_sm).fillna(0.0)  # site-to-site
#         dW0_row = Delta - dBe_row - dS2S_row  # remaining; same as dw0_sm
#
#         # write back (only rows used in fit)
#         im_dbe_sm.loc[idx, period_col] = dBe_row.to_numpy()
#         im_ds2s_sm.loc[idx, period_col] = dS2S_row.to_numpy()
#         im_dw0_sm.loc[idx, period_col] = dW0_row.to_numpy()
#
#         # variance components
#         vc_names = list(vcf.keys())  # ['evid', 'sta']
#         vc_vals = np.asarray(psa_mdf.vcomp, float).ravel()
#         vc_map = dict(zip(vc_names, vc_vals))
#
#         tau2_dbe = float(vc_map.get("evid", np.nan))  # Var(dBe)  = τ^2
#         phi2_s2s = float(vc_map.get("sta", np.nan))  # Var(dS2S) = φ_S2S^2
#         phi2_dw0 = float(psa_mdf.scale)  # Var(dW0)  = φ_W0^2
#
#         tau_dbe = np.sqrt(tau2_dbe) if np.isfinite(tau2_dbe) else np.nan
#         tau_sm.append(tau_dbe)
#
#         phi_s2s = np.sqrt(phi2_s2s) if np.isfinite(phi2_s2s) else np.nan
#         phis2s_sm.append(phi_s2s)
#
#         phi_dw0 = np.sqrt(phi2_dw0)
#         phiw0_sm.append(phi_dw0)
#
#         # standard errors
#         var_beta0 = float(psa_mdf.cov_params().loc["Intercept", "Intercept"])
#         varbeta0_sm.append(var_beta0)
#
#         n_i = evid.value_counts().astype(float)  # event counts
#         m_j = sta.value_counts().astype(float)  # site counts
#
#         se_eta = np.sqrt((tau2_dbe * phi2_dw0) / (phi2_dw0 + n_i * tau2_dbe))
#         se_dBe = np.sqrt(se_eta ** 2 + var_beta0)
#         se_s2s = np.sqrt((phi2_s2s * phi2_dw0) / (phi2_dw0 + m_j * phi2_s2s))
#
#         im_dbe_se_sm.loc[idx, period_col] = evid.map(se_dBe).to_numpy()
#         im_ds2s_se_sm.loc[idx, period_col] = sta.map(se_s2s).to_numpy()
#         im_dw0_se_sm.loc[idx, period_col] = phi_dw0  # scalar, broadcast
#
#         # summary tables (event-level)
#         if event_tbl_dbe.empty:
#             event_tbl_dbe["evid"] = n_i.index
#             event_tbl_dbe["n"] = n_i.values
#             event_tbl_dbe_se["evid"] = n_i.index
#             event_tbl_dbe_se["n"] = n_i.values
#
#         if site_tbl_ds2s.empty:
#             site_tbl_ds2s["sta"] = m_j.index
#             site_tbl_ds2s["m"] = m_j.values
#             site_tbl_ds2s_se["sta"] = m_j.index
#             site_tbl_ds2s_se["m"] = m_j.values
#
#         # period-specific columns
#         event_tbl_dbe[period_col] = (dbe_sm + beta0).reindex(n_i.index).values
#         event_tbl_dbe_se[period_col] = se_dBe.reindex(n_i.index).values
#
#         site_tbl_ds2s[period_col] = ds2s_sm.reindex(m_j.index).values
#         site_tbl_ds2s_se[period_col] = se_s2s.reindex(m_j.index).values
#
#     # variance summary
#     var_tbl = pd.DataFrame({
#         "T_label": pSAobs_T,
#         "tau": tau_sm,
#         "phi_s2s": phis2s_sm,
#         "phi_w0": phiw0_sm,
#         "beta0": beta0_sm,
#         "var_beta0": varbeta0_sm,
#     })
#
#     return (
#         im_dw0_sm,
#         im_ds2s_sm,
#         im_dbe_sm,
#         im_dbe_se_sm,
#         im_ds2s_se_sm,
#         im_dw0_se_sm,
#         event_tbl_dbe,
#         event_tbl_dbe_se,
#         site_tbl_ds2s,
#         site_tbl_ds2s_se,
#         var_tbl,
#     )

