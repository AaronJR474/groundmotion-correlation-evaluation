"""Station-pair empirical correlations, site dissimilarities and LOWESS utilities."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

# <distance metrics>
def safe_log_zero_as_zero(x):
    """
    Log transform with zero mapped to zero.

    Positive values -> log(x)
    Zero values     -> 0
    Negative values -> nan, because log is undefined
    """
    x = np.asarray(x, dtype=float)

    out = np.full_like(x, np.nan, dtype=float)
    out[x == 0.0] = 0.0
    out[x > 0.0] = np.log(x[x > 0.0])

    return out

# Angular distance given angle relative to earthquake
def getAngDist(theta1,theta2):
    cos_angle = np.cos(np.abs(theta1-theta2))
    distA = np.arccos(np.clip(cos_angle,-1,1))
    return distA

# Dissimilarity as a function of the absolute difference (applicable to vs30, Z1.0 and TRI)
def getDiss(x1,x2):
    distS = np.abs(x1-x2)
    return distS

def getlnDiss(x1, x2):
    """
    Natural-logarithmic dissimilarity:
        |log(x1) - log(x2)|

    with zero values mapped to zero before differencing.
    """
    lx1 = safe_log_zero_as_zero(x1)
    lx2 = safe_log_zero_as_zero(x2)

    dist = np.abs(lx1 - lx2)
    return dist

# Circular Median for angular distance
def circmedian(angles, period=2*np.pi):
    """
    Compute the circular median of angles in radians on [0, 2π).
    """
    angles = np.asarray(angles, dtype=float)

    if angles.size == 0:
        return np.nan

    angles = np.sort(np.mod(angles, period))

    diff = np.abs(angles[:, None] - angles[None, :])
    total_deviation = np.minimum(diff, period - diff).sum(axis=1)

    return angles[np.argmin(total_deviation)]

def _circmedian_rows(values, valid, period=2*np.pi, row_block_size=2048):
    """
    Compute circular medians row-wise for pair-by-event arrays.

    This is the vectorized analogue of `circmedian()`. Only entries marked
    True in `valid` participate in each row's circular median.

    The implementation retains the candidate-point definition and tie-breaking
    behaviour of `circmedian()`. Rows with numerically tied candidate minima
    are evaluated using the scalar implementation to preserve backward
    compatibility.

    Parameters
    ----------
    values : array_like, shape (n_rows, n_events)
        Angular values in radians.

    valid : array_like of bool, shape (n_rows, n_events)
        Mask defining the observations used in each row.

    period : float, default 2*pi
        Circular period.

    row_block_size : int, default 2048
        Maximum nominal number of rows processed simultaneously. The actual
        block size is reduced automatically for large event dimensions to
        limit temporary memory use.

    Returns
    -------
    medians : ndarray, shape (n_rows,)
        Circular median for each row.
    """
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)

    if values.shape != valid.shape:
        raise ValueError("values and valid must have the same shape.")

    if values.ndim != 2:
        raise ValueError("values and valid must be two-dimensional.")

    n_rows, n_events = values.shape
    out = np.full(n_rows, np.nan, dtype=float)

    if n_rows == 0 or n_events == 0:
        return out

    # Limit the n_rows x n_events x n_events temporary arrays.
    max_rows_memory = max(1, 4_000_000 // max(n_events * n_events, 1))
    block_size = min(int(row_block_size), max_rows_memory)

    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)

        v = values[start:stop]
        m = valid[start:stop]

        a = np.where(m, np.mod(v, period), np.nan)
        a = np.sort(a, axis=1)

        finite = np.isfinite(a)

        diff = np.abs(a[:, :, None] - a[:, None, :])
        dist = np.minimum(diff, period - diff)
        dist = np.where(finite[:, :, None] & finite[:, None, :], dist, 0.0)

        total = dist.sum(axis=2)
        total[~finite] = np.inf

        idx = np.argmin(total, axis=1)
        row_has = finite.any(axis=1)

        block_out = np.full(stop - start, np.nan, dtype=float)
        rr = np.arange(stop - start)

        block_out[row_has] = a[rr[row_has], idx[row_has]]

        # Preserve legacy scalar tie-breaking exactly where floating-point
        # summation makes two circular-median candidates effectively equal.
        if n_events > 1:
            sorted_total = np.sort(total, axis=1)
            near_tie = (
                row_has
                & np.isclose(
                    sorted_total[:, 0],
                    sorted_total[:, 1],
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        else:
            near_tie = np.zeros(stop - start, dtype=bool)

        # Legacy circmedian() does not explicitly remove non-finite azimuths,
        # so use it directly for these uncommon rows.
        nonfinite_angle = np.any(m & ~np.isfinite(v), axis=1)

        for r in np.flatnonzero(near_tie | nonfinite_angle):
            block_out[r] = circmedian(v[r, m[r]], period=period)

        out[start:stop] = block_out

    return out

# Circular Mean for angular distance
def circmean(angles, period=2*np.pi):
    """
    Compute the circular mean of angles in radians on [0, 2π)
    """
    angles = np.mod(angles, period)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum) % period

def pearsonr_uncen(value1, value2):
    """
    Compute uncentered Pearson correlation (cosine similarity).

    The coefficient is

        rho_u = sum(x_i y_i) / sqrt(sum(x_i^2) sum(y_i^2)),

    without subtracting either vector mean.

    Parameters
    ----------
    value1, value2 : array_like, shape (n,)
        Elementwise paired vectors. Non-finite pairs are removed internally.

    Returns
    -------
    rho : float
        Uncentered correlation in ``[-1, 1]``. Returns ``np.nan`` when fewer
        than three finite pairs remain or either filtered vector has zero
        Euclidean norm.

    Notes
    -----
    Unlike centered Pearson correlation, a nonzero constant vector is valid
    because cosine similarity depends on vector direction rather than variance.
    """
    x = np.asarray(value1, dtype=float).ravel()
    y = np.asarray(value2, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError("value1 and value2 must have the same shape.")

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if x.size < 3:
        return np.nan

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan

    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def pearsonr_cen(value1, value2):
    """
    Compute the centered Pearson correlation coefficient.

    Parameters
    ----------
    value1, value2 : array_like, shape (n,)
        Elementwise paired vectors. Non-finite pairs are removed internally.

    Returns
    -------
    rho : float
        Pearson correlation in ``[-1, 1]``. Returns ``np.nan`` when fewer than
        three finite pairs remain or either filtered vector has zero variance.
    """
    x = np.asarray(value1, dtype=float).ravel()
    y = np.asarray(value2, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError("value1 and value2 must have the same shape.")

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if x.size < 3:
        return np.nan

    x = x - x.mean()
    y = y - y.mean()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan

    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))

EARTH_R_KM = 6371.0088
# source: https://docs.openquake.org/oq-engine/2.7/_modules/openquake/hmtk/seismicity/utils.html#haversine
def haversine_oq(lon1, lat1, lon2, lat2, radians=False, earth_rad=EARTH_R_KM):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if not radians:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
                                            (np.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval),
                                                      np.sqrt(1 - aval))).T
        i += 1
    return distance

def haversine_vec(lat1, lon1, lat2, lon2, earth_rad=EARTH_R_KM):
    """
    Vectorized haversine distance using the arctan2 form
    (numerically robust; consistent with OpenQuake).

    Inputs in degrees. lat/lon can be scalars or 1D arrays.
    If arrays, they should be broadcastable to the same shape.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    φ1 = np.deg2rad(lat1)
    φ2 = np.deg2rad(lat2)
    dφ = φ2 - φ1
    dλ = np.deg2rad(lon2 - lon1)

    a = np.sin(dφ * 0.5)**2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ * 0.5)**2

    # Clamp a to avoid tiny FP issues outside [0,1]
    a = np.clip(a, 0.0, 1.0)

    # OpenQuake-style stable form:
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_rad * c
# <distance metrics>

# <Cosine Similarity/Uncentered correlation coefficient - all station pairs>

def cossim(im_sim_all, nj_lim, hmax=None, ref_sta=None, sta_i=None, sta_j=None, mode = "uncen", show_progress = False):
    """
    Optimized CBB21-style correlation cloud builder.

    Key rules:
    - Station IDs are opaque string identifiers.
    - Leading zeros are part of the TRUE ID.
    - We NEVER ignore or strip leading zeros for identity matching.
    - ref_sta must match EXACTLY a station ID present in the dataframe.

    Performance:
    - Uses dense arrays (stations x events) for dw_bar and epicenter_azimuth.
    - Streams station pairs to avoid large intermediate lists.
    - Uses BallTree pruning only if hmax is finite.

    Parameters
    ----------
    im_sim_all : pd.DataFrame
        Must contain columns:
        ['sta','evid','dw_bar','epicenter_azimuth',
         'sta_lat','sta_lon','vs30','Z1.0','TRI']

    nj_lim : int
        Minimum number of overlapping events required.

    hmax : float or None
        Maximum station separation (km). If None or non-finite, no distance pruning.

    ref_sta : any
        Reference station ID. Converted to string and matched EXACTLY.
        If provided, only pairs involving this station are computed.

    sta_i: list
        List of stations for subset i

    sta_j: list
        List of stations for subset j

    show_progress: bool
        Show progress bar

    Selection modes
    ---------------
    1) Full network (default):
        cossim(df, nj_lim, hmax=None)
       -> all unique station pairs (optionally distance-pruned by hmax)

    2) Single reference station:
        cossim(df, nj_lim, hmax, ref_sta="0200cca")
       -> only pairs (ref_sta, other_sta)

    3) Subset cross-pairs:
        cossim(df, nj_lim, hmax, sta_i=stations_A, sta_j=stations_B)
       -> all pairs (s_i, s_j) with s_i in stations_A, s_j in stations_B

    Returns
    -------
    pd.DataFrame
        Columns (same as your current output):
        p_hat, z_p_hat, h, sta_j, sta_k, n_j,
        vs30_j, vs30_k, dS,
        Z1_j, Z1_k, dZ1,
        TRI_j, TRI_k, dTRI,
        theta_i, theta_j, dA,
        stat_lon_j, stat_lat_j, stat_lon_k, stat_lat_k
    """

    EARTH_R_KM = 6371.0088
    finite_hmax = (hmax is not None) and np.isfinite(hmax)

    # ---------------------------------------------------------------
    # 0) Mode sanity checks
    # ---------------------------------------------------------------
    mode = str(mode).strip().lower()
    if mode not in {"uncen", "cen"}:
        raise ValueError("mode must be either 'uncen' or 'cen'.")

    use_subset_pairs = (sta_i is not None) and (sta_j is not None)
    if (ref_sta is not None) and use_subset_pairs:
        raise ValueError("Specify either ref_sta or sta_i/sta_j, not both.")

    # ------------------------------------------------------------------
    # 1) Pull required columns and enforce sta as string (preserve zeros)
    # ------------------------------------------------------------------
    # 1) Column aliasing: allow 'station' as an alternative to 'sta'
    if 'sta' not in im_sim_all.columns:
        if 'station' in im_sim_all.columns:
            im_sim_all = im_sim_all.copy()
            im_sim_all['sta'] = im_sim_all['station']
        else:
            raise ValueError(
                "Input DataFrame must contain 'sta' or 'station' column for station IDs."
            )

    needed_cols = [
        'sta', 'evid', 'dw_bar', 'epicenter_azimuth',
        'sta_lat', 'sta_lon', 'vs30', 'Z1.0', 'TRI'
    ]

    df_all = im_sim_all[needed_cols].copy()

    # Treat IDs as authoritative strings; do NOT alter leading zeros
    df_all['sta'] = df_all['sta'].astype(str).str.strip()

    # Drop rows missing core keys
    df_all = df_all.dropna(subset=['sta', 'evid'])

    # Normalize ref_sta to string for exact matching
    ref_raw = None
    if ref_sta is not None:
        ref_raw = str(ref_sta).strip()

    # ------------------------------------------------------------------
    # 2) Dense (stations x events) arrays for fast overlap + dot products
    # ------------------------------------------------------------------
    df_key = df_all[['sta', 'evid', 'dw_bar', 'epicenter_azimuth']].copy()

    # Factorize to guaranteed integer codes
    sta_codes, stations = pd.factorize(df_key['sta'], sort=False)
    ev_codes, events    = pd.factorize(df_key['evid'], sort=False)

    S  = len(stations)
    Ne = len(events)

    dw_vals = df_key['dw_bar'].to_numpy(dtype=float)
    az_vals = df_key['epicenter_azimuth'].to_numpy(dtype=float)

    dw = np.full((S, Ne), np.nan, dtype=float)
    az = np.full((S, Ne), np.nan, dtype=float)

    # Safe integer indexing
    dw[sta_codes, ev_codes] = dw_vals
    az[sta_codes, ev_codes] = az_vals

    sta_index = {sta: i for i, sta in enumerate(stations)}

    # ------------------------------------------------------------------
    # 3) Resolve reference station (if any)
    # ------------------------------------------------------------------
    ref_idx = None
    if ref_raw is not None:
        if ref_raw in sta_index:
            ref_idx = sta_index[ref_raw]
        else:
            # Optional: suggest numeric-looking variants WITHOUT auto-matching
            suggestions = []
            if ref_raw.isdigit():
                ref_tail = ref_raw.lstrip("0") or "0"
                for s in stations:
                    s_str = str(s)
                    if s_str.isdigit():
                        s_tail = s_str.lstrip("0") or "0"
                        if s_tail == ref_tail:
                            suggestions.append(s_str)

            msg = (
                f"Reference station '{ref_raw}' not found. "
                "Leading zeros are treated as part of the TRUE station ID, "
                "so only exact matches are allowed."
            )
            if suggestions:
                suggestions = sorted(set(suggestions), key=lambda x: (len(x), x))
                msg += f" Did you mean one of: {suggestions[:10]} ?"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # 3b) Resolve subset pair lists (sta_i, sta_j) if provided
    # ------------------------------------------------------------------
    if use_subset_pairs:
        sta_i_list = [str(s).strip() for s in np.atleast_1d(sta_i)]
        sta_j_list = [str(s).strip() for s in np.atleast_1d(sta_j)]

        missing_i = [s for s in sta_i_list if s not in sta_index]
        missing_j = [s for s in sta_j_list if s not in sta_index]
        if missing_i or missing_j:
            msg = []
            if missing_i:
                msg.append(f"sta_i not found: {missing_i}")
            if missing_j:
                msg.append(f"sta_j not found: {missing_j}")
            raise ValueError("Some requested stations not found in data: "
                             + "; ".join(msg))

        idx_i_list = [sta_index[s] for s in sta_i_list]
        idx_j_list = [sta_index[s] for s in sta_j_list]
    else:
        idx_i_list = idx_j_list = None

    # ------------------------------------------------------------------
    # 4) Station metadata aligned to `stations`
    # ------------------------------------------------------------------
    meta = (
        df_all.drop_duplicates('sta')
              .set_index('sta')[['sta_lat', 'sta_lon', 'vs30', 'Z1.0', 'TRI']]
    )

    # Reindex to factorized station order
    meta = meta.reindex(stations)

    # If these are missing, distance logic cannot be trusted
    if meta[['sta_lat', 'sta_lon']].isna().any().any():
        missing = meta[meta['sta_lat'].isna() | meta['sta_lon'].isna()].index.tolist()
        raise ValueError(
            f"Missing sta_lat/sta_lon for {len(missing)} stations in the working set. "
            "Cannot compute distances reliably."
        )

    lat_arr  = meta['sta_lat'].to_numpy(dtype=float)
    lon_arr  = meta['sta_lon'].to_numpy(dtype=float)
    vs30_arr = meta['vs30'].to_numpy(dtype=float)
    Z1_arr   = meta['Z1.0'].to_numpy(dtype=float)
    TRI_arr  = meta['TRI'].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 5) BallTree neighbor pruning if finite hmax AND not in subset mode
    # ------------------------------------------------------------------
    if finite_hmax and not use_subset_pairs:
        coords_rad = np.deg2rad(np.c_[lat_arr, lon_arr])
        tree = BallTree(coords_rad, metric='haversine')
        rad = hmax / EARTH_R_KM
        nbrs = tree.query_radius(coords_rad, r=rad, return_distance=False)
    else:
        nbrs = None

    # ------------------------------------------------------------------
    # 6) Pair iterator (streaming; avoids huge lists)
    # ------------------------------------------------------------------
    if use_subset_pairs:
        # Cross-product between sta_i_list and sta_j_list
        pair_iter = ((i_idx, j_idx) for i_idx in idx_i_list for j_idx in idx_j_list)
    else:
        # Original modes: full network or single ref_sta
        if nbrs is None:
            if ref_idx is None:
                pair_iter = ((i, j) for i in range(S) for j in range(i + 1, S))
            else:
                pair_iter = ((ref_idx, j) for j in range(S) if j != ref_idx)
        else:
            if ref_idx is None:
                pair_iter = (
                    (i, j)
                    for i in range(S)
                    for j in nbrs[i]
                    if j > i
                )
            else:
                pair_iter = (
                    (ref_idx, j)
                    for j in nbrs[ref_idx]
                    if j != ref_idx
                )

    # ------------------------------------------------------------------
    # 6b) Progress bar for large pair counts
    # ------------------------------------------------------------------
    if use_subset_pairs:
        total_pairs = len(idx_i_list) * len(idx_j_list)
    else:
        if nbrs is None:
            if ref_idx is None:
                total_pairs = S * (S - 1) // 2
            else:
                total_pairs = S - 1
        else:
            if ref_idx is None:
                total_pairs = sum(
                    (np.asarray(neigh) > i).sum()
                    for i, neigh in enumerate(nbrs)
                )
            else:
                neigh = np.asarray(nbrs[ref_idx])
                total_pairs = int((neigh != ref_idx).sum())

    if show_progress:
        pair_iter = tqdm(
            pair_iter,
            total=total_pairs,
            desc="Station pairs"
        )

    # ------------------------------------------------------------------
    # 7) Accumulators
    # ------------------------------------------------------------------
    pspear = []
    dsep = []
    stat_i_id = []
    stat_j_id = []
    nj = []
    stat_vs30_i = []
    stat_vs30_j = []
    stat_Z1_i = []
    stat_Z1_j = []
    stat_TRI_i = []
    stat_TRI_j = []
    stat_lat_i = []
    stat_lon_i = []
    stat_lat_j = []
    stat_lon_j = []
    theta_i = []
    theta_j = []

    # ------------------------------------------------------------------
    # 8) Loop over station pairs
    # ------------------------------------------------------------------
    for i_idx, j_idx in pair_iter:
        sta_i_id_str = stations[i_idx]
        sta_j_id_str = stations[j_idx]

        dw_i = dw[i_idx]
        dw_j = dw[j_idx]

        valid = np.isfinite(dw_i) & np.isfinite(dw_j)
        n_j_temp = int(valid.sum())

        if n_j_temp < nj_lim:
            continue

        xi = dw_i[valid]
        xj = dw_j[valid]

        if mode == 'uncen':
            # Uncentered Pearson p_hat
            num = np.dot(xi, xj)
            den = np.sqrt(np.dot(xi, xi) * np.dot(xj, xj))
            if den == 0.0:
                continue
            p_temp = num / den
        elif mode == 'cen':
            p_temp = pearsonr_cen(xi,xj)

        # Median azimuths over overlapping events
        az_i = az[i_idx, valid]
        az_j = az[j_idx, valid]
        theta1_temp = circmedian(az_j)
        theta2_temp = circmedian(az_i)

        # Great-circle distance (precise)
        distance_temp = haversine_oq(
            np.array([lon_arr[j_idx]]), np.array([lat_arr[j_idx]]),
            np.array([lon_arr[i_idx]]), np.array([lat_arr[i_idx]])
        )[0][0]

        if distance_temp <= 0.0:
            continue
        if finite_hmax and distance_temp > hmax:
            continue

        # Append
        pspear.append(p_temp)
        dsep.append(distance_temp)
        stat_i_id.append(sta_i_id_str)
        stat_j_id.append(sta_j_id_str)
        nj.append(n_j_temp)

        stat_vs30_i.append(vs30_arr[i_idx])
        stat_vs30_j.append(vs30_arr[j_idx])
        stat_Z1_i.append(Z1_arr[i_idx])
        stat_Z1_j.append(Z1_arr[j_idx])
        stat_TRI_i.append(TRI_arr[i_idx])
        stat_TRI_j.append(TRI_arr[j_idx])

        theta_i.append(theta1_temp)
        theta_j.append(theta2_temp)

        stat_lat_i.append(lat_arr[i_idx])
        stat_lon_i.append(lon_arr[i_idx])
        stat_lat_j.append(lat_arr[j_idx])
        stat_lon_j.append(lon_arr[j_idx])

    # ------------------------------------------------------------------
    # 9) Assemble DataFrame (same columns as your current logic)
    # ------------------------------------------------------------------
    cbsparcorr = pd.DataFrame()
    cbsparcorr['p_hat'] = pspear
    cbsparcorr['h'] = dsep
    cbsparcorr['sta_j'] = stat_j_id
    cbsparcorr['sta_k'] = stat_i_id
    cbsparcorr['n_j'] = nj

    vs30_i = np.asarray(stat_vs30_i, dtype=float)
    vs30_j = np.asarray(stat_vs30_j, dtype=float)

    Z1_i = np.asarray(stat_Z1_i, dtype=float)
    Z1_j = np.asarray(stat_Z1_j, dtype=float)

    TRI_i = np.asarray(stat_TRI_i, dtype=float)
    TRI_j = np.asarray(stat_TRI_j, dtype=float)

    cbsparcorr["vs30_j"] = vs30_j
    cbsparcorr["vs30_k"] = vs30_i
    cbsparcorr["dS"] = getDiss(vs30_i, vs30_j)
    cbsparcorr["lndS"] = getlnDiss(vs30_i, vs30_j)

    cbsparcorr["Z1_j"] = Z1_j
    cbsparcorr["Z1_k"] = Z1_i
    cbsparcorr["dZ1"] = getDiss(Z1_i, Z1_j)
    cbsparcorr["lndZ1"] = getlnDiss(Z1_i, Z1_j)

    cbsparcorr["TRI_j"] = TRI_j
    cbsparcorr["TRI_k"] = TRI_i
    cbsparcorr["dTRI"] = getDiss(TRI_i, TRI_j)
    cbsparcorr["lndTRI"] = getlnDiss(TRI_i, TRI_j)

    cbsparcorr['theta_i'] = theta_i
    cbsparcorr['theta_j'] = theta_j
    cbsparcorr['dA'] = getAngDist(
        cbsparcorr.theta_i.to_numpy(),
        cbsparcorr.theta_j.to_numpy()
    ) * 180.0 / np.pi

    cbsparcorr['stat_lon_j'] = stat_lon_j
    cbsparcorr['stat_lat_j'] = stat_lat_j
    cbsparcorr['stat_lon_k'] = stat_lon_i
    cbsparcorr['stat_lat_k'] = stat_lat_i

    # Final filtering
    cbsparcorr = cbsparcorr[cbsparcorr.h > 0.0]
    if finite_hmax:
        cbsparcorr = cbsparcorr[cbsparcorr.h <= hmax]
    cbsparcorr = cbsparcorr[cbsparcorr.n_j >= nj_lim]
    cbsparcorr = cbsparcorr.sort_values('h').reset_index(drop=True)

    # Fisher transform (clip for safety)
    p = np.clip(cbsparcorr['p_hat'].to_numpy(), -1 + 1e-2, 1 - 1e-2)
    cbsparcorr["z_p_hat"] = 0.5 * np.log((1.0 + p) / (1.0 - p))

    return cbsparcorr

# <Cosine Similarity/Uncentered correlation coefficient - all station pairs>


# <LOESS Smoothing function>

def smooth(x, y, K=100, frac=0.2, it=5, xgrid=None, show_progress=True):
    """
        Bootstrap LOWESS smoothing.

        Parameters:
        - x, y: input arrays
        - K: number of bootstrap replicates
        - frac: LOWESS fraction
        - it: number of robustness iterations
        - xgrid: grid of user-defined x values else default is used i.e., np.linspace(x.min(), x.max(), 100)

        Returns:
        - xgrid: uniform x axis
        - y_grid: one example smoothed line (last iteration)
        - mean: mean of all bootstrapped smoothed values
        - stderr: standard error at each xgrid point
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if xgrid is None:
        xgrid = np.linspace(x.min(), x.max(), 50)

    smooths = []
    y_grid = None

    iterator = tqdm(range(K), desc='Bootstrapping LOWESS') if show_progress else range(K)

    for _ in iterator:
        # Faster bootstrap sampling
        samples = np.random.randint(0, len(x), size=len(x))
        x_s = x[samples]
        y_s = y[samples]

        try:
            y_sm = sm_lowess(y_s, x_s, frac=frac, it=it, return_sorted=False)

            # Sort once
            sort_idx = np.argsort(x_s)
            x_sorted = x_s[sort_idx]
            y_sorted = y_sm[sort_idx]

            # Unique x
            unique_x, unique_idx = np.unique(x_sorted, return_index=True)
            if len(unique_x) < 3:
                continue

            unique_y = y_sorted[unique_idx]

            # Much faster interpolation
            y_grid = np.interp(xgrid, unique_x, unique_y)

            smooths.append(y_grid)

        except Exception:
            continue

    if len(smooths) == 0:
        raise RuntimeError("All bootstrap LOWESS fits failed. Check data quality.")

    smooths = np.stack(smooths, axis=0)
    mean = np.nanmean(smooths, axis=0)
    stderr = np.nanstd(smooths, axis=0, ddof=0)

    return xgrid, y_grid, mean, stderr

def _lowess_eval_grid(x, y, xgrid, frac, it, delta_frac):
    """
    Evaluate LOWESS on a fixed x-grid.

    If delta_frac is zero or None, LOWESS is evaluated directly at xgrid.
    If delta_frac is positive, statsmodels' native `delta` acceleration is
    used on the observed x-values and the resulting smooth is interpolated
    onto xgrid.

    Parameters
    ----------
    x, y : array-like
        Finite predictor and response values.

    xgrid : array-like
        Increasing x-values at which the smoothed trend is required.

    frac : float
        Fraction of observations used in each local LOWESS regression.

    it : int
        Number of residual-based robustness iterations.

    delta_frac : float or None
        LOWESS delta expressed as a fraction of the observed x-range.
        Set to 0 or None for no delta approximation.

    Returns
    -------
    y_grid : ndarray
        LOWESS estimate evaluated at xgrid.
    """

    order = np.argsort(x, kind="mergesort")
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]

    if delta_frac is None or delta_frac == 0:
        return sm_lowess(
            y_sorted,
            x_sorted,
            frac=frac,
            it=it,
            xvals=xgrid,
            is_sorted=True,
            missing="none",
            return_sorted=False,
        )

    x_range = x_sorted[-1] - x_sorted[0]
    delta = float(delta_frac) * x_range

    y_smooth = sm_lowess(
        y_sorted,
        x_sorted,
        frac=frac,
        it=it,
        delta=delta,
        is_sorted=True,
        missing="none",
        return_sorted=False,
    )

    # Bootstrap sampling creates duplicate x-values. Reduce to unique
    # locations before interpolation onto the common evaluation grid.
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_smooth[unique_idx]

    return np.interp(xgrid, unique_x, unique_y)

def lowess_sim(
    x,
    y,
    K=100,
    frac=0.01,
    it=3,
    xgrid=None,
    n_grid=50,
    delta_frac=5e-4,
    bootstrap_size=100_000,
    random_state=None,
    show_progress=True,
    return_details=False,
):
    """
    Compute LOWESS smoothing for very large simulation datasets.

    This function adapts the standard statsmodels LOWESS algorithm for large,
    continuous simulation datasets while retaining the underlying local-linear
    LOWESS formulation.

    The primary smooth is always estimated from all finite observations.
    Computational acceleration is obtained using statsmodels' native `delta`
    option rather than binning or aggregation of the predictor.

    Bootstrap uncertainty can optionally be estimated from repeated pairwise
    resamples. For very large datasets, `bootstrap_size` allows each bootstrap
    replicate to contain fewer observations than the full dataset, greatly
    reducing computational cost while leaving the primary LOWESS estimate
    unchanged.

    Parameters
    ----------
    x, y : array-like
        Predictor and response values. Both must be one-dimensional and have
        equal length. Non-finite x-y pairs are removed before fitting.

    K : int, default 100
        Number of bootstrap LOWESS replicates. Set K=0 to calculate only the
        full-data LOWESS trend.

    frac : float, default 0.01
        Fraction of observations used in each local LOWESS regression.
        The local neighbourhood therefore contains approximately
        `frac * n` observations.

    it : int, default 3
        Number of residual-based robustness iterations used by LOWESS.

    xgrid : array-like or None, default None
        Increasing x-values at which the LOWESS trend is evaluated. If None,
        `n_grid` equally spaced values spanning the finite range of x are used.

    n_grid : int, default 50
        Number of evaluation points when xgrid=None.

    delta_frac : float or None, default 5e-4
        Computational acceleration parameter expressed as a fraction of the
        observed x-range.

        For example, delta_frac=5e-4 gives

            delta = 0.0005 * (max(x) - min(x))

        Nearby LOWESS regressions separated by less than this distance are
        replaced by linear interpolation using statsmodels' native `delta`
        implementation.

        Set delta_frac=0 or None to disable this approximation and evaluate
        LOWESS directly at xgrid.

        This is not binning: the original continuous x and y observations
        remain the inputs to the LOWESS regression.

    bootstrap_size : int or None, default 100_000
        Number of observations sampled with replacement for each bootstrap
        LOWESS fit.

        If None, each bootstrap replicate contains the full number of finite
        observations, reproducing the conventional pairs bootstrap.

        If bootstrap_size is smaller than the dataset size, the bootstrap is
        an m-out-of-n approximation used to make uncertainty estimation
        computationally practical. The resulting standard error should not
        automatically be interpreted as the full-sample inferential standard
        error.

    random_state : int or None, default None
        Seed used for bootstrap resampling. Using the same seed with the same
        x-values allows identical resampling across different SCMs, provided
        their finite-data masks are also identical.

    show_progress : bool, default True
        If True, display a progress bar for the bootstrap replicates.

    return_details : bool, default False
        If True, also return a dictionary containing bootstrap quantiles,
        replicate smooths, and sample-size information.

    Returns
    -------
    xgrid : ndarray
        Common x-grid used to evaluate all LOWESS curves.

    smooth_full : ndarray
        Primary LOWESS trend estimated using all finite observations.
        This should generally be used as the plotted central trend.

    bootstrap_mean : ndarray
        Mean LOWESS trend across the K bootstrap replicates. Agreement between
        this curve and `smooth_full` provides a useful check that the chosen
        bootstrap sample size adequately represents the full dataset.

    stderr : ndarray
        Pointwise standard deviation of the bootstrap LOWESS estimates.

        When bootstrap_size equals the full sample size, this corresponds to
        the conventional pairs-bootstrap estimate used by the original
        `smooth()` function.

        When bootstrap_size is smaller than the full sample size, this is the
        variability of the m-out-of-n bootstrap smooths and should be treated
        as an uncertainty/stability diagnostic rather than automatically as a
        full-sample inferential standard error.

    details : dict, optional
        Returned only when return_details=True. Contains:

        n_input
            Number of original observations.

        n_valid
            Number of finite x-y observations used by the primary LOWESS.

        bootstrap_size
            Number of observations in each bootstrap replicate.

        q05, q95
            Pointwise 5th and 95th percentiles of the bootstrap LOWESS curves.

        smooths
            Array of shape (K, len(xgrid)) containing all bootstrap curves.

    Notes
    -----
    The bootstrap implemented here resamples x-y pairs independently, matching
    the statistical structure of the original `smooth()` implementation.
    For pairwise spatial-correlation datasets, rows may not be statistically
    independent because multiple pairs can share stations, events, or other
    spatial information. Consequently, dependence-aware uncertainty estimation
    may ultimately require a station-, event-, or spatial-block bootstrap.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if not 0 < frac <= 1:
        raise ValueError("frac must satisfy 0 < frac <= 1.")

    if it < 0:
        raise ValueError("it must be >= 0.")

    if K < 0:
        raise ValueError("K must be >= 0.")

    if delta_frac is not None and delta_frac < 0:
        raise ValueError("delta_frac must be >= 0 or None.")

    # Remove non-finite x-y pairs once, before any LOWESS calculations.
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]

    if x_valid.size < 3:
        raise ValueError("At least three finite x-y observations are required.")

    # Common evaluation grid.
    if xgrid is None:
        xgrid = np.linspace(x_valid.min(), x_valid.max(), int(n_grid))
    else:
        xgrid = np.asarray(xgrid, dtype=float).ravel()

        if xgrid.size < 2:
            raise ValueError("xgrid must contain at least two values.")

        if not np.all(np.isfinite(xgrid)):
            raise ValueError("xgrid must contain only finite values.")

        if np.any(np.diff(xgrid) <= 0):
            raise ValueError("xgrid must be strictly increasing.")

    # ------------------------------------------------------------------
    # Primary LOWESS: always use the complete finite simulation dataset.
    # ------------------------------------------------------------------
    smooth_full = _lowess_eval_grid(
        x_valid,
        y_valid,
        xgrid=xgrid,
        frac=frac,
        it=it,
        delta_frac=delta_frac,
    )

    # No bootstrap requested.
    if K == 0:
        bootstrap_mean = smooth_full.copy()
        stderr = np.full_like(smooth_full, np.nan)

        if return_details:
            details = {
                "n_input": len(x),
                "n_valid": len(x_valid),
                "bootstrap_size": 0,
                "q05": np.full_like(smooth_full, np.nan),
                "q95": np.full_like(smooth_full, np.nan),
                "smooths": np.empty((0, len(xgrid))),
            }

            return xgrid, smooth_full, bootstrap_mean, stderr, details

        return xgrid, smooth_full, bootstrap_mean, stderr

    # ------------------------------------------------------------------
    # Bootstrap size.
    # ------------------------------------------------------------------
    if bootstrap_size is None:
        n_boot = len(x_valid)
    else:
        n_boot = min(int(bootstrap_size), len(x_valid))

    if n_boot < 3:
        raise ValueError("bootstrap_size must contain at least three observations.")

    if frac * n_boot < 2:
        raise ValueError(
            "frac * bootstrap_size is too small for local regression. "
            "Increase frac or bootstrap_size."
        )

    rng = np.random.default_rng(random_state)

    # Only K x n_grid values are retained, so memory use remains small.
    smooths = np.empty((K, len(xgrid)), dtype=float)

    iterator = range(K)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrapping LOWESS")

    for k in iterator:
        # Draw directly from the complete finite dataset. No intermediate
        # fixed subsample is used.
        sample = rng.integers(0, len(x_valid), size=n_boot)

        smooths[k] = _lowess_eval_grid(
            x_valid[sample],
            y_valid[sample],
            xgrid=xgrid,
            frac=frac,
            it=it,
            delta_frac=delta_frac,
        )

    bootstrap_mean = np.mean(smooths, axis=0)

    # ddof=0 retains the definition used by the original smooth() function.
    stderr = np.std(smooths, axis=0, ddof=0)

    if return_details:
        details = {
            "n_input": len(x),
            "n_valid": len(x_valid),
            "bootstrap_size": n_boot,
            "q05": np.quantile(smooths, 0.05, axis=0),
            "q95": np.quantile(smooths, 0.95, axis=0),
            "smooths": smooths,
        }

        return xgrid, smooth_full, bootstrap_mean, stderr, details

    return xgrid, smooth_full, bootstrap_mean, stderr
# <LOESS Smoothing function>