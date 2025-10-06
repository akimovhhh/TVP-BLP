# src/tvpblp/tvblp.py

# ============================
# Imports
# ============================
import warnings
import numpy as np
from numpy.linalg import inv

from .utils import (
    ensure_list,
    as_float64,
    is_zero_L,
    build_base_ppf_grid,
)


# ============================
# Class
# ============================


class TvPBLP:
    """
    FILL THIS LATER
    """

    # IMPROVE: add a class-level docstring describing the math contract, shapes, and computational complexity per method.
    # IMPROVE: consider dataclass-style config object to keep __init__ signature concise (integration config, logging, solver options).
    # IMPROVE: add a random_state for reproducibility if you later support MC nodes.

    # ---------------------------------------------------------------------
    #                         CONSTRUCTOR / SETUP
    # ---------------------------------------------------------------------
    def __init__(
        self,
        data,
        share_col="share",
        price_col="price",
        lin=None,
        random=None,
        instruments=None,
        market_id="market",
        product_id="product",
        integration_points=50,
        integration_int=(0.1, 0.9),
        og_share_col="og_share",
    ):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Long product-level panel with columns listed below.
        share_col : str
            Observed inside-good share column s_{jm}.
        price_col : str
            Price column name (also used as default linear regressor).
        lin : list[str] or None
            Linear (mean-utility) regressors. Default: [price_col].
        random : list[str] or None
            Product characteristics with random coefficients (RC).
            If None or empty -> simple logit branch (no RC).
        instruments : list[str] or None
            IVs. If None -> reuse linear regressors as IVs.
        market_id : str
            Market identifier column.
        product_id : str
            Product identifier column (not used here but kept for API).
        integration_points : int
            Number of nodes per dimension for PPF grid (RC case).
        integration_int : tuple[float, float]
            Open-interval quantiles (qmin, qmax) to avoid +/-inf in PPF.
        og_share_col : str
            Outside good share s0_m or its proxy (per market).
        """
        # IMPROVE: add type hints; validate that og_share is per-market constant if required; assert shares sum < 1 per market (sanity).
        # --- store args
        self.data = data.copy()
        self.share_col = share_col
        self.price_col = price_col
        self.market_id = market_id
        self.product_id = product_id
        self.og_share_col = og_share_col
        self.integration_points = int(integration_points)
        self.integration_int = (float(integration_int[0]), float(integration_int[1]))

        # Creating all the necessary const attributes
        self.lin = ensure_list(lin, fallback=[price_col])
        self.random = ensure_list(random)
        if instruments is None:
            self.instruments = self.lin.copy()
            print(
                "No instruments specified, using linear characteristics as instruments"
            )
            # IMPROVE: route prints through a logger with verbosity levels; expose `verbose` flag at class-level.
        else:
            self.instruments = ensure_list(instruments)

        # ---- check for required columns
        required_cols = (
            [share_col, price_col, market_id, product_id, og_share_col]
            + self.lin
            + self.random
            + self.instruments
        )
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")
        # IMPROVE: deduplicate required_cols before checking to avoid false "missing" due to repeats; warn on constant columns.

        # ---- design matrices (contiguous float64)
        self.X = as_float64(self.data, self.lin)
        unique_cols = list(
            dict.fromkeys(
                [c for c in (self.instruments + self.lin) if c != self.price_col]
            )
        )
        self.Z = as_float64(self.data, unique_cols)
        # IMPROVE: You exclude price from Z via c != self.price_col. Make this explicit in docs or add flag include_price_as_instrument.
        # IMPROVE: Precompute ZTy only when needed; keep memory footprint smaller for large L.

        # ---- random-coeff matrix or None
        if self.random:
            self.random_coeff_data = as_float64(self.data, self.random)
        else:
            self.random_coeff_data = None
        # IMPROVE: standardize/center RC covariates to improve conditioning; cache means/std.

        # ---- shares (clamped logs)
        eps = 1e-300
        self.share = np.ascontiguousarray(
            self.data[share_col].to_numpy(dtype=np.float64)
        )
        self.og_share = np.ascontiguousarray(
            self.data[og_share_col].to_numpy(dtype=np.float64)
        )
        self.log_shares = np.log(np.maximum(self.share, eps))
        self.log_og_shares = np.log(np.maximum(self.og_share, eps))
        # IMPROVE: validate that s_inside + s0 ≈ 1 per market; warn if off by > tol (data issues).

        # ---- dimensions / flags
        self.n_obs = self.X.shape[0]
        self.k_vars = self.X.shape[1]
        self.k_instruments = self.Z.shape[1]
        self.k_random = (
            0 if self.random_coeff_data is None else self.random_coeff_data.shape[1]
        )
        self.is_simple_logit = self.k_random == 0

        # ---- identification warning (YOUR logic)
        needed_instruments = self.k_vars + (self.k_random * (self.k_random + 1)) // 2
        if self.k_instruments < needed_instruments:
            warnings.warn(
                f"Not enough instruments for your specification: have {self.k_instruments}, "
                f"but need at least {needed_instruments} (= k_vars + k_random*(k_random+1)/2 with k_random={self.k_random}).",
                RuntimeWarning,
            )
        # IMPROVE: support micro-moments later; expose formula in docstring; consider rank checks on Z'X.

        # ---- market grouping, need this for fast computations
        self.markets = self.data[market_id].to_numpy()
        self.uniq_markets, self.market_inv = np.unique(
            self.markets, return_inverse=True
        )
        self.n_markets = self.uniq_markets.size

        order = np.argsort(self.market_inv, kind="mergesort")
        self._order = order
        self._rev_order = np.empty_like(order)
        self._rev_order[order] = np.arange(self.n_obs)

        g_sorted = self.market_inv[order]
        self._seg_starts = np.r_[0, 1 + np.flatnonzero(g_sorted[1:] != g_sorted[:-1])]
        self._seg_ends = np.r_[self._seg_starts[1:], g_sorted.size]
        self._seg_sizes = self._seg_ends - self._seg_starts
        # IMPROVE: also store a list of slice objects per market to avoid recomputing start/end in loops; precompute repeat indexers for speed.

        print(
            f"Initialized BLP with {self.n_obs} observations across {self.n_markets} markets."
        )
        print(f"Linear characteristics: {self.lin}")
        if self.is_simple_logit:
            print("Model: simple logit (no random coefficients).")
        else:
            print(f"Random coefficients on: {self.random}")
        print(f"Instruments: {unique_cols}")
        # IMPROVE: add `__repr__` summarizing config; send prints to logger.

        # ============================
        # Precomputed “constants”
        # ============================

        # Cross-products & transposes you’ll reuse a lot
        self._XTZ = self.X.T @ self.Z  # k × L
        self._ZTX = self._XTZ.T  # L × k
        self._XTX = self.X.T @ self.X  # k × k
        self._ZTZ = self.Z.T @ self.Z  # L × L
        self._X_T = self.X.T  # cached to avoid .T in hot paths
        self._Z_T = self.Z.T
        # IMPROVE: guard against unnecessary big matrices if never used (lazy eval properties).
        # IMPROVE: consider Cholesky of Z'Z, X'X for repeated solves; ridge regularization knobs.

        # Base standard-normal tensor grid (RC only) — built once
        self.grid = build_base_ppf_grid(
            q_dim=self.k_random,
            points_per_dim=self.integration_points,
            qmin=self.integration_int[0],
            qmax=self.integration_int[1],
        )
        # IMPROVE: allow alternative quadrature (Gauss-Hermite) when normality assumed; add option for antithetic pairs for variance reduction.

        # Caches
        self._grid_cache = None  # correlated tensor nodes v = z @ L^T
        self._last_L = None  # last L used to build _grid_cache
        self._rc_cache = {}  # cache for RC intermediates per L
        # IMPROVE: use np.allclose for L cache key; include tolerance to reuse across near-identical iterations.

    # ---------------------------------------------------------------------
    #                         GROUP REDUCERS (FAST)
    # ---------------------------------------------------------------------
    def _seg_max(self, x_sorted):
        """Per-market maximum with reduceat. x_sorted must be sorted by market."""
        # IMPROVE: for numerical stability, you use marketwise max—good. Consider returning both max and an index if needed later for diagnostics.
        return np.maximum.reduceat(x_sorted, self._seg_starts)

    def _seg_sum(self, x_sorted):
        """Per-market sum with reduceat. x_sorted must be sorted by market."""
        # IMPROVE: could expose axis argument to reuse for 2D arrays (broadcasted reduceat), removing Python loops elsewhere.
        return np.add.reduceat(x_sorted, self._seg_starts)

    # ---------------------------------------------------------------------
    #               CORRELATED NODE GRID FROM CHOLESKY L
    # ---------------------------------------------------------------------
    def mvn_ppf_grid(self, L):
        """
        Given Cholesky factor L (q x q), transform the base standard-normal grid z
        into correlated nodes v = z @ L^T on the tensor grid.
        (YOUR original contract: full L or zero/None -> simple logit)
        """
        if is_zero_L(L):
            return None
        if self.grid is None:
            raise ValueError("Random-coefficients grid requested but k_random == 0.")

        L = np.asarray(L, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(f"L must be square; got shape {L.shape}")
        if L.shape[0] != self.k_random:
            raise ValueError(f"L is {L.shape} but k_random = {self.k_random}")

        X = self.grid @ L.T  # (..., q)
        # IMPROVE: if L is lower-tri, @ L.T is fine. Consider broadcasting for batched L if you do multi-starts.
        return np.ascontiguousarray(X, dtype=np.float64)

    def get_integration_grid(self, L):
        """
        Cache correlated grid for given L.
        """
        # IMPROVE: use np.allclose for cache equality; parameterize tolerance; track cache hits/misses for profiling.
        L_arr = None if L is None else np.asarray(L, dtype=np.float64)
        need = (
            self._grid_cache is None
            or self._last_L is None
            and L_arr is not None
            or (
                self._last_L is not None
                and (L_arr is None or not np.array_equal(self._last_L, L_arr))
            )
        )
        if need:
            self._grid_cache = self.mvn_ppf_grid(L)
            self._last_L = None if L_arr is None else np.array(L_arr, copy=True)
        return self._grid_cache

    # ---------------------------------------------------------------------
    #           RANDOM-COEFFS CACHE (MUS = X_random @ grid_flat.T), PER L
    # ---------------------------------------------------------------------
    def _prepare_rc_cache(self, L):
        """
        Precompute heavy pieces for RC share computation:
        - grid (tensor), grid_flat (R x q), mus = X_random @ grid_flat.T (n_obs x R)
        """
        if self.k_random == 0 or is_zero_L(L):
            return  # simple-logit branch; nothing to prepare

        L_arr = np.asarray(L, dtype=np.float64)
        key_ok = (self._rc_cache.get("L_key") is not None) and np.array_equal(
            self._rc_cache["L_key"], L_arr
        )
        if not key_ok:
            grid = self.get_integration_grid(L_arr)  # (..., q)
            if grid is None:
                return
            grid_flat = grid.reshape(-1, grid.shape[-1])  # (n_nodes, q)
            mus = self.random_coeff_data @ grid_flat.T  # (n_obs, n_nodes)
            # IMPROVE: if memory tight, compute mus in blocks over nodes; or store grid_flat.T and matmul on demand.
            self._rc_cache = {
                "L_key": np.array(L_arr, copy=True),
                "grid": grid,
                "grid_flat": np.ascontiguousarray(grid_flat, dtype=np.float64),
                "mus": np.ascontiguousarray(mus, dtype=np.float64),
            }

    # ---------------------------------------------------------------------
    #                  SHARE COMPUTATION (SIMPLE / RC)
    # ---------------------------------------------------------------------
    def _compute_shares_simple(self, deltas):
        """
        Simple logit shares with per-market max subtraction.

            s_{jm} = exp(δ_{jm} - max_m) / ( 1 + Σ_{j'∈m} exp(δ_{j'm} - max_m) )
        """
        # IMPROVE: vectorize repeats by precomputing a "repeat indexer" array once; repeating every call allocs new arrays.
        d_sorted = deltas[self._order]
        mmax = self._seg_max(d_sorted)  # (n_markets,)
        mmax_obs_sorted = np.repeat(mmax, self._seg_sizes)

        with np.errstate(over="ignore"):
            ex_sorted = np.exp(d_sorted - mmax_obs_sorted)

        sum_sorted = self._seg_sum(ex_sorted)  # (n_markets,)
        denom_sorted = np.exp(-mmax) + sum_sorted  # + outside good
        # NOTE: denom uses "1" (exp(-mmax)) as outside; this matches the logit normalization with δ0=0 after subtraction.

        denom_obs_sorted = np.repeat(denom_sorted, self._seg_sizes)
        shares_sorted = ex_sorted / denom_obs_sorted
        return shares_sorted[self._rev_order]

    def _compute_shares_rc(self, deltas, L):
        """
        Random-coefficients shares via node-wise logit, averaged over nodes:

            u_{jm}^{(r)} = δ_{jm} + x_{jm}^RC · v^{(r)}
            s_{jm}^{(r)} = exp(u_{jm}^{(r)} - max_m^{(r)}) /
                           ( exp(-max_m^{(r)}) + Σ_{j'∈m} exp(u_{j'm}^{(r)} - max_m^{(r)}) )
            s_{jm} = average_r s_{jm}^{(r)}
        """
        # IMPROVE: reduce Python loops by using 3D arrays: shape (n_obs_sorted, R) and reduceat along axis=0 with broadcasting.
        self._prepare_rc_cache(L)
        mus = self._rc_cache["mus"]  # (n_obs, n_nodes)

        util = mus + deltas[:, None]  # (n_obs, n_nodes)
        util_s = util[self._order, :]  # (n_obs, n_nodes)

        # Per-market, per-node maxima
        # IMPROVE: avoid Python list comprehension by transposing to (R, n_obs_sorted) and using reduceat along axis=1 once; OK here but costly for big R.
        u_t = util_s.T
        mmax_t = np.vstack(
            [np.maximum.reduceat(u_t[i], self._seg_starts) for i in range(u_t.shape[0])]
        )
        mmax = mmax_t.T  # (n_markets, n_nodes)
        mmax_obs_s = np.repeat(mmax, self._seg_sizes, axis=0)

        with np.errstate(over="ignore"):
            ex_s = np.exp(util_s - mmax_obs_s)  # (n_obs, n_nodes)

        # Per-market sums for each node
        sum_s_t = np.vstack(
            [
                np.add.reduceat(ex_s[:, i], self._seg_starts)
                for i in range(ex_s.shape[1])
            ]
        )
        sum_s = sum_s_t.T  # (n_markets, n_nodes)

        with np.errstate(over="ignore"):
            denom = sum_s + np.exp(-mmax)  # (n_markets, n_nodes)

        denom_obs_s = np.repeat(denom, self._seg_sizes, axis=0)
        shares_s = ex_s / denom_obs_s
        shares = shares_s[self._rev_order, :].mean(axis=1)
        # IMPROVE: expose option for weighted quadrature average vs plain mean.
        return shares

    def compute_shares(self, L, deltas):
        """
        Dispatcher to simple-logit vs RC branch (Cholesky L parameterization).
        (YOUR logic preserved)
        """
        # IMPROVE: accept Sigma and internally Cholesky to L with softplus diag for positivity (better parameterization for optimization).
        if (self.k_random == 0) or is_zero_L(L):
            return self._compute_shares_simple(deltas)
        return self._compute_shares_rc(deltas, np.asarray(L, dtype=np.float64))

    # ---------------------------------------------------------------------
    #                     BLP CONTRACTION (WITH SQUAREM)
    # ---------------------------------------------------------------------
    def BLP_inversion(self, L, tol=1e-12, max_iter=1000, verbose=True):
        """
        Fixed-point inversion to recover mean utilities δ given observed shares s.

            δ_{k+1} = T(δ_k) := δ_k + (log s_obs - log s_pred(δ_k))

        SQUAREM forms an extrapolated step from T(δ_k) and T(T(δ_k)).
        (YOUR logic preserved)
        """
        # IMPROVE: add monotone line search safeguard for SQUAREM (fallback to plain iterate if overshoot increases residual norm).
        # IMPROVE: add maximum step cap to avoid NaNs; track and reproject finite values if needed.
        log_s_obs = self.log_shares
        deltas = log_s_obs - self.log_og_shares  # standard warm start
        # NOTE: This warm start assumes δ0 = log(s) - log(s0). Good default.

        err = np.inf
        it = 1
        while (err > tol) and (it < max_iter):
            old = deltas

            # First map
            shares = self.compute_shares(L, old)
            shares = np.maximum(shares, 1e-300)
            r = log_s_obs - np.log(shares)
            x1 = old + r

            # Second map
            shares2 = self.compute_shares(L, x1)
            shares2 = np.maximum(shares2, 1e-300)
            v = (x1 + (log_s_obs - np.log(shares2))) - x1
            v = v - r

            # SQUAREM extrapolation
            vv = float(np.dot(v, v))
            alpha = (float(np.dot(v, r)) / vv) if vv != 0.0 else 1.0
            step = -2.0 * alpha * r + (alpha * alpha) * v
            # IMPROVE: clamp alpha within [a_min, a_max]; if residual grows, backtrack α→0 (i.e., use x1).
            deltas = old + step

            err = float(np.max(np.abs(step)))
            if verbose and (it == 1 or it % 10 == 0):
                print(f"Iteration {it:4d} | max Δ: {err:.3e}")
            it += 1

        if verbose:
            if err <= tol:
                print(f"Converged in {it-1} iterations (max Δ ≈ {err:.3e}).")
            else:
                print(
                    f"Stopped at {it-1} iterations; tolerance not met (max Δ ≈ {err:.3e})."
                )
        # IMPROVE: return diagnostics (iters, last_residual_norm) for adaptive outer optimization.
        return deltas

    # ---------------------------------------------------------------------
    #                         MOMENTS & OBJECTIVE
    # ---------------------------------------------------------------------
    def g_moments(self, L, W):
        """
        GMM moments g(β, L) = (1/n) Z' (δ - Xβ) where δ solves the BLP inversion
        for the given L and W.
        (YOUR logic preserved, but uses precomputed cross-products)
        """
        deltas = self.BLP_inversion(L, verbose=False)
        XTZ = self._XTZ  # k × L
        ZTd = self._Z_T @ deltas  # L
        A = inv(XTZ @ (W @ (XTZ.T)))  # k × k
        beta = A @ (XTZ @ (W @ ZTd))  # k
        resid = deltas - (self.X @ beta)  # n
        g = (self._Z_T @ resid) / self.n_obs  # L
        # IMPROVE: use solves (chol) instead of explicit inv; cache W^(1/2) if repeated; optionally center Z to reduce collinearity.
        return g

    def G_objective(self, L, W):
        """
        GMM objective G(L) = g(β(L), L)' W g(β(L), L).
        """
        g = self.g_moments(L, W)
        G = 1e6 * float(g.T @ (W @ g))  # scale is arbitrary for optimization
        # IMPROVE: expose scale as constant; add small ridge penalty on L (e.g., λ * ||L||_F^2) to help with local minima.
        return G

    # ---------------------------------------------------------------------
    #                           GRADIENT
    # ---------------------------------------------------------------------
    # IMPROVE: add finite-difference checker (debug mode) on small toy data to validate grad_L; expose jit/numba option.

    def _adjoint_seed_lambda(self, L, W, deltas, g):
        """
        λ = (1/n) * M' Z W g,  M = I - X A^{-1} X' Z W Z',  A = X'Z W Z'X.
        Uses cached cross-products.
        """
        # IMPROVE: avoid forming Z W Z' explicitly (good—you're not); ensure A SPD and use chol solve; add ridge if ill-conditioned.
        y = W @ g  # (L,)
        u = self.Z @ y  # (n,)   = Z W g
        t = self._XTZ @ y  # (k,)   = X'Z W g
        A = self._XTZ @ (W @ self._XTZ.T)  # (k,k)  = X'Z W Z'X
        w = np.linalg.solve(A, t)  # (k,)   = A^{-1} X'Z W g
        xaw = self.X @ w  # (n,)   = X A^{-1} X'Z W g
        zw = self._Z_T @ xaw  # (L,)   = Z' X A^{-1} X' Z W g
        zWzw = self.Z @ (W @ zw)  # (n,)   = Z W Z' X A^{-1} X' Z W g
        return (u - zWzw) / self.n_obs  # (n,)

    def _node_probabilities_sorted(self, deltas, L):
        # IMPROVE: same vectorization note as _compute_shares_rc—replace Python vstack loop with reduceat over axis using broadcasted dims.
        self._prepare_rc_cache(L)
        mus = self._rc_cache["mus"]  # (n, R)
        util_s = (mus + deltas[:, None])[self._order, :]
        u_t = util_s.T
        mmax_t = np.vstack(
            [np.maximum.reduceat(u_t[i], self._seg_starts) for i in range(u_t.shape[0])]
        )
        mmax = mmax_t.T
        mmax_obs = np.repeat(mmax, self._seg_sizes, axis=0)
        with np.errstate(over="ignore"):
            ex = np.exp(util_s - mmax_obs)
        sum_t = np.vstack(
            [np.add.reduceat(ex[:, i], self._seg_starts) for i in range(ex.shape[1])]
        )
        sums = sum_t.T
        denom = sums + np.exp(-mmax)
        p_s = ex / np.repeat(denom, self._seg_sizes, axis=0)  # (n_sorted, R)
        s_s = p_s.mean(axis=1)  # (n_sorted,)
        return p_s, s_s

    def _Jbar_matvec_for_market(self, p_nodes_m):
        """
        Closure for v -> Jbar_m v  where
        Jbar_m v = mean_r( diag(p_r) v - p_r (p_r' v) ).
        Inputs:
        p_nodes_m : (n_m, R)
        """
        # IMPROVE: use this CG path in grad_L instead of forming dense Jbar; better scaling for large n_m.
        R = p_nodes_m.shape[1]

        def matvec(v):
            term1 = (p_nodes_m * v[:, None]).mean(axis=1)  # mean_r (p_r ⊙ v)
            scal = (p_nodes_m * v[:, None]).sum(axis=0)  # [p_r' v]_r
            term2 = (p_nodes_m @ scal) / R  # mean_r p_r (p_r' v)
            return term1 - term2

        return matvec

    def _cg_solve(self, matvec, b, tol=1e-10, maxiter=1000):
        # IMPROVE: return convergence info (#iters, residual norm); add preconditioner if available.
        x = np.zeros_like(b)
        r = b - matvec(x)
        p = r.copy()
        rs = float(r @ r)
        for _ in range(maxiter):
            Ap = matvec(p)
            denom = float(p @ Ap) + 1e-30
            alpha = rs / denom
            x += alpha * p
            r -= alpha * Ap
            rs_new = float(r @ r)
            if rs_new**0.5 < tol:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    def _build_Jbar_dense(self, p_nodes_m):
        """
        Construct dense Jbar_m = mean_r( diag(p_r) - p_r p_r' ) for a single market.
        p_nodes_m: (n_m, R)
        Returns: (n_m, n_m) float64
        """
        # IMPROVE: prefer CG via _Jbar_matvec_for_market to avoid O(n_m^2) memory; this dense path is fine for small markets only.
        n_m, R = p_nodes_m.shape
        # mean diag(p_r)
        mean_diag = p_nodes_m.mean(axis=1)  # (n_m,)
        Jbar = np.diag(mean_diag)
        # mean outer(p_r, p_r)
        # Do it stably: accumulate outer products as means
        # (n_m, n_m) = (1/R) * sum_r p_r p_r'
        outer_mean = (p_nodes_m @ p_nodes_m.T) / R  # uses BLAS
        Jbar -= outer_mean
        return Jbar

    def grad_L(self, L, W, deltas=None):
        """
        Gradient of G(L) = 1e6 * g(L)^T W g(L) w.r.t. the Cholesky L (lower-tri).
        """
        # IMPROVE: parameterize L via unconstrained vector (softplus for diag; identity for lower off-diag) to keep SPD; reduces local minima from invalid L.
        # If no RC or L is zero → no dependence on L
        if (self.k_random == 0) or (L is None):
            return np.zeros((self.k_random, self.k_random), dtype=np.float64)

        L_arr = np.asarray(L, dtype=np.float64)
        if np.allclose(L_arr, 0.0):
            return np.zeros_like(L_arr)

        # 1) delta, moments g, and adjoint seed λ
        deltas = self.BLP_inversion(L, verbose=False) if deltas is None else deltas
        g = self.g_moments(L, W)
        lam = self._adjoint_seed_lambda(L, W, deltas, g)
        # IMPROVE: cache deltas, g, lam across obj/grad calls inside optimizer callback for 2x speed.

        # 2) Node probabilities (sorted by market)
        p_s, _ = self._node_probabilities_sorted(deltas, L)
        lam_s = lam[self._order]

        # 3) Pre-allocations
        q = self.k_random
        grad = np.zeros((q, q), dtype=np.float64)

        # RC characteristics & base nodes
        Xrc_s = self.random_coeff_data[self._order, :]
        z_nodes = self.grid.reshape(-1, q)
        R = z_nodes.shape[0]  # Total number of integration nodes
        # IMPROVE: if using Gauss-Hermite, include weights and propagate into all means/sums.

        # 4) Market-by-market accumulation
        start = 0
        for m_size in self._seg_sizes:
            end = start + m_size
            p_m = p_s[start:end, :]  # (n_m, R)
            lam_m = lam_s[start:end]  # (n_m,)
            Xm = Xrc_s[start:end, :]  # (n_m, q)

            # Solve Jbar_m u_m = λ_m
            Jbar = self._build_Jbar_dense(p_m)
            eps = 1e-12 * max(1.0, float(np.trace(Jbar)) / max(1, Jbar.shape[0]))
            try:
                Lc = np.linalg.cholesky(Jbar + eps * np.eye(Jbar.shape[0]))
                u_m = np.linalg.solve(Lc.T, np.linalg.solve(Lc, lam_m))
            except np.linalg.LinAlgError:
                u_m = np.linalg.solve(Jbar + 1e-9 * np.eye(Jbar.shape[0]), lam_m)
            # IMPROVE: switch to CG with matvec to avoid dense Jbar; add early-stopping tol.

            # Compute statistics (sums over products)
            beta = p_m.T @ Xm  # (R, q)
            eta = p_m.T @ (u_m[:, None] * Xm)  # (R, q)
            alpha = p_m.T @ u_m  # (R,)

            diff = eta - alpha[:, None] * beta  # (R, q)

            # Fill lower triangle (b >= a)
            # IMPROVE: vectorize these nested loops:
            #   M = -(2/R) * (z_nodes.T @ diff)  → then take tril(M)
            for a in range(q):
                za = z_nodes[:, a]  # (R,)
                for b in range(a, q):  # b >= a for lower triangle
                    grad[b, a] += -2.0 * np.dot(za, diff[:, b])

            start = end

        # Convert sum to mean by dividing by number of nodes
        grad /= R

        # Scale to match objective
        grad *= 1e6

        # Zero out upper triangle (ensure strictly lower triangular + diagonal)
        for i in range(q):
            for j in range(i + 1, q):
                grad[i, j] = 0.0
        # IMPROVE: use np.tril(grad) to zero-out upper triangle faster.
        return grad

    def fit(self, W, L_init=None, verbose=True, tol=1e-12):
        """
        Minimal GMM optimization to test gradient logic.

        Parameters
        ----------
        tol : float
            Stop when ||L_new - L_old||_inf < tol
        """
        from scipy.optimize import minimize

        # IMPROVE (Interface): allow W=None and do two-step GMM internally:
        #   step1 W=I → estimate β, residuals → W2 = (Z'εε'Z / n)^-1 with ridge.
        # IMPROVE (Interface): accept callbacks, max_time, random_restarts, trust-constr with bounds on diag(L) > 0 via softplus parametrization.
        # IMPROVE: accept "method" and optimizer options dict; log progress via callback rather than print.

        # Simple logit case
        if self.k_random == 0:
            print("Simple logit - no optimization needed")
            return {"L_opt": None}

        # Vector/matrix conversion
        def vec_to_L(x):
            L = np.zeros((self.k_random, self.k_random))
            idx = 0
            for i in range(self.k_random):
                for j in range(i + 1):
                    L[i, j] = x[idx]
                    idx += 1
            return L

        # IMPROVE: store index map once (triangular indices) to speed conversions; use np.tril_indices.

        def L_to_vec(L):
            vec = []
            for i in range(self.k_random):
                for j in range(i + 1):
                    vec.append(L[i, j])
            return np.array(vec)

        # IMPROVE: vectorize using tril_indices; avoid Python loops.

        # Track iterations for custom stopping
        iteration_data = {"prev_L": None, "iter": 0, "should_stop": False}
        # IMPROVE: also track best_obj, best_L; early-stopping on obj plateaus; stagnation detection.

        # Objective and gradient with tracking
        def obj_func(x):
            L = vec_to_L(x)
            obj = self.G_objective(L, W)

            # Check stopping criterion
            if iteration_data["prev_L"] is not None:
                L_diff = np.max(np.abs(L - iteration_data["prev_L"]))
                if verbose and iteration_data["iter"] % 5 == 0:
                    print(
                        f"  Iter {iteration_data['iter']:3d}: obj = {obj:.6e}, max|ΔL| = {L_diff:.3e}"
                    )
                if L_diff < tol:
                    iteration_data["should_stop"] = True
                    if verbose:
                        print(f"  Converged! max|ΔL| = {L_diff:.3e} < {tol}")
            # IMPROVE: add trust-region stop if gradient norm small; check δ inversion residual as additional termination criterion.

            iteration_data["prev_L"] = L.copy()
            iteration_data["iter"] += 1

            # Force stop by returning very small objective if converged
            if iteration_data["should_stop"]:
                return obj * 1e-20  # Trick to make optimizer stop
            # IMPROVE: prefer optimizer-native termination (set 'maxiter' and callback to signal stop) instead of skewing objective.
            return obj

        def grad_func(x):
            if iteration_data["should_stop"]:
                # Return zero gradient to stop optimizer
                return np.zeros_like(x)
            L = vec_to_L(x)
            grad_mat = self.grad_L(L, W)
            return L_to_vec(grad_mat)

        # IMPROVE: cache deltas/g/λ between obj/grad calls using closure state to avoid recomputation per iteration.

        # Initial guess - CONVERT TO NUMPY ARRAY
        if L_init is None:
            L_init = 0.1 * np.eye(self.k_random)
        else:
            L_init = np.asarray(L_init, dtype=np.float64)
            if L_init.shape != (self.k_random, self.k_random):
                if L_init.size == 1:
                    L_init = L_init.flat[0] * np.eye(self.k_random)
                else:
                    raise ValueError(
                        f"L_init must be ({self.k_random}, {self.k_random})"
                    )
        # IMPROVE: multi-start strategy (e.g., random lower-tri draws scaled) to avoid local minima; select best by objective.

        x0 = L_to_vec(L_init)

        # Minimize
        print(f"Starting optimization with L_init diagonal: {np.diag(L_init)}")
        print(f"Stopping tolerance: max|ΔL| < {tol}")

        res = minimize(
            fun=obj_func,
            x0=x0,
            jac=grad_func,
            method="L-BFGS-B",
            options={"disp": verbose, "maxiter": 1000, "ftol": 1e-20, "gtol": 1e-20},
        )
        # IMPROVE: try 'trust-constr' with bound/constraint on diag(L) ≥ ε; or reparametrize to unconstrained θ and map to L via softplus on diag.
        # IMPROVE: set reasonable ftol/gtol; current 1e-20 is very tight and may waste iterations.

        # Get final L
        if iteration_data["should_stop"]:
            L_opt = iteration_data["prev_L"]
        else:
            L_opt = vec_to_L(res.x)

        # Results
        print(f"\nOptimization finished after {iteration_data['iter']} iterations")
        print(f"Convergence by L change: {iteration_data['should_stop']}")
        print(f"Success: {res.success}")
        print(f"Final objective: {self.G_objective(L_opt, W):.6e}")
        print(f"Optimal L:\n{L_opt}")
        # IMPROVE: return β_hat, δ_hat, W_used, diagnostics (timings, gradients, restarts tried).

        return {
            "L_opt": L_opt,
            "obj_val": self.G_objective(L_opt, W),
            "result": res,
            "n_iter": iteration_data["iter"],
        }
