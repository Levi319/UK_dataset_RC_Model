# ============================================
# 1R1C_fit_all_homes_fixed_weights_linearstyle.py
#
# Goal:
#   - Fixed weights (phi_q, phi_e, phi_u) = (450, 1, 350)
#   - Fit model parameters for EACH home (Property_ID) using CVXPY formulation (NOT regression)
#   - Use SAME input files / schema style as your linear-regression script:
#       df_train_detached = pd.read_parquet("retrieved_weather_data/merged_data_final_homes.parquet")
#       df_house = pd.read_csv("retrieved_weather_data/home_characteristics.csv")
#   - For each home, pick a home-specific 12-month window:
#       * Prefer a full calendar year (Jan1–Dec31) if coverage is good
#       * Else fall back to best rolling 12 months
#       * Then REMOVE summer months (Jun–Aug) from the training set
#   - Compute DHW_sum using summer months *within the chosen 12-month window*
#   - Save train_start/train_end in the output CSV
#
# Notes:
#   - This script is "inspired" by linear-regression version in how windows/gating are chosen
#     and how data is read, but the solve method is CVXPY (different model/optimization).
# ============================================

import os

# Cap threads (important for mac stability)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MallocNanoZone", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import cvxpy as cp
import multiprocessing as mp
import psutil
import gc
import traceback
import datetime


# ----------------------------
# Memory logging (parent)
# ----------------------------
PROC = psutil.Process(os.getpid())

def rss_gb() -> float:
    return PROC.memory_info().rss / (1024**3)

def log_mem(tag: str) -> None:
    print(f"[mem] {tag:30s} RSS={rss_gb():.2f} GB")


# ----------------------------
# Window selection helper (calendar year preferred; else best rolling 12 months)
# ----------------------------
def find_annual_window_home(
    idx: pd.DatetimeIndex,
    *,
    enforce_freq: str = "30min",
    summer_months=(6, 7, 8),
    prefer_calendar_year: bool = True,
    min_coverage_non_summer: float = 0.70,
) -> tuple[pd.Timestamp, pd.Timestamp, dict]:
    """
    Returns (t_start, t_end, diag) for a 12-month window.

    Coverage scoring uses a forced regular grid at enforce_freq and computes
    fraction of timestamps present for NON-summer months only (since summer
    is later excluded from training anyway).
    """
    diag = {}

    if idx is None or len(idx) == 0:
        raise ValueError("Empty index; cannot select annual window.")

    idx = pd.DatetimeIndex(idx).sort_values()
    t_min = idx.min().floor(enforce_freq)
    t_max = idx.max().ceil(enforce_freq)

    grid = pd.date_range(t_min, t_max, freq=enforce_freq)
    present = pd.Series(False, index=grid)
    present.loc[grid.intersection(idx)] = True

    def score_window(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
        g = present.loc[t0:t1]
        if g.empty:
            return -np.inf
        non_summer_mask = ~g.index.month.isin(list(summer_months))
        denom = int(non_summer_mask.sum())
        if denom == 0:
            return -np.inf
        return float(g[non_summer_mask].mean())

    best = None  # (score, t0, t1, kind)

    # 1) calendar years
    if prefer_calendar_year:
        for y in range(t_min.year, t_max.year + 1):
            t0 = pd.Timestamp(y, 1, 1)
            t1 = pd.Timestamp(y, 12, 31, 23, 59, 59)
            if t0 < t_min or t1 > t_max:
                continue
            sc = score_window(t0, t1)
            if (best is None) or (sc > best[0]):
                best = (sc, t0, t1, "calendar_year")

        if best is not None and best[0] >= min_coverage_non_summer:
            diag.update({"window_kind": best[3], "coverage_non_summer": best[0]})
            return best[1], best[2], diag

    # 2) rolling 12-month windows starting at month boundaries
    start_month = pd.Timestamp(t_min.year, t_min.month, 1)
    end_month = pd.Timestamp(t_max.year, t_max.month, 1)
    candidate_starts = pd.date_range(start_month, end_month, freq="MS")

    for t0 in candidate_starts:
        t1 = (t0 + pd.DateOffset(months=12)) - pd.Timedelta(seconds=1)
        if t0 < t_min or t1 > t_max:
            continue
        sc = score_window(t0, t1)
        if (best is None) or (sc > best[0]):
            best = (sc, t0, t1, "rolling_12mo")

    if best is None or best[0] == -np.inf:
        # last-ditch fallback: first 12 months available
        t0 = t_min
        t1 = min(t_max, t0 + pd.DateOffset(months=12) - pd.Timedelta(seconds=1))
        diag.update({"window_kind": "fallback_first_12mo", "coverage_non_summer": None})
        return t0, t1, diag

    diag.update({"window_kind": best[3], "coverage_non_summer": best[0]})
    return best[1], best[2], diag

import numpy as np
import pandas as pd

def pick_good_feb_week_bounds(
    t_start: pd.Timestamp,
    idx: pd.DatetimeIndex,
    signals: dict[str, np.ndarray],
    *,
    week_days: int = 7,
    max_missing_frac: float = 0.10,
    max_gap_steps: int = 3,  # 3 * 30min = 1.5 hours
):
    """
    Choose a Feb week [t0, t1] where:
      - worst missing fraction across provided signals <= max_missing_frac
      - no consecutive NaN run longer than max_gap_steps in any provided signal

    If t_start.month > 2, uses February of the following year.

    Returns (t0, t1) as Timestamps, or (None, None) if not found.
    """
    idx = pd.DatetimeIndex(idx)

    year_use = t_start.year + (1 if t_start.month > 2 else 0)
    feb_start = pd.Timestamp(year_use, 2, 1, 0, 0, 0)
    mar_start = pd.Timestamp(year_use, 3, 1, 0, 0, 0)

    week = pd.Timedelta(days=week_days) - pd.Timedelta(minutes=30)

    def longest_nan_run_steps(x: np.ndarray) -> int:
        is_bad = ~np.isfinite(x)
        run = longest = 0
        for v in is_bad:
            if v:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
        return longest

    t0 = feb_start
    while (t0 + week) < mar_start:
        t1 = t0 + week
        mask = (idx >= t0) & (idx <= t1)

        if mask.sum() < 10:
            t0 += pd.Timedelta(days=1)
            continue

        ok = True
        for name, arr in signals.items():
            w = np.asarray(arr)[mask]
            miss = float(np.mean(~np.isfinite(w)))
            if miss > max_missing_frac:
                ok = False
                break
            if longest_nan_run_steps(w) > max_gap_steps:
                ok = False
                break

        if ok:
            return t0, t1

        t0 += pd.Timedelta(days=1)

    return None, None


# ----------------------------
# Gating / cleaning inspired by regression script (but simplified)
# ----------------------------
def clean_and_gate_window(
    df_single: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    *,
    enforce_freq: str = "30min",
    max_gap_steps: int = 6,              # 6*30min = 3 hours
    do_interpolate: bool = True,
    # gating columns (must exist after preprocessing)
    col_Ti: str = "Internal_Air_Temperature",
    col_qhp_diff: str = "Heat_Pump_Energy_Output_Diff",
    max_missing: float = 0.20,           # overall missingness threshold in window (on required cols)
    min_points: int = 1000,              # ~ 21 days at 30min
) -> tuple[pd.DataFrame | None, dict]:
    """
    Returns (df_window_clean, diag).
    - Ensures Timestamp index
    - Slices to [t_start, t_end]
    - Rounds/resamples to 30min grid
    - Interpolates short gaps (<= max_gap_steps) on numeric cols
    - Drops remaining NaNs on required columns
    - Enforces min_points
    """
    diag = {}

    df = df_single.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(subset=[col_Ti], how="all").sort_index()

    # slice
    dfw = df.loc[(df.index >= t_start) & (df.index <= t_end)].copy()
    if dfw.empty:
        diag["reason"] = "empty_window"
        return None, diag

    # regularize
    dfw.index = dfw.index.round(enforce_freq)
    dfw = dfw.groupby(dfw.index).mean(numeric_only=True)
    dfw = dfw.resample(enforce_freq).mean()

    # required cols check
    req_cols = [c for c in [col_Ti, col_qhp_diff] if c in dfw.columns]
    if len(req_cols) < 2:
        diag["reason"] = "missing_required_cols"
        diag["required_cols_present"] = req_cols
        return None, diag

    miss = float(dfw[req_cols].isna().any(axis=1).mean())
    diag["missing_any_req_before"] = miss
    diag["n_rows_before"] = int(len(dfw))

    if miss > max_missing:
        diag["reason"] = "missingness_too_high_before_interp"
        return None, diag

    # interpolate short gaps
    df_filled = dfw.copy()
    if do_interpolate:
        num_cols = df_filled.select_dtypes(include=["number"]).columns
        for col in num_cols:
            s = df_filled[col]
            grp = s.isna().ne(s.isna().shift()).cumsum()
            run_len = s.isna().groupby(grp).transform("sum")
            allow_fill = s.isna() & (run_len <= max_gap_steps)
            s_interp = s.interpolate(method="time", limit_area="inside", limit_direction="forward")
            df_filled[col] = s_interp.where(allow_fill, s)

    df_clean = df_filled.dropna(subset=req_cols).copy()
    diag["n_rows_after"] = int(len(df_clean))

    if len(df_clean) < min_points:
        diag["reason"] = "too_few_points_after_clean"
        return None, diag

    diag["reason"] = "ok"
    return df_clean, diag


# ----------------------------
# DHW_sum computed from summer months INSIDE the chosen 12-month window
# ----------------------------
def compute_DHW_sum_from_window(
    df_window: pd.DataFrame,
    *,
    hp_diff_col: str = "Heat_Pump_Energy_Output_Diff",
    summer_months=(6, 7, 8),
    day_hours: int = 24,
    dhw_threshold_kW: float = 0.15,
) -> float:
    """
    Same idea as your old heuristic, but summer is relative to the selected window (no hard-coded year).
    Returns 0 if no usable summer data exists.
    """
    if df_window is None or df_window.empty:
        return 0.0
    if hp_diff_col not in df_window.columns:
        return 0.0

    df_summer = df_window[df_window.index.month.isin(list(summer_months))].copy()
    if df_summer.empty:
        return 0.0

    df_num = df_summer.select_dtypes(include="number")
    if df_num.empty or hp_diff_col not in df_num.columns:
        return 0.0

    df_daily = df_num.resample(f"{day_hours}h").mean()
    if df_daily.empty:
        return 0.0

    df_dhw_only = df_daily[df_daily[hp_diff_col] <= dhw_threshold_kW]
    if df_dhw_only.empty:
        return 0.0

    return float(df_dhw_only[hp_diff_col].sum() * 24.0)


# ----------------------------
# CVXPY model (same as your pareto version; weights fixed per run)
# ----------------------------
class InnerSolver:
    def __init__(self, N: int, q_max: float, delta_t: float):
        self.N = int(N)
        self.delta_t = float(delta_t)
        self.q_max = float(q_max)

        self.invC   = cp.Parameter(nonneg=True)
        self.ph_q   = cp.Parameter(nonneg=True)
        self.phi_e  = cp.Parameter(nonneg=True)
        self.phi_u  = cp.Parameter(nonneg=True)

        self.q_hp       = cp.Parameter(self.N)
        self.delta_T_i  = cp.Parameter(self.N)
        self.Q_sc       = cp.Parameter()

        self.X_a = cp.Parameter(self.N)  # delta_T_a * delta_t
        self.X_s = cp.Parameter(self.N)  # q_solar

        self.a     = cp.Variable(pos=True)
        self.w_s   = cp.Variable(nonneg=True)
        self.w     = cp.Variable()
        self.e     = cp.Variable()
        self.u     = cp.Variable(self.N)
        self.q_hat = cp.Variable(self.N)

        ones = np.ones(self.N)

        expr = (self.a * self.X_a
                + self.q_hat
                + self.w_s * self.X_s
                + self.w * ones * self.delta_t)

        constraints = [
            self.e == self.Q_sc - cp.sum(self.q_hat),
            self.delta_T_i + self.u == cp.multiply(expr, self.invC),
            self.q_hat >= 0,
            self.q_hat <= self.q_max,
            self.a >= 1/25,
            self.e >= 0
        ]

        obj = cp.Minimize(
            self.ph_q   * cp.norm1(self.q_hat - self.q_hp) * self.delta_t
            + self.phi_e * cp.square(self.e) * self.delta_t
            + self.phi_u * cp.norm2(self.u) * self.delta_t
        )

        self.prob = cp.Problem(obj, constraints)

    def solve(self, *, C, q_hp, delta_T_i, delta_T_a, q_solar, Q_sc, weights, solver=cp.GUROBI):
        C = float(C)
        self.invC.value = 1.0 / C

        self.q_hp.value = np.asarray(q_hp, dtype=float)
        self.delta_T_i.value = np.asarray(delta_T_i, dtype=float)
        self.Q_sc.value = float(Q_sc)

        self.X_a.value = np.asarray(delta_T_a, dtype=float) * self.delta_t
        self.X_s.value = np.asarray(q_solar, dtype=float)

        self.ph_q.value  = float(weights[0])
        self.phi_e.value = float(weights[1])
        self.phi_u.value = float(weights[2])

        self.prob.solve(
            solver=solver,
            warm_start=True,
            verbose=False,
            Threads=1
        )

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"CVXPY status={self.prob.status}")

        if any(v.value is None for v in [self.a, self.w_s, self.w, self.e, self.u, self.q_hat]):
            raise RuntimeError("CVXPY returned None variable values.")

        a_val = float(self.a.value)
        R_a = 1.0 / a_val

        ws_val = float(self.w_s.value)
        w_val = float(self.w.value)
        q_hat_val = np.asarray(self.q_hat.value).copy()

        rmse_dTi = float(np.sqrt(np.mean(np.square(self.u.value))))
        rmse_q = float(np.sqrt(np.mean(np.square(self.q_hat.value - self.q_hp.value))))

        cost_q = float(np.linalg.norm(self.q_hat.value - self.q_hp.value, 1) * self.delta_t)
        cost_e = float((self.e.value ** 2) * self.delta_t)
        cost_u = float(np.linalg.norm(self.u.value, 2) * self.delta_t)

        return R_a, ws_val, w_val, rmse_q, rmse_dTi, q_hat_val, cost_q, cost_e, cost_u


def _fit_worker(payload, out_q):
    """
    Subprocess worker: fits one home for fixed weights by scanning C_values.
    """
    try:
        import os
        import numpy as np
        import cvxpy as cp
        import psutil

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ.setdefault("MallocNanoZone", "0")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss / (1024**3)

        def rss_gb_local():
            return proc.memory_info().rss / (1024**3)

        def mem_alarm_local(growth_gb):
            return (rss_gb_local() - rss0) > growth_gb

        weights    = payload["weights"]
        C_values   = payload["C_values"]
        N          = payload["N"]
        q_max      = payload["q_max"]
        delta_t    = payload["delta_t"]
        q_hp       = payload["q_hp"]
        delta_T_i  = payload["delta_T_i"]
        delta_T_a  = payload["delta_T_a"]
        q_solar    = payload["q_solar"]
        Q_sc       = payload["Q_sc"]
        t_start = payload["t_start"]
        growth_gb  = payload.get("growth_gb", 1.5)
        solver_name = payload.get("solver_name", "GUROBI")

        solver = getattr(cp, solver_name)

        inner = InnerSolver(N=N, q_max=q_max, delta_t=delta_t)

        best_obj = (np.inf, np.inf)
        best_row = None

        delta_T_i_wk = payload.get("delta_T_i_wk", None)
        q_hp_wk      = payload.get("q_hp_wk", None)
        delta_T_a_wk = payload.get("delta_T_a_wk", None)   # ADD
        q_solar_wk   = payload.get("q_solar_wk", None)
        Ti_meas_state_wk = payload.get("Ti_meas_state_week", None)
        is_midnight_state_wk = payload.get("is_midnight_state_week", None)



        for C in C_values:
            if mem_alarm_local(growth_gb):
                inner = InnerSolver(N=N, q_max=q_max, delta_t=delta_t)
                rss0 = rss_gb_local()

            R_a, ws, w, rmse_q, rmse_dTi, q_hat_val, cost_q, cost_e, cost_u = inner.solve(
                C=C,
                q_hp=q_hp,
                delta_T_i=delta_T_i,
                delta_T_a=delta_T_a,
                q_solar=q_solar,
                Q_sc=Q_sc,
                weights=weights,
                solver=solver,
            )

            # ---- RMSE of Ti over the week, with midnight reinitialization ----
            if (Ti_meas_state_wk is None) or (is_midnight_state_wk is None) or \
                    (q_hp_wk is None) or (delta_T_a_wk is None) or (
                    q_solar_wk is None):
                rmse_Ti_val = np.inf
            else:
                Ti_meas = np.asarray(Ti_meas_state_wk, dtype=float)
                midn = np.asarray(is_midnight_state_wk, dtype=bool)

                # We need ΔTa*Δt, q_hp, q_solar aligned to the Ti timeline steps.
                # Assume these are same length as Ti_meas except possibly last step.
                L = len(Ti_meas)
                dTa = np.asarray(delta_T_a_wk, dtype=float)[:L - 1]
                qhp = np.asarray(q_hp_wk, dtype=float)[:L - 1]
                qso = np.asarray(q_solar_wk, dtype=float)[:L - 1]

                # simulate Ti_hat
                Ti_hat = np.empty(L, dtype=float)
                Ti_hat[:] = np.nan

                a = 1.0 / float(R_a)

                # initialize at first point
                Ti_hat[0] = Ti_meas[0] if np.isfinite(Ti_meas[0]) else np.nan

                for k in range(L - 1):
                    # midnight reinit at time k (state at k)
                    if midn[k] and np.isfinite(Ti_meas[k]):
                        Ti_hat[k] = Ti_meas[k]

                    if not np.isfinite(Ti_hat[k]):
                        continue

                    # your model in ΔTi form:
                    # ΔTi_hat = (a*(ΔTa*Δt) + q_hp + ws*q_solar + w*Δt) / C
                    dTi_hat = (a * (dTa[k] * delta_t) + qhp[k] + float(ws) *
                               qso[k] + float(w) * delta_t) / float(C)
                    Ti_hat[k + 1] = Ti_hat[k] + dTi_hat

                # compute error only where both are finite
                ok = np.isfinite(Ti_hat) & np.isfinite(Ti_meas)
                rmse_Ti_val = float(np.sqrt(np.mean(
                    (Ti_meas[ok] - Ti_hat[ok]) ** 2))) if ok.any() else np.inf

            obj = (rmse_dTi, rmse_q)
            if obj < best_obj:
                best_obj = obj
                best_row = {
                    "phi_q": float(weights[0]),
                    "phi_e": float(weights[1]),
                    "phi_u": float(weights[2]),
                    "C": float(C),
                    "R_a": float(R_a),
                    "w_s": float(ws),
                    "w": float(w),
                    "Q_hat": float(np.sum(q_hat_val)),
                    "Q_sc": float(Q_sc),
                    "rmse_dTi_train": float(rmse_dTi),
                    "rmse_q_hp_train": float(rmse_q),
                    "cost_q": float(cost_q),
                    "cost_e": float(cost_e),
                    "cost_u": float(cost_u),
                }

        out_q.put(("ok", best_row))

    except Exception:
        out_q.put(("err", traceback.format_exc()))


def run_fit_in_subprocess(
    *,
    weights,
    C_values,
    N, q_max, delta_t,
    q_hp, delta_T_i, delta_T_a, q_solar,
    Q_sc, t_start,
    growth_gb=1.5,
    solver_name="GUROBI",
    **extra_payload
):
    payload = dict(
        weights=np.asarray(weights, dtype=float),
        C_values=np.asarray(C_values, dtype=float),
        N=int(N),
        q_max=float(q_max),
        delta_t=float(delta_t),
        q_hp=np.asarray(q_hp, dtype=float),
        delta_T_i=np.asarray(delta_T_i, dtype=float),
        delta_T_a=np.asarray(delta_T_a, dtype=float),
        q_solar=np.asarray(q_solar, dtype=float),
        Q_sc=float(Q_sc),
        t_start = pd.to_datetime(t_start),
        growth_gb=float(growth_gb),
        solver_name=str(solver_name),
    )

    payload.update(extra_payload)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_fit_worker, args=(payload, q))
    p.start()
    p.join()

    if p.exitcode != 0:
        print(f"[SKIP] worker crashed (exitcode={p.exitcode})")
        return None

    if q.empty():
        print("[SKIP] worker returned no result")
        return None

    status, msg = q.get()
    if status == "err":
        print(f"[SKIP] worker exception:\n{msg}")
        return None

    return msg


# ----------------------------
# Fit one home (data reading / gating inspired by regression script)
# ----------------------------
def fit_one_home_fixed_weights(
    *,
    df_train_detached: pd.DataFrame,
    df_house: pd.DataFrame,
    id_use,
    weights_fixed,
    C_values,
    solver_name="GUROBI",
):
    # ---- home meta ----
    df_house_row = df_house[df_house["Property_ID"] == id_use]
    if df_house_row.empty:
        print(f"[SKIP HOME] {id_use}: missing in home_characteristics")
        return None

    # keep same "to_string" spirit as your regression script, but store clean values
    floor_area = float(df_house_row["Total_Floor_Area"].iloc[0]) if "Total_Floor_Area" in df_house_row.columns else np.nan
    storeys = float(df_house_row["No_Storeys"].iloc[0]) if "No_Storeys" in df_house_row.columns else np.nan
    wall = str(df_house_row["Wall_Type"].iloc[0]) if "Wall_Type" in df_house_row.columns else ""
    hp_type = str(df_house_row["HP_Installed"].iloc[0]) if "HP_Installed" in df_house_row.columns else ""
    house_sap = str(df_house_row["House_SAP"].iloc[0]) if "House_SAP" in df_house_row.columns else ""
    Q_DHW_estimate = float(df_house_row["MCS_DHWAnnual"].iloc[0]) if "MCS_DHWAnnual" in df_house_row.columns else 0.0
    q_max = float(df_house_row["HP_Size_kW"].iloc[0]) if "HP_Size_kW" in df_house_row.columns else 5.0

    # ---- slice house time series ----
    df_single = df_train_detached[df_train_detached["Property_ID"] == id_use].copy()
    if df_single.empty:
        print(f"[SKIP HOME] {id_use}: no rows in merged_data_final_homes.parquet")
        return None

    # Ensure timestamp is datetime
    df_single["Timestamp"] = pd.to_datetime(df_single["Timestamp"], errors="coerce")
    df_single = df_single.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # Compute diffs (same columns you used in regression script)
    # NOTE: Heat_Pump_Energy_Output_Diff corresponds to "power/energy per 30min step" in your data
    if "Heat_Pump_Energy_Output" not in df_single.columns:
        print(f"[SKIP HOME] {id_use}: missing Heat_Pump_Energy_Output")
        return None
    if "Internal_Air_Temperature" not in df_single.columns:
        print(f"[SKIP HOME] {id_use}: missing Internal_Air_Temperature")
        return None
    if "temp" not in df_single.columns:
        print(f"[SKIP HOME] {id_use}: missing ambient temp column 'temp'")
        return None
    if "solarradiation" not in df_single.columns:
        # allow, but then q_solar=0
        pass

    df_single["Heat_Pump_Energy_Output_Diff"] = df_single["Heat_Pump_Energy_Output"].diff()
    df_single["Heat_Pump_Energy_Output_Diff"] = df_single["Heat_Pump_Energy_Output_Diff"].clip(lower=0)

    df_single["Internal_Temperature_Diff"] = df_single["Internal_Air_Temperature"].diff()
    df_single["Internal_Ambient_Temperature_Diff"] = df_single["temp"] - df_single["Internal_Air_Temperature"]

    if "Immersion_Heater_Energy_Consumed" in df_single.columns:
        df_single["Immersion_Diff"] = df_single["Immersion_Heater_Energy_Consumed"].diff().clip(lower=0)
    else:
        df_single["Immersion_Diff"] = 0.0

    # Index by time like your cleaning function expects
    df_single = df_single.set_index("Timestamp").sort_index()

    # ---- Choose home-specific 12-month window (calendar year preferred) ----
    try:
        t_start, t_end, diag_win = find_annual_window_home(
            df_single.index,
            enforce_freq="30min",
            summer_months=(6, 7, 8),
            prefer_calendar_year=True,
            min_coverage_non_summer=0.70,
        )
    except Exception as e:
        print(f"[SKIP HOME] {id_use}: could not pick annual window: {e}")
        return None

    # ---- Clean/gate the WINDOW dataframe (still includes summer for DHW_sum computation) ----
    df_window_clean, diag_clean = clean_and_gate_window(
        df_single.reset_index(),   # function accepts with Timestamp col too
        t_start=t_start,
        t_end=t_end,
        enforce_freq="30min",
        max_gap_steps=6,
        do_interpolate=True,
        col_Ti="Internal_Air_Temperature",
        col_qhp_diff="Heat_Pump_Energy_Output_Diff",
        max_missing=0.20,
        min_points=1000,
    )

    if df_window_clean is None:
        print(f"[SKIP HOME] {id_use}: window gating failed: {diag_clean.get('reason')}")
        return None

    # Ensure key diffs exist (they might have been averaged/resampled; recompute diffs on cleaned series)
    # Recompute diffs AFTER resampling so they're consistent with 30min grid.
    dfw = df_window_clean.copy()
    dfw["Heat_Pump_Energy_Output_Diff"] = dfw["Heat_Pump_Energy_Output"].diff().clip(lower=0) if "Heat_Pump_Energy_Output" in dfw.columns else dfw["Heat_Pump_Energy_Output_Diff"].clip(lower=0)
    dfw["Internal_Temperature_Diff"] = dfw["Internal_Air_Temperature"].diff()
    dfw["Internal_Ambient_Temperature_Diff"] = dfw["temp"] - dfw["Internal_Air_Temperature"]
    if "Immersion_Heater_Energy_Consumed" in dfw.columns:
        dfw["Immersion_Diff"] = dfw["Immersion_Heater_Energy_Consumed"].diff().clip(lower=0)
    else:
        dfw["Immersion_Diff"] = 0.0

    # Drop the first diff NaNs
    dfw = dfw.dropna(subset=["Internal_Temperature_Diff", "Heat_Pump_Energy_Output_Diff"])

    # ---- Compute DHW_sum from SUMMER INSIDE this chosen window ----
    DHW_sum = compute_DHW_sum_from_window(
        dfw,
        hp_diff_col="Heat_Pump_Energy_Output_Diff",
        summer_months=(6, 7, 8),
        dhw_threshold_kW=0.15,
    )

    # ---- Training dataframe = window EXCLUDING summer months ----
    df_train = dfw[~dfw.index.month.isin([6, 7, 8])].copy()
    if df_train.empty or len(df_train) < 500:
        print(f"[SKIP HOME] {id_use}: too little non-summer data in chosen window")
        return None

    # Enforce contiguous 30-min steps (avoid May->Sep jumps)
    dt = df_train.index.to_series().diff()
    df_train = df_train[(dt.isna()) | (dt == pd.Timedelta(minutes=30))].copy()
    if len(df_train) < 500:
        print(f"[SKIP HOME] {id_use}: too little contiguous 30-min data after gap gating")
        return None

    # ---- Build arrays for CVXPY solver (aligned to your older pareto script) ----
    t_step = 30
    delta_t = t_step / 60

    # q_solar column name in merged_data_final_homes.parquet is "solarradiation"
    if "solarradiation" in df_train.columns:
        q_solar_series = df_train["solarradiation"]
    else:
        q_solar_series = pd.Series(0.0, index=df_train.index)

    delta_T_a = df_train["Internal_Ambient_Temperature_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    delta_T_i = (df_train["Internal_Temperature_Diff"].iloc[1:].reset_index(drop=True).to_numpy() / delta_t)
    q_hp = df_train["Heat_Pump_Energy_Output_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    q_solar = q_solar_series.iloc[:-1].reset_index(drop=True).to_numpy()

    N = len(delta_T_i)
    if N < 50:
        print(f"[SKIP HOME] {id_use}: N too small ({N})")
        return None

    # ---- Q_sc (same structure as your pareto script) ----
    # Use DHW_sum computed from summer inside the same window.
    Q_sc = float(np.sum(q_hp) - Q_DHW_estimate + DHW_sum)


    # adding week val ------
    if t_start.month > 2:
        year_use = t_start.year + 1
    else:
        year_use = t_start.year

    # Use df_train timestamps (not reset_index arrays) to choose a *good* Feb week.
    idx0 = pd.DatetimeIndex(df_train.index)

    # IMPORTANT: these signals must align to idx0[:-1] because several of your arrays use .iloc[:-1]
    idx_for_week = idx0[:-1]

    # Build “signal arrays” aligned with idx_for_week (same alignment as q_hp, delta_T_a, q_solar below)
    delta_T_a_all = df_train["Internal_Ambient_Temperature_Diff"].iloc[
        :-1].to_numpy(dtype=float)
    q_hp_all = df_train["Heat_Pump_Energy_Output_Diff"].iloc[:-1].to_numpy(
        dtype=float)
    q_solar_all = q_solar_series.iloc[:-1].to_numpy(dtype=float)

    # Pick week bounds (Feb of correct year) with simple missing-data gating
    t0_wk, t1_wk = pick_good_feb_week_bounds(
        t_start=t_start,
        idx=idx_for_week,
        signals={
            "delta_T_a": delta_T_a_all,
            "q_hp": q_hp_all,
            "q_solar": q_solar_all,
        },
        max_missing_frac=0.30,
        max_gap_steps=3,  # 1.5 hours at 30-min
    )
    if t0_wk is None:
        print(f"[SKIP HOME] {id_use}: no clean February week found")
        return None

    # Build slice mask on idx_for_week (step-aligned signals)
    wk_mask = (idx_for_week >= t0_wk) & (idx_for_week <= t1_wk)

    # Now build the week arrays you want to pass
    # NOTE: delta_T_i uses .iloc[1:], so we slice it on idx0[1:] separately with same (t0_wk, t1_wk)
    idx_for_dTi = idx0[1:]
    wk_mask_dTi = (idx_for_dTi >= t0_wk) & (idx_for_dTi <= t1_wk)

    delta_T_i_meas_week = (
                df_train["Internal_Temperature_Diff"].iloc[1:].to_numpy(
                    dtype=float) / delta_t)[wk_mask_dTi]
    delta_T_a_week = delta_T_a_all[wk_mask]
    q_solar_week = q_solar_all[wk_mask]
    q_real_week = q_hp_all[wk_mask]  # measured q_hp

    # ---- ADD Ti (state) week, plus midnight flags ----
    # State series is aligned to idx0 (not idx0[:-1] or idx0[1:])
    wk_mask_Ti = (idx0 >= t0_wk) & (idx0 <= t1_wk)
    Ti_meas_state_week = \
    df_train["Internal_Air_Temperature"].to_numpy(dtype=float)[wk_mask_Ti]

    idx_state_week = idx0[wk_mask_Ti]
    is_midnight_state_week = ((idx_state_week.hour == 0) & (idx_state_week.minute == 0)).astype(bool)


    week_payload = dict(
        # keep dTi-based pieces
        delta_T_i_meas_week=np.asarray(delta_T_i_meas_week, dtype=float),
        delta_T_a_week=np.asarray(delta_T_a_week, dtype=float),
        q_solar_week=np.asarray(q_solar_week, dtype=float),
        q_real_week=np.asarray(q_real_week, dtype=float),

        # new Ti-based pieces
        Ti_meas_state_week=np.asarray(Ti_meas_state_week, dtype=float),
        is_midnight_state_week=np.asarray(is_midnight_state_week, dtype=bool),
    )

    # (Optional but convenient) also provide keys your worker can consume directly
    week_payload.update(dict(
        delta_T_i_wk=week_payload["delta_T_i_meas_week"],
        delta_T_a_wk=week_payload["delta_T_a_week"],
        q_solar_wk=week_payload["q_solar_week"],
        q_hp_wk=week_payload["q_real_week"],

        Ti_meas_state_week=week_payload["Ti_meas_state_week"],
        is_midnight_state_week=week_payload["is_midnight_state_week"],
    ))

    # ------------------ END ADD PORTION OF CODE HERE ------------------


# ---- Solve once per home (fixed weights), scanning C_values ----
    best_row = run_fit_in_subprocess(
        weights=weights_fixed,
        C_values=C_values,
        N=N,
        q_max=q_max,
        delta_t=delta_t,
        q_hp=q_hp,
        delta_T_i=delta_T_i,
        delta_T_a=delta_T_a,
        q_solar=q_solar,
        Q_sc=Q_sc,
        t_start = t_start,
        growth_gb=1.5,
        solver_name=solver_name,
        **week_payload
    )

    if best_row is None:
        print(f"[SKIP HOME] {id_use}: CVXPY solve failed/crashed")
        return None

    # ---- Attach metadata + requested training window bounds ----
    best_row["Property_ID"] = id_use
    best_row["Floor Area"] = floor_area
    best_row["No_Storeys"] = storeys
    best_row["Wall_Type"] = wall
    best_row["HP_Type"] = hp_type
    best_row["House_SAP"] = house_sap
    best_row["HP_Size_kW"] = q_max
    best_row["MCS_DHWAnnual"] = Q_DHW_estimate

    # Save training window start/end (the chosen 12-month span)
    best_row["train_start"] = pd.Timestamp(t_start).isoformat()
    best_row["train_end"] = pd.Timestamp(t_end).isoformat()

    # Helpful diagnostics (optional but useful for debugging)
    best_row["window_kind"] = diag_win.get("window_kind")
    best_row["coverage_non_summer"] = diag_win.get("coverage_non_summer")
    best_row["DHW_sum_window_summer"] = float(DHW_sum)

    return best_row


# ----------------------------
# Main
# ----------------------------
def main():
    log_mem("start")

    # INPUT FILES (match your linear regression version)
    df_train_detached = pd.read_parquet(
        "../retrieved_weather_data/merged_data_final_homes.parquet")
    df_house = pd.read_csv("../retrieved_weather_data/home_characteristics.csv")

    unique_ids = df_train_detached["Property_ID"].unique()

    # Fixed weights requested
    weights_fixed = [450.0, 0.003, 1.0]#[450.0, 1.0, 300.0]

    # C scan (same as before)
    C_values = np.arange(1.5, 20, 1)

    rows = []
    log_mem("begin per-home fits")

    for n, id_use in enumerate(unique_ids, start=1):
        print(f"\n[HOME {n}/{len(unique_ids)}] Property_ID={id_use} weights={weights_fixed}")

        row = fit_one_home_fixed_weights(
            df_train_detached=df_train_detached,
            df_house=df_house,
            id_use=id_use,
            weights_fixed=weights_fixed,
            C_values=C_values,
            solver_name="GUROBI",   # change if needed
        )

        if row is not None:
            rows.append(row)

        gc.collect()
        if n % 25 == 0:
            log_mem(f"after {n} homes")

    log_mem("end per-home fits")

    if len(rows) == 0:
        print("No homes successfully fit.")
        return

    df_out = pd.DataFrame(rows).sort_values(["Property_ID"])
    out_csv = "trained_params_allhomes_fixedweights_linearstyle_v3.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved per-home trained parameters to {out_csv}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
