## ============================
# pareto_subproc_run.py
# Uses 1-week validation RMSE(q_real) + RMSE(Ti with midnight reinit)
# to choose best C per weight triple
# ============================

import os

# Hard cap threads across common native libs (must be set before numpy/scipy import)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MallocNanoZone", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ----------------------------
# Imports
# ----------------------------
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")  # safe for PyCharm + long runs
import matplotlib.pyplot as plt

import multiprocessing as mp
import traceback
import psutil
import gc


# ----------------------------
# Memory logging (parent)
# ----------------------------
PROC = psutil.Process(os.getpid())

def rss_gb() -> float:
    return PROC.memory_info().rss / (1024**3)

def log_mem(tag: str) -> None:
    print(f"[mem] {tag:30s} RSS={rss_gb():.2f} GB")


# ----------------------------
# Data helpers
# ----------------------------
def read_q_exact_30min(df_train_q_exact: pd.DataFrame,
                       id_use,
                       t_start: pd.Timestamp,
                       t_end: pd.Timestamp) -> pd.DataFrame:
    """
    df_train_q_exact: wide DataFrame with
      index = Timestamp (30-min)
      columns = MultiIndex [Property_ID, variable]
        variable in {"Q_hp_total","Q_immersion","Q_dhw","Q_hp_sc","Q_total"}

    Returns a single-home DataFrame indexed by Timestamp with flat columns.
    Also creates:
      - Q_dhw_exact = Q_dhw
      - Q_spc_exact = Q_total - Q_dhw
    """
    df_win = df_train_q_exact.loc[t_start:t_end]

    if not isinstance(df_win.columns, pd.MultiIndex):
        raise ValueError("Expected df_train_q_exact columns to be MultiIndex [Property_ID, variable].")

    if id_use not in df_win.columns.get_level_values(0):
        raise KeyError(f"id_use={id_use} not found in df_train_q_exact columns level 0 (Property_ID).")

    df_id = df_win.loc[:, (id_use, slice(None))].copy()
    df_id.columns = df_id.columns.droplevel(0)

    if "Q_dhw" in df_id.columns:
        df_id["Q_dhw_exact"] = df_id["Q_dhw"]

    if "Q_total" in df_id.columns and "Q_dhw" in df_id.columns:
        df_id["Q_spc_exact"] = df_id["Q_total"] - df_id["Q_dhw"]

    return df_id


# ----------------------------
# CVXPY reusable model
# ----------------------------
class InnerSolver:
    def __init__(self, N: int, q_max: float, delta_t: float):
        self.N = int(N)
        self.delta_t = float(delta_t)
        self.q_max = float(q_max)

        # Parameters
        self.invC   = cp.Parameter(nonneg=True)
        self.ph_q   = cp.Parameter(nonneg=True)
        self.phi_e  = cp.Parameter(nonneg=True)
        self.phi_u  = cp.Parameter(nonneg=True)

        self.q_hp       = cp.Parameter(self.N)
        self.delta_T_i  = cp.Parameter(self.N)
        self.Q_sc       = cp.Parameter()

        self.X_a = cp.Parameter(self.N)   # delta_T_a * delta_t
        self.X_s = cp.Parameter(self.N)   # q_solar

        # Variables
        self.a     = cp.Variable(pos=True)
        self.w_s   = cp.Variable(nonneg=True)
        self.w     = cp.Variable()
        self.e     = cp.Variable()
        self.u     = cp.Variable(self.N)
        self.q_hat = cp.Variable(self.N)

        self.ones = np.ones(self.N)

        expr = (self.a * self.X_a
                + self.q_hat
                + self.w_s * self.X_s
                + self.w * self.ones * self.delta_t)

        constraints = [
            self.e == self.Q_sc - cp.sum(self.q_hat),
            self.delta_T_i + self.u == cp.multiply(expr, self.invC),
            self.q_hat >= 0,
            self.q_hat <= self.q_max,
            self.a >= 1/25,
            self.e >= 0,
        ]

        obj = cp.Minimize(
            self.ph_q  * cp.norm1(self.q_hat - self.q_hp) * self.delta_t
            #+ self.phi_e * cp.square(self.e) * self.delta_t
            + self.phi_e * self.e * self.delta_t
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
            raise RuntimeError(f"CVXPY status={self.prob.status}")

        if any(v.value is None for v in [self.a, self.w_s, self.w, self.e, self.u, self.q_hat]):
            raise RuntimeError("CVXPY returned None variable values.")

        a_val = float(self.a.value)
        R_a = 1.0 / a_val
        ws_val = float(self.w_s.value)
        w_val = float(self.w.value)
        q_hat_val = np.asarray(self.q_hat.value).copy()

        # training metrics (kept, but NOT used for selection)
        rmse_dTi = float(np.sqrt(np.mean(np.square(self.u.value))))
        rmse_q = float(np.sqrt(np.mean(np.square(self.q_hat.value - self.q_hp.value))))

        cost_q = float(np.linalg.norm(self.q_hat.value - self.q_hp.value, 1) * self.delta_t)
        cost_e = float((self.e.value ** 2) * self.delta_t)
        cost_u = float(np.linalg.norm(self.u.value, 2) * self.delta_t)

        return R_a, ws_val, w_val, rmse_q, rmse_dTi, q_hat_val, cost_q, cost_e, cost_u


# ----------------------------
# Validation metrics (week)
# ----------------------------
def week_rmse_metrics(
    *,
    C: float, R_a: float, w_s: float, w: float,
    delta_t: float,
    # measured terms for the week
    delta_T_i_meas_week: np.ndarray,   # (L,) measured dTi (per hour)
    delta_T_a_week: np.ndarray,        # (L,) (Ta - Ti) or your delta_T_a
    q_solar_week: np.ndarray,          # (L,)
    # "real" week load stream
    q_real_week: np.ndarray,           # (L,) Q_spc_exact aligned with iloc[:-1]
    # measured Ti to compare simulation against
    Ti_meas_state_week: np.ndarray,    # (L,) corresponds to df_val["Ti"].iloc[1:]
    # integration initial state and midnight reset flags
    Ti0: float,
    is_midnight_state_week: np.ndarray # (L,) flags for state timestamps
):
    # q_hat implied by rearranging the model using *measured* delta_T_i
    q_hat_sim_week = (
        delta_T_i_meas_week * C
        - (delta_T_a_week / R_a) * delta_t
        - w_s * q_solar_week
        - w * np.ones_like(delta_T_i_meas_week) * delta_t
    )

    rmse_q_week = float(np.sqrt(np.mean((q_hat_sim_week - q_real_week) ** 2)))

    # Ti simulation driven by *real* q_real_week (your request)
    delta_T_i_sim_week = (
        (delta_T_a_week / R_a) * delta_t
        + q_real_week
        + w_s * q_solar_week
        + w * np.ones_like(q_real_week) * delta_t
    ) / C

    Ti_sim_week = np.zeros_like(delta_T_i_sim_week)
    Ti_prev = float(Ti0)

    for k in range(len(delta_T_i_sim_week)):
        if bool(is_midnight_state_week[k]):
            # reset Ti_prev to measured at that state timestamp
            Ti_prev = float(Ti_meas_state_week[k])
        Ti_sim_week[k] = Ti_prev + float(delta_T_i_sim_week[k]) * delta_t
        Ti_prev = float(Ti_sim_week[k])

    rmse_Ti_week = float(np.sqrt(np.mean((Ti_sim_week - Ti_meas_state_week) ** 2)))

    return rmse_q_week, rmse_Ti_week


def _weights_worker(payload, out_q):
    """
    Top-level multiprocessing worker (must be picklable under spawn).
    """
    try:
        import os
        import numpy as np
        import psutil
        import cvxpy as cp
        import traceback

        # thread caps inside subprocess
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

        # unpack payload
        weights   = payload["weights"]
        C_values  = payload["C_values"]
        N         = payload["N"]
        q_max     = payload["q_max"]
        delta_t   = payload["delta_t"]
        q_hp      = payload["q_hp"]
        delta_T_i = payload["delta_T_i"]
        delta_T_a = payload["delta_T_a"]
        q_solar   = payload["q_solar"]
        Q_sc      = payload["Q_sc"]
        growth_gb = payload.get("growth_gb", 1.5)
        solver_name = payload.get("solver_name", "GUROBI")

        # week validation arrays
        wk = payload["week"]
        delta_T_i_meas_week = wk["delta_T_i_meas_week"]
        delta_T_a_week      = wk["delta_T_a_week"]
        q_solar_week        = wk["q_solar_week"]
        q_real_week         = wk["q_real_week"]
        Ti_meas_state_week  = wk["Ti_meas_state_week"]
        Ti0                 = wk["Ti0"]
        is_midnight_state_week = wk["is_midnight_state_week"]

        alpha_T = float(payload.get("alpha_T", 1.0))
        alpha_q = float(payload.get("alpha_q", 1.0))

        solver = getattr(cp, solver_name)

        inner = InnerSolver(N=N, q_max=q_max, delta_t=delta_t)

        best_score = np.inf
        best_row = None
        best_q_hat = None

        for C in C_values:
            if mem_alarm_local(growth_gb):
                inner = InnerSolver(N=N, q_max=q_max, delta_t=delta_t)
                rss0 = rss_gb_local()

            out = inner.solve(
                C=C,
                q_hp=q_hp,
                delta_T_i=delta_T_i,
                delta_T_a=delta_T_a,
                q_solar=q_solar,
                Q_sc=Q_sc,
                weights=weights,
                solver=solver,
            )

            R_a, ws, w, rmse_q_train, rmse_dTi_train, q_hat_val, cost_q, cost_e, cost_u = out
            '''
            # ---- WEEK validation selection metrics (the point of this update) ----
            rmse_q_week, rmse_Ti_week = week_rmse_metrics(
                C=float(C), R_a=float(R_a), w_s=float(ws), w=float(w),
                delta_t=float(delta_t),
                delta_T_i_meas_week=delta_T_i_meas_week,
                delta_T_a_week=delta_T_a_week,
                q_solar_week=q_solar_week,
                q_real_week=q_real_week,
                Ti_meas_state_week=Ti_meas_state_week,
                Ti0=float(Ti0),
                is_midnight_state_week=is_midnight_state_week,
            )
            '''

            #score = alpha_T * rmse_Ti_week + alpha_q * rmse_q_week
            score = alpha_T * rmse_dTi_train + alpha_q * rmse_q_train


            if score < best_score:
                best_score = score
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

                    # keep train metrics for reference
                    "rmse_dTi_train": float(rmse_dTi_train),
                    "rmse_q_hp_train": float(rmse_q_train),

                    # NEW: week validation metrics (used for selection!)
                    #"rmse_q_week": float(rmse_q_week),
                    #"rmse_Ti_week": float(rmse_Ti_week),
                    "score_train": float(score),

                    "cost_q": float(cost_q),
                    "cost_e": float(cost_e),
                    "cost_u": float(cost_u),
                }
                best_q_hat = np.asarray(q_hat_val, dtype=np.float32)

        if best_row is None or best_q_hat is None:
            out_q.put(("ok", (None, None)))
            return

        rmse_q_week, rmse_Ti_week = week_rmse_metrics(
            C=best_row["C"], R_a=best_row["R_a"], w_s=best_row["w_s"],
            w=best_row["w"], delta_t=float(delta_t),
            delta_T_i_meas_week=delta_T_i_meas_week,
            delta_T_a_week=delta_T_a_week,
            q_solar_week=q_solar_week,
            q_real_week=q_real_week,
            Ti_meas_state_week=Ti_meas_state_week,
            Ti0=float(Ti0),
            is_midnight_state_week=is_midnight_state_week,
        )

        best_row["rmse_q_week"] = float(rmse_q_week)
        best_row["rmse_Ti_week"] = float(rmse_Ti_week)

        out_q.put(("ok", (best_row, best_q_hat)))

    except Exception:
        out_q.put(("err", traceback.format_exc()))


# ----------------------------
# Subprocess runner
# ----------------------------
def run_weights_in_subprocess(
    *,
    weights,
    C_values,
    N, q_max, delta_t,
    q_hp, delta_T_i, delta_T_a, q_solar,
    Q_sc,
    week_payload,
    growth_gb=1.5,
    solver_name="GUROBI",
    alpha_T=1.0,
    alpha_q=1.0,
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
        week=week_payload,
        growth_gb=float(growth_gb),
        solver_name=str(solver_name),
        alpha_T=float(alpha_T),
        alpha_q=float(alpha_q),
    )

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_weights_worker, args=(payload, q))
    p.start()
    p.join()

    if p.exitcode != 0:
        print(f"[SKIP WEIGHTS] weights={weights} worker crashed (exitcode={p.exitcode})")
        return None

    if q.empty():
        print(f"[SKIP WEIGHTS] weights={weights} worker returned no result")
        return None

    status, payload_out = q.get()
    if status == "err":
        print(f"[SKIP WEIGHTS] weights={weights} worker exception:\n{payload_out}")
        return None

    best_row, best_q_hat = payload_out
    if best_row is None or best_q_hat is None:
        print(f"[SKIP WEIGHTS] weights={weights} no feasible solution")
        return None

    return best_row, best_q_hat


# ----------------------------
# Main run
# ----------------------------
def main():
    log_mem("start")

    # ---- load data ----
    df_train_q_exact = pd.read_parquet(
        "../retrieved_weather_data/q_streams_30min.parquet")
    df_train_detached = pd.read_parquet(
        "../training_data/data_detached_with_weather.parquet")
    df_house = pd.read_csv("../training_data/home_characteristics.csv")

    unique_ids = df_train_detached["Property_ID"].unique()
    id_use = unique_ids[5]  # keep consistent with your notebook

    # ---- home_dict ----
    cp_air = 1.005
    rho_air = 1.225
    kj_to_kWh = 1/3600
    df_house["Volume"] = df_house["Total_Floor_Area"] * df_house["Floor_Height"]
    df_house["Min Capacity"] = df_house["Volume"] * cp_air * rho_air * kj_to_kWh
    df_home_values = df_house[df_house["Property_ID"].isin(unique_ids)]
    home_dict = df_home_values.set_index("Property_ID").to_dict(orient="index")

    wall = home_dict[id_use]["Wall_Type"]
    storeys = home_dict[id_use]["No_Storeys"]
    floor_area = home_dict[id_use]["Total_Floor_Area"]
    Q_DHW_estimate = float(home_dict[id_use]["MCS_DHWAnnual"])
    q_max = float(home_dict[id_use]["HP_Size_kW"])

    # ---- clean one home's dataframe ----
    df_single = df_train_detached[df_train_detached["Property_ID"] == id_use].copy()

    df_single["Heat_Pump_Energy_Output_Diff"] = df_single["Heat_Pump_Energy_Output"].diff()
    df_single["Internal_Temperature_Diff"] = df_single["Internal_Air_Temperature"].diff()
    df_single["Internal_Ambient_Temperature_Diff"] = (
        df_single["External_Air_Temperature"] - df_single["Internal_Air_Temperature"]
    )

    threshold = 0.90 * len(df_single)
    df_single = df_single.dropna(axis=1, thresh=threshold)

    df_single = df_single.set_index("Timestamp").sort_index()

    numeric_cols = df_single.select_dtypes(include=["number"]).columns
    df_single[numeric_cols] = df_single[numeric_cols].interpolate(
        method="time", limit=4, limit_direction="both"
    )
    df_single = df_single.dropna()

    # ---- DHW_sum / SPC_sum ----
    df_heating_single = df_single.copy()
    t_start_summer = pd.to_datetime("2022-06-01 00:00:00")
    t_end_summer = pd.to_datetime("2022-08-30 23:59:00")
    df_DHW = df_heating_single.loc[t_start_summer:t_end_summer].copy()

    for c in ["Property_ID", "half-hour", "Date", "has_data"]:
        if c in df_DHW.columns:
            df_DHW.drop(c, axis=1, inplace=True)

    df_resampled = df_DHW.select_dtypes(include="number").resample("24h").mean()
    DHW_sum = float(df_resampled.loc[df_resampled["Heat_Pump_Energy_Output_Diff"] <= 0.15,
                                     "Heat_Pump_Energy_Output_Diff"].sum() * 24)
    SPC_sum = float(df_resampled.loc[df_resampled["Heat_Pump_Energy_Output_Diff"] > 0.15,
                                     "Heat_Pump_Energy_Output_Diff"].sum() * 24)

    # ---- annual training window ----
    t_start = pd.to_datetime("2022-01-01 00:00:00")
    t_mid_end = pd.to_datetime("2022-05-31 23:59:59")
    t_mid_start = pd.to_datetime("2022-08-31 00:00:00")
    t_end = pd.to_datetime("2022-12-31 23:59:59")

    df_heating_annual = df_heating_single.loc[t_start:t_end].copy()
    df_heating_annual = df_heating_annual[
        (df_heating_annual.index <= t_mid_end) | (df_heating_annual.index >= t_mid_start)
    ]

    t_step = 30
    delta_t = t_step / 60

    delta_T_a = df_heating_annual["Internal_Ambient_Temperature_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    delta_T_i = (df_heating_annual["Internal_Temperature_Diff"].iloc[1:].reset_index(drop=True).to_numpy() / delta_t)
    q_hp = df_heating_annual["Heat_Pump_Energy_Output_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    q_solar = df_heating_annual["SolarRadiation"].iloc[:-1].reset_index(drop=True).to_numpy()

    N = len(delta_T_i)

    # ---- validation window (month) ----
    t_start_val = pd.to_datetime("2023-02-01 00:00:00")
    t_end_val = pd.to_datetime("2023-02-14 23:59:00")

    df_heating_val = df_heating_single.loc[t_start_val:t_end_val].copy()
    df_30min_val = read_q_exact_30min(df_train_q_exact, id_use, t_start_val, t_end_val)

    # arrays for the month
    delta_T_a_val = df_heating_val["Internal_Ambient_Temperature_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    delta_T_i_val = (df_heating_val["Internal_Temperature_Diff"].iloc[1:].reset_index(drop=True).to_numpy() / delta_t)
    q_solar_val = df_heating_val["SolarRadiation"].iloc[:-1].reset_index(drop=True).to_numpy()
    T_i_val_0 = df_heating_val["Internal_Air_Temperature"].iloc[:-1].reset_index(drop=True).to_numpy()
    q_real_val = df_30min_val["Q_spc_exact"].iloc[:-1].reset_index(drop=True).to_numpy()

    # state for month is Ti(t+1) side
    Ti_meas_state_val = df_heating_val["Internal_Air_Temperature"].iloc[1:].to_numpy()

    # midnight flags for month "state timestamps"
    state_times = pd.DatetimeIndex(df_heating_val.index[1:])
    is_midnight_state = (state_times.hour == 0) & (state_times.minute == 0)

    # ---- choose 1-week slice inside validation month ----
    week_days = 7
    L_week = int(week_days * 24 * (60 // t_step))  # 7*24*2 = 336 for 30-min

    # guard: ensure enough length
    if len(delta_T_i_val) < L_week:
        raise RuntimeError(f"Validation window too short for {week_days} days: got {len(delta_T_i_val)} steps")

    # we take the FIRST week of validation month by default
    sl = slice(0, L_week)

    week_payload = dict(
        delta_T_i_meas_week=np.asarray(delta_T_i_val[sl], dtype=float),
        delta_T_a_week=np.asarray(delta_T_a_val[sl], dtype=float),
        q_solar_week=np.asarray(q_solar_val[sl], dtype=float),
        q_real_week=np.asarray(q_real_val[sl], dtype=float),
        Ti_meas_state_week=np.asarray(Ti_meas_state_val[sl], dtype=float),
        Ti0=float(T_i_val_0[0]),
        is_midnight_state_week=np.asarray(is_midnight_state[sl], dtype=bool),
    )

    # ---- weights grid ----
    phi_q = np.arange(0, 500, 50)
    phi_q[0] = 1
    phi_e = 1 / phi_q
    phi_u = phi_q

    C_values = np.arange(1.5, 20, 1)

    # scalarization weights for picking best C (inside each weight triple)
    alpha_T = 1.0
    alpha_q = 1.0

    rows = []

    Q_sc = float(np.sum(q_hp) - Q_DHW_estimate + DHW_sum)

    log_mem("begin weight search (subprocess)")

    for i in phi_q:
        for j in phi_e:
            for k in phi_u:
                weights = [float(i), float(j), float(k)]
                print(f"\n[WEIGHTS] {weights}")

                res = run_weights_in_subprocess(
                    weights=weights,
                    C_values=C_values,
                    N=N,
                    q_max=q_max,
                    delta_t=delta_t,
                    q_hp=q_hp,
                    delta_T_i=delta_T_i,
                    delta_T_a=delta_T_a,
                    q_solar=q_solar,
                    Q_sc=Q_sc,
                    week_payload=week_payload,
                    growth_gb=1.5,
                    solver_name="GUROBI",
                    alpha_T=alpha_T,
                    alpha_q=alpha_q,
                )

                if res is None:
                    continue

                best_row, best_q_hat = res

                # attach house metadata
                best_row["Floor Area"] = float(floor_area)
                best_row["No_Storeys"] = float(storeys)
                best_row["Wall_Type"] = wall

                rows.append(best_row)

                print(f"  -> WEEK rmse_q={best_row['rmse_q_week']:.4f}, "
                      f"WEEK rmse_Ti={best_row['rmse_Ti_week']:.4f}, score="
                      f"{best_row['score_train']:.4f}")
                log_mem("after one weights")
                gc.collect()

    log_mem("end weight search")

    if len(rows) == 0:
        print("No successful weight runs to plot.")
        return

    df_trained = pd.DataFrame(rows)

    # Scatter uses WEEK metrics now (what you asked)
    plt.figure()
    plt.scatter(df_trained["rmse_Ti_week"], df_trained["rmse_q_week"])
    plt.grid(True)
    plt.xlabel("Week Validation RMSE of Ti")
    plt.ylabel("Week Validation RMSE of q (Q_spc_exact)")
    plt.title("Pareto front over weight sets (week validation)")

    out_png = "pareto_scatter_home5_val_week_v3.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved scatter plot to {out_png}")

    out_csv = "trained_params_home5_val_week_v3.csv"
    df_trained.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
