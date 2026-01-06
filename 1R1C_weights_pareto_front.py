# ============================
# pareto_subproc_run.py
# Copy/paste into a .py file and run in PyCharm
# ============================

import os

# Hard cap threads across common native libs (must be set before numpy/scipy import)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# On macOS, this often reduces malloc nano allocator weirdness in long native workloads
os.environ.setdefault("MallocNanoZone", "0")

# If you see OpenMP duplicate runtime issues on mac, this can prevent aborts
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

        # These are the "data vectors" used in the model
        self.X_a = cp.Parameter(self.N)   # delta_T_a * delta_t
        self.X_s = cp.Parameter(self.N)   # q_solar

        # Variables
        self.a     = cp.Variable(pos=True)
        self.w_s   = cp.Variable(nonneg=True)
        self.w     = cp.Variable()
        self.e     = cp.Variable()
        self.u     = cp.Variable(self.N)
        self.q_hat = cp.Variable(self.N)

        # Constant ones
        self.ones = np.ones(self.N)

        # Expression
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
        ]

        obj = cp.Minimize(
            self.ph_q  * cp.norm1(self.q_hat - self.q_hp) * self.delta_t
            + self.phi_e * cp.square(self.e) * self.delta_t
            + self.phi_u * cp.norm2(self.u) * self.delta_t
        )

        self.prob = cp.Problem(obj, constraints)
        print("InnerSolver built. DCP:", self.prob.is_dcp(), "DPP:", self.prob.is_dpp())

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

        rmse_dTi = float(np.sqrt(np.mean(np.square(self.u.value))))
        rmse_q = float(np.sqrt(np.mean(np.square(self.q_hat.value - self.q_hp.value))))

        cost_q = float(np.linalg.norm(self.q_hat.value - self.q_hp.value, 1) * self.delta_t)
        cost_e = float((self.e.value ** 2) * self.delta_t)
        cost_u = float(np.linalg.norm(self.u.value, 2) * self.delta_t)

        return R_a, ws_val, w_val, rmse_q, rmse_dTi, q_hat_val, cost_q, cost_e, cost_u

def _weights_worker(payload, out_q):
    """
    Top-level multiprocessing worker (must be picklable under spawn).
    payload is a dict of numpy arrays + scalars.
    """
    try:
        import os
        import numpy as np
        import psutil
        import cvxpy as cp
        import traceback

        # Re-apply thread caps inside subprocess too (spawn starts fresh)
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

        solver = getattr(cp, solver_name)

        inner = InnerSolver(N=N, q_max=q_max, delta_t=delta_t)

        best_obj = (np.inf, np.inf)
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

            R_a, ws, w, rmse_q, rmse_dTi, q_hat_val, cost_q, cost_e, cost_u = out
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
                best_q_hat = np.asarray(q_hat_val, dtype=np.float32)

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
    growth_gb=1.5,
    solver_name="GUROBI",
):
    """
    Runs one weight triple over all C_values in a clean subprocess.
    If cvxcore aborts, the subprocess dies and parent skips weights.
    Returns (best_row_dict, best_q_hat_float32) or None.
    """
    import multiprocessing as mp

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
        growth_gb=float(growth_gb),
        solver_name=str(solver_name),
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

    # ---- load data (your paths) ----
    df_train_q_exact = pd.read_parquet("retrieved_weather_data/q_streams_30min.parquet")
    df_train_detached = pd.read_parquet("training_data/data_detached_with_weather.parquet")
    df_house = pd.read_csv("training_data/home_characteristics.csv")

    unique_ids = df_train_detached["Property_ID"].unique()
    id_use = unique_ids[5]

    # ---- home_dict like yours ----
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

    # ---- build annual training arrays (copied structure from your notebook) ----
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

    # ---- DHW_sum / SPC_sum (your method) ----
    df_heating_single = df_single.copy()
    t_start_summer = pd.to_datetime("2022-06-01 00:00:00")
    t_end_summer = pd.to_datetime("2022-08-30 23:59:00")
    df_DHW = df_heating_single.loc[t_start_summer:t_end_summer].copy()

    for c in ["Property_ID", "half-hour", "Date", "has_data"]:
        if c in df_DHW.columns:
            df_DHW.drop(c, axis=1, inplace=True)

    i_hours = 24
    df_numeric = df_DHW.select_dtypes(include="number")
    df_resampled = df_numeric.resample(f"{i_hours}h").mean()

    df_DHW_only = df_resampled[df_resampled["Heat_Pump_Energy_Output_Diff"] <= 0.15]
    DHW_sum = float(df_DHW_only["Heat_Pump_Energy_Output_Diff"].sum() * 24)

    df_spc_only = df_resampled[df_resampled["Heat_Pump_Energy_Output_Diff"] > 0.15]
    SPC_sum = float(df_spc_only["Heat_Pump_Energy_Output_Diff"].sum() * 24)

    # ---- annual training window (same as you) ----
    t_start = pd.to_datetime("2022-01-01 00:00:00")
    t_mid_end = pd.to_datetime("2022-05-31 23:59:59")
    t_mid_start = pd.to_datetime("2022-08-31 00:00:00")
    t_end = pd.to_datetime("2022-12-31 23:59:59")

    df_heating_annual = df_heating_single.loc[t_start:t_end].copy()
    df_heating_annual = df_heating_annual[(df_heating_annual.index <= t_mid_end) | (df_heating_annual.index >= t_mid_start)]

    t_step = 30
    delta_t = t_step / 60

    delta_T_a = df_heating_annual["Internal_Ambient_Temperature_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    delta_T_i = (df_heating_annual["Internal_Temperature_Diff"].iloc[1:].reset_index(drop=True).to_numpy() / delta_t)
    q_hp = df_heating_annual["Heat_Pump_Energy_Output_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    q_solar = df_heating_annual["SolarRadiation"].iloc[:-1].reset_index(drop=True).to_numpy()

    N = len(delta_T_i)

    # ---- validation window (your Jan 2022 block) ----
    t_start_val = pd.to_datetime("2022-01-01 00:00:00")
    t_end_val = pd.to_datetime("2022-01-30 23:59:00")

    df_heating_val = df_heating_single.loc[t_start_val:t_end_val].copy()
    df_30min_val = read_q_exact_30min(df_train_q_exact, id_use, t_start_val, t_end_val)

    delta_T_a_val = df_heating_val["Internal_Ambient_Temperature_Diff"].iloc[:-1].reset_index(drop=True).to_numpy()
    delta_T_i_val = (df_heating_val["Internal_Temperature_Diff"].iloc[1:].reset_index(drop=True).to_numpy() / delta_t)
    q_solar_val = df_heating_val["SolarRadiation"].iloc[:-1].reset_index(drop=True).to_numpy()
    T_i_val = df_heating_val["Internal_Air_Temperature"].iloc[:-1].reset_index(drop=True).to_numpy()
    q_real_val = df_30min_val["Q_spc_exact"].iloc[:-1].reset_index(drop=True).to_numpy()

    # Build validation indices from training time base (works with duplicate timestamps)
    train_time = df_heating_annual.index[:-1]
    tmask_val = (train_time >= t_start_val) & (train_time <= t_end_val)
    val_idx = np.flatnonzero(tmask_val)
    val_idx = val_idx[:-1]  # mimic your .iloc[:-1] behavior

    # Hard alignment check
    if len(val_idx) != len(delta_T_i_val):
        raise RuntimeError(f"val_idx len mismatch: {len(val_idx)} vs delta_T_i_val {len(delta_T_i_val)}")

    # ---- weights grid (your style) ----
    phi_q = np.arange(0, 500, 50)
    phi_q[0] = 1
    phi_e = 1 / phi_q
    phi_u = phi_q

    C_values = np.arange(1.5, 20, 1)

    # ---- store results here (parent survives crashes) ----
    rows = []

    # Shared scalar for Q_sc
    Q_sc = float(np.sum(q_hp) - Q_DHW_estimate + DHW_sum)

    log_mem("begin weight search (subprocess)")

    # IMPORTANT: keep weight loops modest at first
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
                    growth_gb=1.5,
                    solver_name="GUROBI",
                )

                if res is None:
                    # crashed or failed => skip
                    continue

                best_row, best_q_hat = res

                # ---- validation (parent only; no cvxpy here) ----
                C_best = best_row["C"]
                R_a_best = best_row["R_a"]
                w_s_best = best_row["w_s"]
                w_best = best_row["w"]

                q_opt = best_q_hat[val_idx].astype(float, copy=False)

                # Your simulated dTi model
                delta_T_i_sim = (
                    (delta_T_a_val / R_a_best * delta_t + q_opt + w_s_best * q_solar_val + w_best * np.ones_like(q_opt) * delta_t)
                    / C_best
                )

                # Integrate Ti, with midnight reset like you did
                T_i_sim = np.zeros_like(delta_T_i_sim)
                T_i_prev = float(T_i_val[0])

                # Build timestamps aligned to delta_T_i_sim
                time_idx = df_heating_val.index[1:]  # same length as delta_T_i_val
                if len(time_idx) != len(delta_T_i_sim):
                    raise RuntimeError("time_idx length mismatch in validation integration")

                for idx, t in enumerate(time_idx):
                    if t.hour == 0 and t.minute == 0:
                        # reset from measurement
                        # note: df_heating_val index aligns with original timestamps
                        T_i_prev = float(df_heating_val["Internal_Air_Temperature"].loc[t])
                    T_i_sim[idx] = T_i_prev + delta_T_i_sim[idx] * delta_t
                    T_i_prev = float(T_i_sim[idx])

                # q_hat_sim (your algebra)
                q_hat_sim = (
                    delta_T_i_val * C_best
                    - (delta_T_a_val / R_a_best) * delta_t
                    - w_s_best * q_solar_val
                    - w_best * np.ones_like(q_real_val) * delta_t
                )

                rmse_q_val = float(np.sqrt(np.mean((q_hat_sim - q_real_val) ** 2)))
                rmse_Ti_val = float(np.sqrt(np.mean((T_i_sim - df_heating_val["Internal_Air_Temperature"].iloc[1:].to_numpy()) ** 2)))

                best_row["Floor Area"] = float(floor_area)
                best_row["No_Storeys"] = float(storeys)
                best_row["Wall_Type"] = wall
                best_row["rmse_q_hp_val"] = rmse_q_val
                best_row["rmse_dTi_val"] = rmse_Ti_val  # keeping your naming convention

                rows.append(best_row)

                print(f"  -> val RMSE(q)={rmse_q_val:.4f}, val RMSE(Ti)={rmse_Ti_val:.4f}")
                log_mem("after one weights")

                # keep parent memory in check
                gc.collect()

    log_mem("end weight search")

    # ---- Results dataframe ----
    if len(rows) == 0:
        print("No successful weight runs to plot.")
        return

    df_trained_val = pd.DataFrame(rows)

    # ---- Scatter plot (same idea as your notebook) ----
    plt.figure()
    plt.scatter(df_trained_val["rmse_dTi_val"], df_trained_val["rmse_q_hp_val"])
    plt.grid(True)
    plt.xlabel("Validation RMSE of Ti")
    plt.ylabel("Validation RMSE of q")
    plt.title("Pareto front over weight sets")

    out_png = "pareto_scatter.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved scatter plot to {out_png}")

    out_csv = "trained_params_val.csv"
    df_trained_val.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    # Critical for macOS/PyCharm multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
