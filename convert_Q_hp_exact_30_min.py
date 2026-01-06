import numpy as np
import pandas as pd
import time


def export_q_streams_30min_from_multiindex(
    in_path_2min: str = "detached_home_data.parquet",
    in_path_allowed: str = "retrieved_weather_data/merged_data_final_homes.parquet",
    out_path: str = "q_streams_30min.parquet",
    id_col_allowed: str = "Property_ID",
    # source columns in detached_home_data.parquet
    col_hp_cum: str = "Heat_Pump_Energy_Output",
    col_im_cum: str = "Immersion_Heater_Energy_Consumed",
    col_dhw_flag: str = "Hot_Water_Flow_Temperature",
    freq: str = "30min",
    # resample aggregation: for step-energy, SUM is correct
    energy_mode: str = "sum",  # "sum" or "mean"
    # missingness / quality control
    min_coverage_frac: float = 0.5,      # require >= 50% of expected 2-min samples in a 30-min bin
    max_gap: pd.Timedelta = pd.Timedelta("2h"),  # if too far from any sample, treat as missing (optional)
    print_every: int = 10,
    verbose_id: bool = False,
) -> str:
    t_all = time.time()

    print(f"[stage] loading allowed ids: {in_path_allowed}", flush=True)
    df_allowed = pd.read_parquet(in_path_allowed)
    allowed_ids = set(df_allowed[id_col_allowed].unique())
    print(f"[stage] allowed_ids loaded: {len(allowed_ids):,}", flush=True)

    print(f"[stage] reading 2-min parquet: {in_path_2min}", flush=True)
    df = pd.read_parquet(in_path_2min)
    print(f"[stage] read done: shape={df.shape}", flush=True)

    # Expect MultiIndex: (Property_ID, Timestamp)
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
        raise ValueError("Expected detached_home_data.parquet to have a MultiIndex (Property_ID, Timestamp).")

    # Name levels if unnamed
    if df.index.names[0] is None:
        df.index = df.index.set_names(["Property_ID", "Timestamp"])
    else:
        names = list(df.index.names)
        names[0] = "Property_ID"
        names[1] = "Timestamp"
        df.index = df.index.set_names(names)

    # Ensure Timestamp level is datetime
    ts_level = df.index.get_level_values("Timestamp")
    if not np.issubdtype(ts_level.dtype, np.datetime64):
        df = df.copy()
        df = df.reset_index()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).set_index(["Property_ID", "Timestamp"]).sort_index()

    # Filter to allowed IDs early (big speedup)
    present_ids = set(df.index.get_level_values("Property_ID").unique())
    keep_ids = sorted(list(present_ids.intersection(allowed_ids)))
    print(f"[stage] ids in 2-min file: {len(present_ids):,}", flush=True)
    print(f"[stage] ids after intersection: {len(keep_ids):,}", flush=True)

    if len(keep_ids) == 0:
        raise ValueError("No overlapping Property_IDs between detached_home_data.parquet and merged_data_final_homes.parquet")

    # Validate required columns exist
    needed = [col_hp_cum, col_im_cum, col_dhw_flag]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in 2-min parquet: {missing}")

    # Determine base dt from one ID (assumes regular 2-min)
    sample_id = keep_ids[0]
    s = df.loc[sample_id]
    diffs = s.index.to_series().diff().dropna()
    base_dt = diffs.median() if len(diffs) else pd.Timedelta("2min")
    if pd.isna(base_dt) or base_dt <= pd.Timedelta(0):
        base_dt = pd.Timedelta("2min")
    print(f"[stage] inferred base_dt = {base_dt}", flush=True)

    out_dt = pd.to_timedelta(freq)
    expected_per_bin = max(int(round(out_dt / base_dt)), 1)
    min_required = max(int(np.ceil(min_coverage_frac * expected_per_bin)), 1)
    print(f"[stage] resample={freq} expected_per_bin={expected_per_bin} min_required={min_required}", flush=True)

    def process_one_id(df_2min_id: pd.DataFrame, pid: str) -> pd.DataFrame:
        """
        df_2min_id has DatetimeIndex (Timestamp) and source columns.
        """
        t0 = time.time()
        df_2min_id = df_2min_id.sort_index()

        if verbose_id:
            print(f"    [id:{pid}] rows={len(df_2min_id):,} span={df_2min_id.index.min()} -> {df_2min_id.index.max()}", flush=True)

        # Step energies via diff()
        et_hp = df_2min_id[col_hp_cum].diff()
        et_im = df_2min_id[col_im_cum].diff()

        # Treat negative diffs as resets/bad data -> set NaN
        et_hp = et_hp.where(et_hp >= 0, np.nan)
        et_im = et_im.where(et_im >= 0, np.nan)

        # Keep the components
        q_hp_total = et_hp
        q_immersion = et_im

        # Total energy (hp + immersion): NaN iff both missing, else sum with missing treated as 0
        both_missing = q_hp_total.isna() & q_immersion.isna()
        q_total = q_hp_total.fillna(0.0) + q_immersion.fillna(0.0)
        q_total = q_total.where(~both_missing, np.nan)

        # DHW periods: can be HP + immersion
        is_dhw = df_2min_id[col_dhw_flag].notna()
        q_dhw = np.where(is_dhw, q_total, 0.0)

        # Space conditioning: ONLY HP when DHW flag is NOT available
        q_hp_sc = np.where(is_dhw, 0.0, q_hp_total)

        out_2min = pd.DataFrame(
            {
                "Q_hp_total": q_hp_total,
                "Q_immersion": q_immersion,
                "Q_dhw": q_dhw,
                "Q_hp_sc": q_hp_sc,
                "Q_total": q_total,
            },
            index=df_2min_id.index,
        )

        # Coverage per 30-min bin: based on availability of q_total
        coverage = out_2min["Q_total"].notna().resample(freq).sum()

        # Resample
        if energy_mode == "sum":
            out_30 = out_2min.resample(freq).sum(min_count=1)
        elif energy_mode == "mean":
            out_30 = out_2min.resample(freq).mean()
        else:
            raise ValueError("energy_mode must be 'sum' or 'mean'")

        # Enforce minimum coverage
        out_30 = out_30.where(coverage >= min_required)

        if verbose_id:
            print(f"    [id:{pid}] done in {time.time()-t0:.1f}s", flush=True)

        return out_30

    # Main loop
    pieces = []
    t_loop = time.time()
    n_ids = len(keep_ids)
    print(f"[stage] processing {n_ids:,} homes", flush=True)

    for i, pid in enumerate(keep_ids, start=1):
        if i == 1 or i % print_every == 0:
            print(f"[{i}/{n_ids}] starting {pid}", flush=True)

        df_id = df.loc[pid]  # DatetimeIndex
        out_pid = process_one_id(df_id, pid)

        out_pid.columns = pd.MultiIndex.from_product([[pid], out_pid.columns], names=["Property_ID", "variable"])
        pieces.append(out_pid)

        if i == 1 or i % print_every == 0 or i == n_ids:
            elapsed = (time.time() - t_loop) / 60
            print(f"[{i}/{n_ids}] finished {pid} | elapsed={elapsed:.1f} min", flush=True)

    print("[stage] concatenating outputs", flush=True)
    out_wide = pd.concat(pieces, axis=1).sort_index()

    print(f"[stage] writing parquet: {out_path}", flush=True)
    out_wide.to_parquet(out_path)

    print(f"[stage] done. total elapsed={(time.time()-t_all)/60:.1f} min", flush=True)
    return out_path


if __name__ == "__main__":
    print("[main] Starting", flush=True)
    out = export_q_streams_30min_from_multiindex(
        in_path_2min="detached_home_data.parquet",
        in_path_allowed="retrieved_weather_data/merged_data_final_homes.parquet",
        out_path="retrieved_weather_data/q_streams_30min.parquet",
        freq="30min",
        energy_mode="sum",          # diff() gives step-energy -> sum is correct
        min_coverage_frac=0.5,
        max_gap=pd.Timedelta("2h"),
        print_every=10,
        verbose_id=False,
    )
    print(f"[main] Wrote: {out}", flush=True)
