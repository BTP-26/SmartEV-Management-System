"""U-4 - per-driving-style energy analysis (Track U). SIMULATION.

Produces the honest, measured per-style energy table that replaces the withdrawn "up to 65%"
recovery claim / Table 5. This module is an ANALYSIS layer: it orchestrates U-1 (vehicle),
U-2 (cycle adapter) and U-3 (coupling) and aggregates their outputs. It re-implements no physics.

What it adds on top of U-2/U-3 (review items it closes):
  * C-3/R-6 - its own auditable trip enumeration over (driver x style x road type), so per-style
              comparisons are ROAD-TYPE MATCHED and span multiple drivers. It deliberately does
              NOT use u2._pick_trips(), which picks one arbitrary trip per style.
  * R-2     - GPS gap detection: long dropouts are linearly interpolated by the adapter, which
              fabricates plausible-looking speed, so affected trips are flagged and excluded.
  * C-1     - a clipping diagnostic quantifying how much `condition_speed` alters each trace,
              reported PER STYLE. Originally found a speed-independent power budget biasing the
              style contrast (AGGRESSIVE clipped far more than NORMAL); U-2 fixed this with a
              speed-dependent propulsion limit probed from FASTSim itself. The gate stays in
              place as a standing monitor - it re-verifies per run that AGGRESSIVE is not clipped
              materially more than NORMAL, rather than assuming the fix holds forever.
  * R-1     - every result row carries `oracle_upper_bound=True`: intent is an oracle and the
              controller keeps the anticipatory run only when it helps, so savings are an UPPER
              BOUND, not a deployable result.
  * R-5     - built-in UDDS/HWFET reference cycles as a dataset-independent anchor.

Runs fully WITHOUT the UAH dataset (reference cycles only); the per-style table additionally
requires UAH RAW_GPS.txt under UAH-DRIVESET-v1/.

Run:  .venv-fastsim/bin/python experiments/fastsim/u4_style_analysis.py
      .venv-fastsim/bin/python experiments/fastsim/u4_style_analysis.py --reference-only
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import u2_cycle_adapter as u2                 # U-2 (imported, never modified)
import u3_coupling as u3                      # U-3 (imported, never modified)
from vehicle_config import build_mendeley_bev, max_fwd_propulsion_power_w   # U-1
from sim_config import DEFAULT as CFG

DATA_DIR = HERE.parent.parent / "UAH-DRIVESET-v1"
FIG_DIR = HERE / "u4_figures"
TS_DIR = HERE / "u4_timeseries"              # .npz per run (regenerable; gitignored)

STYLES = ("NORMAL", "AGGRESSIVE", "DROWSY")
ROADS = ("MOTORWAY", "SECONDARY")


# --------------------------- enumeration (C-3 / R-6) ---------------------------
def parse_trip_name(name):
    """UAH folder name -> metadata, or None if it doesn't match the expected layout.

    Format: <timestamp>-<distance>-<driver>-<behaviour>-<road>
    e.g. 20151110175712-16km-D1-NORMAL1-SECONDARY
    Behaviour carries a suffix on repeats (NORMAL1/NORMAL2) - normalized to the base style.
    """
    parts = name.split("-")
    if len(parts) < 5:
        return None
    raw_style, road = parts[3].upper(), parts[4].upper()
    style = next((s for s in STYLES if s in raw_style), None)
    if style is None or road not in ROADS:
        return None
    return {"trip": name, "driver": parts[2].upper(), "style": style,
            "road": road, "label_distance": parts[1]}


def enumerate_trips(data_dir=DATA_DIR):
    """Every parseable UAH trip as a metadata record. Returns [] (never raises) when the
    dataset is absent, so the reference-only path still runs (R-6)."""
    if not Path(data_dir).is_dir():
        return []
    out = []
    for driver in sorted(os.listdir(data_dir)):
        dp = Path(data_dir) / driver
        if not dp.is_dir() or not driver.upper().startswith("D"):
            continue
        for trip in sorted(os.listdir(dp)):
            if not (dp / trip).is_dir():
                continue
            rec = parse_trip_name(trip)
            if rec is not None:
                rec["path"] = str(dp / trip)
                out.append(rec)
    return out


# --------------------------- diagnostics ---------------------------
def gps_gap_stats(trip_path, max_gap_s=CFG.max_gps_gap_s):
    """Raw-GPS sampling gaps (R-2). Long gaps get linearly interpolated by the adapter, which
    fabricates speed - so we measure them before that happens."""
    t, _ = u2.read_uah_gps(trip_path)
    if t is None or len(t) < 2:
        return {"n_fixes": 0, "max_gap_s": None, "gap_ok": False}
    ts = np.unique(np.sort(np.asarray(t, float)))
    gaps = np.diff(ts)
    return {
        "n_fixes": int(len(ts)),
        "median_dt_s": round(float(np.median(gaps)), 3),
        "max_gap_s": round(float(gaps.max()), 2),
        "n_gaps_over_threshold": int((gaps > max_gap_s).sum()),
        "gap_ok": bool(gaps.max() <= max_gap_s),
    }


def clipping_stats(time_s, speed_mps, cfg=CFG):
    """How much does `condition_speed` alter this trace? (C-1 gate.)

    The conditioner caps acceleration using a speed-INDEPENDENT power budget, so it can clip
    genuine high-speed accelerations. Quantified per trip, then aggregated per style: a
    style-correlated clip rate means the per-style contrast is biased.
    """
    t = np.asarray(time_s, float)
    v = np.asarray(speed_mps, float)
    _, vc = u2.condition_speed(t, v)
    diff = np.abs(v - vc)
    clipped = diff > 1e-6
    d_raw = float(np.trapezoid(v, t))
    d_con = float(np.trapezoid(vc, t))
    return {
        "clip_rate": round(float(clipped.mean()), 5),
        "n_clipped": int(clipped.sum()),
        "mean_clip_err_mps": round(float(diff[clipped].mean()), 4) if clipped.any() else 0.0,
        "max_clip_err_mps": round(float(diff.max()), 4),
        "distance_delta_pct": round(100.0 * (d_con - d_raw) / d_raw, 4) if d_raw > 0 else 0.0,
    }


# --------------------------- per-run execution ---------------------------
def _row(meta, ablation, clip, cfg=CFG):
    """Flatten one baseline-vs-anticipatory ablation into a tidy result row."""
    b, a = ablation["baseline"], ablation["anticipatory"]
    row = dict(meta)
    row.update({
        "distance_km": b["distance_km"],
        "wh_per_km_baseline": b["wh_per_km"],
        "wh_per_km_anticipatory": a["wh_per_km"],
        "saved_wh_per_km": ablation["energy_saved_wh_per_km"],
        "saved_pct": ablation["energy_saved_pct"],
        "regen_wh_baseline": b["regen_energy_wh"],
        "regen_wh_anticipatory": a["regen_energy_wh"],
        "soc_start": b["soc_start"], "soc_end_baseline": b["soc_end"],
        "soc_end_anticipatory": a["soc_end"],
        "intervened": ablation["intervened"],
        "any_nan": bool(b["meta"]["any_nan"] or a["meta"]["any_nan"]),
        # honesty flags (R-1) - must survive into the paper
        "oracle_upper_bound": cfg.oracle_upper_bound,
        "simulation": cfg.simulation,
        "prop_budget": cfg.prop_budget,
        "accel_cap_mps2": cfg.accel_cap_mps2,
        "brake_trigger_w": cfg.brake_trigger_w,
    })
    row.update(clip)
    return row


def _save_timeseries(name, ablation):
    TS_DIR.mkdir(exist_ok=True)
    b, a = ablation["baseline"], ablation["anticipatory"]
    np.savez_compressed(
        TS_DIR / f"{name}.npz",
        time_s=np.asarray(b["time_s"], float),
        soc_baseline=np.asarray(b["soc"], float),
        soc_anticipatory=np.asarray(a["soc"], float),
        power_baseline_w=np.asarray(b["batt_power_w"], float),
        power_anticipatory_w=np.asarray(a["batt_power_w"], float),
        current_baseline_a=np.asarray(b["batt_current_a"], float),
    )


def run_reference_cycles(vehicle, cfg=CFG):
    """UDDS/HWFET anchor (R-5) - dataset-independent, validates the whole path."""
    rows = []
    for res, label in (("udds.csv", "UDDS"), ("hwfet.csv", "HWFET")):
        import fastsim as fsim
        d = fsim.Cycle.from_resource(res).to_pydict()
        t = np.asarray(d["time_seconds"], float)
        v = np.asarray(d["speed_meters_per_second"], float)
        abl = u3.couple_ablation(t, v, vehicle=vehicle)
        meta = {"source": "reference", "trip": label, "driver": "-",
                "style": "REFERENCE", "road": label}
        rows.append(_row(meta, abl, clipping_stats(t, v, cfg), cfg))
        _save_timeseries(f"reference_{label}", abl)
        print(f"  [ref] {label:6} {rows[-1]['wh_per_km_baseline']:7.2f} -> "
              f"{rows[-1]['wh_per_km_anticipatory']:7.2f} Wh/km "
              f"(saved {rows[-1]['saved_pct']}%) clip={rows[-1]['clip_rate']*100:.2f}%")
    return rows


def run_uah_trips(records, vehicle, cfg=CFG):
    """Sweep every enumerated UAH trip. Returns (rows, manifest)."""
    rows, manifest = [], []
    for rec in records:
        entry = dict(rec)
        gaps = gps_gap_stats(rec["path"], cfg.max_gps_gap_s)
        entry["gps"] = gaps
        arr = u2.cycle_arrays(rec["path"])
        if arr is None:
            entry.update(included=False, reason="unreadable/too-short GPS")
            manifest.append(entry); continue
        t, v = arr
        checks = u2.validate_arrays(t, v)
        duration, dist = checks["duration_s"], checks["distance_km"]
        if not checks["valid"]:
            entry.update(included=False, reason="failed structural validation")
        elif not gaps["gap_ok"]:
            entry.update(included=False, reason=f"GPS gap {gaps['max_gap_s']}s > {cfg.max_gps_gap_s}s")
        elif duration < cfg.min_duration_s or dist < cfg.min_distance_km:
            entry.update(included=False, reason="too short")
        else:
            entry.update(included=True, reason="ok")
        entry["duration_s"], entry["distance_km"] = duration, dist
        clip = clipping_stats(t, v, cfg)
        entry["clipping"] = clip
        manifest.append(entry)
        if not entry["included"]:
            print(f"  [skip] {rec['trip']}: {entry['reason']}")
            continue
        # Some real trips brake harder than the electric machine can absorb (e.g. -253 kW demanded
        # vs a 239 kW machine). A real vehicle blends friction braking for the excess; FASTSim-3's
        # BEV model raises instead. Exclude with an explicit reason rather than distorting the
        # trace - and report the per-style breakdown so any sampling bias is visible.
        try:
            abl = u3.couple_ablation(t, v, vehicle=vehicle)
        except Exception as exc:
            detail = next((l.strip() for l in str(exc).splitlines() if "pwr_out_req" in l),
                          str(exc).splitlines()[-1][:80])
            entry.update(included=False, reason=f"powertrain limit exceeded ({detail})")
            print(f"  [skip] {rec['trip']}: powertrain limit — {detail}")
            continue
        meta = {"source": "uah", "trip": rec["trip"], "driver": rec["driver"],
                "style": rec["style"], "road": rec["road"]}
        rows.append(_row(meta, abl, clip, cfg))
        _save_timeseries(f"{rec['driver']}_{rec['style']}_{rec['road']}_{rec['trip'][:14]}", abl)
        print(f"  {rec['driver']:3} {rec['style']:10} {rec['road']:9} "
              f"{rows[-1]['wh_per_km_baseline']:7.2f} -> {rows[-1]['wh_per_km_anticipatory']:7.2f} Wh/km "
              f"(saved {rows[-1]['saved_pct']:5.2f}%) clip={clip['clip_rate']*100:.2f}%")
    return rows, manifest


# --------------------------- aggregation ---------------------------
def _stats(vals):
    a = np.asarray(vals, float)
    return (round(float(a.mean()), 2), round(float(a.std(ddof=1)), 2) if len(a) > 1 else 0.0, len(a))


def aggregate_by_style(rows):
    """Per (style, road) aggregation - ROAD-TYPE MATCHED (C-3). Pooling road types would
    attribute a road-type difference to driving style, which is the confound U-4 exists to avoid."""
    out = []
    uah = [r for r in rows if r["source"] == "uah"]
    for style in STYLES:
        for road in ROADS:
            sel = [r for r in uah if r["style"] == style and r["road"] == road]
            if not sel:
                continue
            m_b, s_b, n = _stats([r["wh_per_km_baseline"] for r in sel])
            m_a, s_a, _ = _stats([r["wh_per_km_anticipatory"] for r in sel])
            m_s, s_s, _ = _stats([r["saved_pct"] for r in sel])
            m_r, s_r, _ = _stats([r["regen_wh_baseline"] for r in sel])
            m_c, s_c, _ = _stats([r["clip_rate"] * 100 for r in sel])
            out.append({
                "style": style, "road": road, "n_trips": n,
                "n_drivers": len({r["driver"] for r in sel}),
                "wh_per_km_baseline_mean": m_b, "wh_per_km_baseline_std": s_b,
                "wh_per_km_anticipatory_mean": m_a, "wh_per_km_anticipatory_std": s_a,
                "saved_pct_mean": m_s, "saved_pct_std": s_s,
                "regen_wh_mean": m_r, "regen_wh_std": s_r,
                "clip_rate_pct_mean": m_c, "clip_rate_pct_std": s_c,
                "oracle_upper_bound": True, "simulation": True,
            })
    return out


def clipping_gate(style_table):
    """C-1 gate: is the conditioner's clipping style-correlated? If AGGRESSIVE is clipped much
    more than NORMAL on the SAME road type, the speed-independent power budget is compressing the
    very contrast the table reports, and must be made speed-dependent before publishing."""
    verdict = {"gate": "clipping_style_correlation", "checked": [], "passed": True}
    for road in ROADS:
        by = {r["style"]: r["clip_rate_pct_mean"] for r in style_table if r["road"] == road}
        if "AGGRESSIVE" in by and "NORMAL" in by:
            agg, nor = by["AGGRESSIVE"], by["NORMAL"]
            ok = (agg - nor) < 5.0          # >5 percentage-point excess = material bias
            verdict["checked"].append({"road": road, "aggressive_clip_pct": agg,
                                       "normal_clip_pct": nor, "excess_pp": round(agg - nor, 3),
                                       "passed": ok})
            verdict["passed"] = verdict["passed"] and ok
    if not verdict["checked"]:
        verdict.update(passed=None, note="insufficient data (needs UAH trips for both styles)")
    return verdict


# --------------------------- outputs ---------------------------
def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        # lineterminator="\n": the csv module's excel dialect defaults to CRLF, which git
        # normalizes to LF on commit - so a regenerated file would otherwise differ from the
        # committed one and show as a spurious diff. LF keeps re-runs byte-identical.
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_figures(rows, style_table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    if style_table:
        for road in ROADS:
            sel = [r for r in style_table if r["road"] == road]
            if not sel:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(sel)); w = 0.38
            ax.bar(x - w/2, [s["wh_per_km_baseline_mean"] for s in sel], w,
                   yerr=[s["wh_per_km_baseline_std"] for s in sel], capsize=4, label="baseline")
            ax.bar(x + w/2, [s["wh_per_km_anticipatory_mean"] for s in sel], w,
                   yerr=[s["wh_per_km_anticipatory_std"] for s in sel], capsize=4, label="anticipatory")
            ax.set_xticks(x, [f'{s["style"]}\n(n={s["n_trips"]})' for s in sel])
            ax.set_ylabel("energy (Wh/km)")
            ax.set_title(f"Per-style energy — {road} (SIMULATION, oracle upper bound)")
            ax.legend(); ax.grid(alpha=0.3, axis="y")
            fig.tight_layout(); fig.savefig(FIG_DIR / f"u4_energy_{road.lower()}.png", dpi=110)
            plt.close(fig)
    # reference-cycle anchor figure (always available)
    ref = [r for r in rows if r["source"] == "reference"]
    if ref:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(ref)); w = 0.38
        ax.bar(x - w/2, [r["wh_per_km_baseline"] for r in ref], w, label="baseline")
        ax.bar(x + w/2, [r["wh_per_km_anticipatory"] for r in ref], w, label="anticipatory")
        ax.set_xticks(x, [r["trip"] for r in ref])
        ax.set_ylabel("energy (Wh/km)")
        ax.set_title("Reference cycles (SIMULATION, oracle upper bound)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(FIG_DIR / "u4_reference_cycles.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="U-4 per-style energy analysis (SIMULATION)")
    ap.add_argument("--reference-only", action="store_true",
                    help="skip UAH sweep even if the dataset is present")
    args = ap.parse_args()

    print(f"U-4 per-style energy analysis (SIMULATION) | config: "
          f"prop_budget={CFG.prop_budget} accel_cap={CFG.accel_cap_mps2}m/s^2 "
          f"trigger={CFG.brake_trigger_w:.0f}W lead={CFG.lead_s}s")
    vehicle, veh_meta = build_mendeley_bev()
    print(f"  vehicle: {veh_meta['modified']['battery_capacity_kwh']['new']} kWh, "
          f"max fwd propulsion {max_fwd_propulsion_power_w()/1000:.0f} kW")

    rows = run_reference_cycles(vehicle, CFG)

    records = [] if args.reference_only else enumerate_trips()
    manifest = []
    if records:
        print(f"  enumerated {len(records)} UAH trips "
              f"({len({r['driver'] for r in records})} drivers)")
        uah_rows, manifest = run_uah_trips(records, vehicle, CFG)
        rows += uah_rows
    else:
        print(f"  no UAH trips found at {DATA_DIR} — reference-only mode "
              f"(per-style table pending dataset)")

    style_table = aggregate_by_style(rows)
    gate = clipping_gate(style_table)

    write_csv(HERE / "u4_results.csv", rows)
    write_csv(HERE / "u4_style_table.csv", style_table)
    with open(HERE / "u4_trip_manifest.json", "w") as f:
        json.dump({"note": "SIMULATION; UAH trip enumeration with include/exclude reasons",
                   "config": CFG.to_dict(),
                   "n_enumerated": len(records), "n_included": sum(1 for m in manifest if m.get("included")),
                   "clipping_gate": gate, "trips": manifest}, f, indent=2)
    make_figures(rows, style_table)

    print(f"\n  clipping gate: passed={gate['passed']} {gate.get('note','')}")
    for s in style_table:
        print(f"  {s['style']:10} {s['road']:9} n={s['n_trips']} "
              f"{s['wh_per_km_baseline_mean']}±{s['wh_per_km_baseline_std']} Wh/km "
              f"saved {s['saved_pct_mean']}%")
    print(f"\n  wrote u4_results.csv ({len(rows)} rows), u4_style_table.csv "
          f"({len(style_table)} groups), u4_trip_manifest.json, u4_figures/")


if __name__ == "__main__":
    main()
