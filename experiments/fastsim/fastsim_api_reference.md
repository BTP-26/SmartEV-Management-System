# FASTSim API Reference (for Track U, U-1 → U-4)

Captured during U-0 so we don't rediscover it. **Installed version: `fastsim` 3.0.6** (fastsim-3,
Rust-backed; results come back as **polars** DataFrames, not pandas).

## Install

```bash
.venv/bin/python -m pip install fastsim   # resolves to 3.0.6
```

Pulls polars, ipykernel/jupyter, and **downgrades numpy 2.4.6→2.3.2 + PyYAML 6.0.3→6.0.2**.
→ For U-1+, install FASTSim in its **own venv** to avoid perturbing the main pipeline's deps
(see U0_FINDINGS.md, Issues).

## Load a vehicle

```python
import fastsim as fsim
fsim.Vehicle.list_resources()
# ['2012_Ford_Fusion.yaml', '2016 Nissan Leaf 30 kWh thrml.yaml', '2016_TOYOTA_Prius_Two.yaml',
#  '2020 Chevrolet Bolt EV thrml.yaml', '2021_Hyundai_Sonata_Hybrid_Blue_thrml.yaml',
#  '2022 Tesla Model 3 RWD thrml.yaml', '2022_Renault_Zoe_ZE50_R135.yaml']
veh = fsim.Vehicle.from_resource("2022 Tesla Model 3 RWD thrml.yaml")
```

- **BEVs available:** Nissan Leaf 30 kWh, Chevy Bolt EV, Tesla Model 3 RWD, Renault Zoe ZE50.
- Custom vehicle (U-1): `Vehicle.from_file / from_yaml / from_pydict`. Edit the pack via
  `veh.to_pydict()` → `pt_type.BEV.res.energy_capacity_joules` (kWh = joules / 3.6e6),
  then `Vehicle.from_pydict(d)`.

## Load / build a cycle

```python
cyc = fsim.Cycle.from_resource("udds.csv")     # or "hwfet.csv"
# custom (U-2 will build from UAH speed): copy the udds pydict template, replace arrays
d = fsim.Cycle.from_resource("udds.csv").to_pydict()
# keys: init_elev_meters, time_seconds, speed_meters_per_second, dist_meters, grade,
#       elev_meters, pwr_max_chrg_watts, temp_amb_air_kelvin, pwr_solar_load_watts,
#       grade_interp, elev_interp
cyc = fsim.Cycle.from_pydict(d)
```

- U-2 minimum: uniform 1 Hz `time_seconds` + `speed_meters_per_second`; `grade=0` (flat);
  `temp_amb_air_kelvin≈295.15` (thermal vehicles need it); compute `dist_meters` = cumsum(v·dt).

## Run

```python
sd = fsim.SimDrive(veh, cyc)
sd.walk()                    # NB: fastsim-3 uses .walk(), not .sim_drive()
df = sd.to_dataframe()       # polars DataFrame, ~138 columns
```

polars gotchas: `df[col].to_numpy()` (no `dtype=` kwarg — wrap with `np.asarray(..., dtype=float)`);
column membership via `col in df.columns`.

## Key result columns (BEV)

| Quantity | Column |
|---|---|
| SoC | `veh.pt_type.BEV.res.history.soc` |
| Battery power, electrical (+discharge / −regen) | `veh.pt_type.BEV.res.history.pwr_out_electrical_watts` |
| Battery power, chemical (internal) | `veh.pt_type.BEV.res.history.pwr_out_chemical_watts` |
| Max regen power limit | `veh.pt_type.BEV.res.history.pwr_regen_max_watts` |
| Battery temperature (K) | `veh.pt_type.BEV.res.thrml.RESLumpedThermal.history.temperature_kelvin` |
| Achieved speed (m/s) | `veh.history.speed_ach_meters_per_second` |
| Time (s) | `veh.history.time_seconds` |
| Distance (m) | `veh.history.dist_meters` |
| Tractive power (W) | `veh.history.pwr_tractive_watts` |

## Derived quantities (not native to FASTSim-3)

- **Regen energy (Wh)** = −Σ min(pwr_out_electrical, 0)·Δt / 3600.
- **Battery current (A)** = pwr_out_electrical / assumed_voltage — **FASTSim-3 has no native
  current or pack voltage** (RES is an energy/power reservoir). Use the Mendeley nominal (424 V)
  or a chosen value; state it as an assumption.
- **Energy used (kWh)** = (soc_start − soc_end) × capacity_kwh.

## Notes / gotchas
- `fsim.resources_root()` is a **function**; the returned path may not exist on disk — use
  `from_resource(...)`, not filesystem paths.
- BEV paths contain `pt_type.BEV`; HEV/PHEV/Conventional vehicles use different `pt_type` keys.
- "thrml" vehicles carry a thermal model → the cycle must provide `temp_amb_air_kelvin`.
