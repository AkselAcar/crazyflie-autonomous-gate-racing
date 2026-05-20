# Coordinate Frames: CSV vs Drone

This document describes the two coordinate systems involved in the racing
pipeline and the transformation applied to gate positions read from the
race CSV.

## The two frames

### Drone frame (`D`)

The frame used by the Crazyflie's onboard estimator. With the **Lighthouse**
positioning system, this is an absolute frame anchored to the physical
Lighthouse base-station calibration — *not* relative to the drone's boot
position.

- The drone is placed at position `(1, 0, 0)` in `D` at takeoff.
- Yaw `0` in `D` corresponds to the drone's `+x_D` heading.
- All firmware-level setpoints (position, yaw) sent over CRTP are
  interpreted in `D`.

### CSV frame (`E`)

The frame in which gate positions are expressed in the race CSV
(`gate_pos_csv/*.csv`, columns `x, y, z, theta, width, height`).

It is related to `D` by:

| Aspect       | Definition                                          |
| ------------ | --------------------------------------------------- |
| Origin       | `O_E` is at `(1, 0, 0)` in `D` coordinates.         |
| Orientation  | `E` is rotated 180° around `+z` w.r.t. `D`.         |
| Yaw `theta`  | Measured **from the `−x_E` axis** (not `+x_E`), CCW around `+z`. |

Equivalently:
- `+x_E` direction in `D` = `−x_D` direction
- `+y_E` direction in `D` = `−y_D` direction
- `+z_E` direction in `D` = `+z_D` direction

## Position transformation (CSV → Drone)

For a point `p_E = (x_E, y_E, z_E)` in the CSV frame, its coordinates in
the drone frame are:

```
p_D = O_E_in_D + R_z(180°) · p_E
    = (1, 0, 0) + (−x_E, −y_E, z_E)
    = (1 − x_E, −y_E, z_E)
```

Component-wise:

```
x_D = 1 − x_E
y_D = −y_E
z_D = z_E
```

## Yaw transformation (CSV → Drone)

The CSV's `theta` measures the gate surface angle from the `−x_E` axis,
CCW around `+z`. Since:

- `−x_E` direction in physical space = `+x_D` direction (180° rotation
  flips the `x` axis sense),
- the `+z` axis is unchanged by the rotation, so CCW in `E` = CCW in `D`,

the angle value transfers as-is:

```
theta_D = theta_E
```

No numerical conversion of yaw is needed — only positions are
transformed.

### Why this elegant cancellation happens

Two effects compose to give the identity on yaw:

1. The 180° rotation between `E` and `D` would by itself add `π` to any
   angle expressed from `+x`.
2. The CSV's choice to measure yaw from `−x_E` instead of `+x_E`
   subtracts `π`.

The two cancel, so `theta_E` can be used directly as a yaw in `D`.

## Where the transformation is applied in code

A single place: `motion_planner.py::construct_waypoints`. Right after
`pd.read_csv`, the `x` and `y` columns of the gate DataFrame are
transformed before any downstream use:

```python
gate_pos_df["x"] = 1.0 - gate_pos_df["x"]
gate_pos_df["y"] = -gate_pos_df["y"]
# z and theta unchanged
```

Everything downstream (waypoints, approach/exit points, plotting, the
racing FSM, the firmware setpoints) reasons in the drone frame `D` only.

## Physical setup: a calibration constraint, not a takeoff constraint

The motion planner adapts to whatever pose the drone is in after
takeoff: in `race_fsm.py`, the current estimator pose `drone_pos` is
passed to `MotionPlanner3D` as the start waypoint, and the trajectory
returns to that same `xy` at the end. So the takeoff location itself
is **not** constrained — the drone can lift off from anywhere inside
the Lighthouse-calibrated volume.

What **is** required is that the two frames `D` and `E` line up
correctly in physical space. The race CSV is authored in `E`, which is
defined by the race organizers relative to a fixed physical reference
in the arena (typically a marked point on the ground). The drone reads
positions in `D`, defined by your Lighthouse calibration. For the
transformation `p_D = (1 − x_E, −y_E, z_E)` to actually put the gates
in the right physical place, you must calibrate the Lighthouse such
that:

- The **physical point taken as the CSV origin `O_E`** has coordinates
  `(1, 0, 0)` in `D`.
- The **physical `+x_E` direction** points along `−x_D` (i.e., the
  Lighthouse frame's `x` axis is anti-parallel to the CSV frame's `x`
  axis).

The `z` axes are both vertical and need only point upward.

### Practical verification

By race convention, the takeoff pad coincides with the CSV origin. So
the easiest way to verify your calibration is:

1. Place the drone exactly on the marked takeoff pad, oriented along
   the conventional heading.
2. Check on the cfclient that the estimator reads `(1, 0, 0)` with
   yaw `0` in `D`.

If both readings match, your Lighthouse calibration is consistent with
the CSV frame and the gates will be physically where the CSV says they
are. The drone happening to be at `(1, 0, 0)_D` at takeoff is then a
*consequence* of this convention, not a hard requirement of the
trajectory code.

## Quick reference

| Quantity            | CSV (`E`)            | Drone (`D`)                     |
| ------------------- | -------------------- | ------------------------------- |
| Drone takeoff pose  | `(0, 0, 0)`, yaw `0` | `(1, 0, 0)`, yaw `0`            |
| Gate position       | `(x, y, z)`          | `(1 − x, −y, z)`                |
| Gate yaw (`theta`)  | `theta`              | `theta` (unchanged)             |
| Yaw convention      | from `−x_E` axis     | from `+x_D` axis                |
