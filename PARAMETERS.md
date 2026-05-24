# Hardware_Project — Parameter Reference

Inventory of every tolerance, limit and tunable constant reachable from
`main.py` (lap 1). Organized by location:

- **A. `config.py`** — single source of truth, intended to be tuned.
- **B. Module-level constants** — defined at the top of a `.py` file,
  outside any class.
- **C. Magic numbers** — hardcoded inside function bodies; candidates
  for promotion to `config.py` if you ever need to tune them.

---

## A. `config.py`

### Radio and network

| Constant | Value | Role |
|---|---|---|
| `CF_URI` | `radio://0/125/2M/E7E7E7E710` | Crazyflie URI used by `cflib`. Change the suffix `E7E7E7E710` to match the address printed on the drone, and the channel/datarate (`125/2M`) to match the dongle channel. |
| `AI_DECK_IP` | `192.168.4.1` | IP of the AI Deck in WiFi AP mode. Default Bitcraze firmware always uses this address — change only if you reflashed the WiFi module with custom credentials. |
| `AI_DECK_PORT` | `5000` | TCP/UDP port the AI Deck listens on for image streaming. Same default for both transports. |

### Map geometry

| Constant | Value | Role |
|---|---|---|
| `CIRCLE_CENTRE` | `[-0.10, 0.0]` m | World-frame XY position of the centre of the gate ring. Offset 10 cm in -X from the Lighthouse origin to match the physical course layout. |
| `INNER_RADIUS` | `0.2` m | Inner radius of the annulus inside which all 5 gates are placed. Also the closest radial distance the FSM expects to find a gate (used by `gate_search_corners`). |
| `OUTER_RADIUS` | `1.5` m | Outer radius of the annulus. Gates outside this are unreachable / ignored. |
| `NUM_GATES` | `5` | Number of gates in lap 1. |
| `NUM_SEGMENTS` | `6` | `NUM_GATES + 1` = 1 take-off sector + 5 gate sectors. |
| `SEGMENT_ANGULAR_SIZE` | `π/6` (30°) | **Full** angular width of one sector. Sectors are spaced 60° centre-to-centre with 30° gaps in between. |
| `SECTOR_ANGLE_OFFSET` | `0.0` rad | Global angular rotation of all sectors. Set so that sector 0 (take-off) aligns with the take-off pad direction (+x). |
| `X_BOUNDS` | `(-2.0, 2.0)` m | XY map extents used for sanity checks on commanded waypoints. |
| `Y_BOUNDS` | `(-1.5, 1.5)` m | Same as X_BOUNDS for Y axis. |
| `Z_BOUNDS` | `(0.05, 2.0)` m | Vertical envelope. The visual-servo z setpoint is clamped to this. |

### Flight altitude

| Constant | Value | Role |
|---|---|---|
| `EXPLORE_ALT` | `1.2` m | Cruise altitude during lap 1 — high enough to clear gates whose centre is at ~1 m, low enough to keep gates inside the camera FOV. |

### Gate physical properties

| Constant | Value | Role |
|---|---|---|
| `GATE_OPENING_HEIGHT` | `0.4` m | Height **fixed** in the reprojection model (the optimizer never updates it). Gate width is free; height stays constant by convention. |
| `GATE_WIDTH_BOUNDS` | `(0.3, 0.5)` m | Physical bounds on the gate opening width. Used to bracket the solver. |
| `GATE_Z_OFFSET` | `0.20` m | Compensation for the uncalibrated camera vertical bias. The reprojection and visual servo both place the gate ~0.2 m too low; this constant shifts the controller's altitude target up by the same amount. Re-tune if you recalibrate the camera. |

### Camera intrinsics (HM01B0, uncalibrated)

| Constant | Value | Role |
|---|---|---|
| `CAM_WIDTH`, `CAM_HEIGHT` | `324`, `244` px | Image resolution streamed by the AI Deck. |
| `CAM_FOV_H` | `87°` | Nominal horizontal field of view from the AI Deck 1.1 datasheet. |
| `FOCAL_PX` | `~170` px (derived) | Focal length in pixels, derived from `CAM_WIDTH` and `CAM_FOV_H`. Drives all reprojection math. |
| `CAM_CX`, `CAM_CY` | image centre | Nominal principal point. Exact value unknown without calibration — vertical mismatch is compensated by `GATE_Z_OFFSET`. |
| `CAM_OFFSET_BODY` | `[0.03, 0, -0.025]` m | Position of the camera optical centre in the drone body frame (3 cm forward, 2.5 cm below the IMU). |
| `R_CAM2BODY` | 3×3 matrix | Rotation matrix from the camera frame (x=right, y=down, z=forward) to the body frame (x=forward, y=left, z=up). |

### LED detector

| Constant | Value | Role |
|---|---|---|
| `LED_BRIGHTNESS_THRESHOLD` | `200` (0–255) | Grayscale threshold to binarize LED pixels. Raise it if dim background pixels survive the mask, lower it if some LEDs get clipped. |
| `DET_MIN_AREA` | `200` px² | Minimum contour area to be considered a gate candidate. Filters out small reflections and individual LED specks. |
| `DET_BORDER_MARGIN` | `3` px | Contours touching the image border by less than this margin are rejected (partial gates would break the corner ordering and the reprojection). |
| `MASK_OPEN_KERNEL` | `3` px | Ellipse kernel for the de-speckle `MORPH_OPEN`, applied **before** the close, that removes sub-LED salt noise. |
| `MASK_CLOSE_KERNEL` | `9` px | Ellipse kernel for the `MORPH_CLOSE` that bridges the gaps between discrete LEDs so the dotted ring welds into one continuous contour. Must exceed the inter-LED gap (~7 px, up to ~15 px) but stay well below the gate opening in pixels, or it fills the hole. Raise it if the ring stays broken in the mask view; lower it if distant gates lose their hole. **Order matters**: the open runs before this close — an open *after* the close severs the thin bridges and re-fragments the ring. |

### Approach & traverse

| Constant | Value | Role |
|---|---|---|
| `APPROACH_DIST` | `0.8` m | Distance in front of the gate plane where the drone holds for alignment before traversing. |
| `TRAVERSE_OVERSHOOT` | `0.50` m | Distance past the gate plane targeted by the HLC traverse command. Overshooting keeps non-zero velocity through the opening (the HLC min-snap reaches zero velocity at its endpoint). |
| `TRAVERSE_CLEAR_DIST` | `0.20` m | Distance past the gate plane at which the gate is declared crossed. Must be smaller than `TRAVERSE_OVERSHOOT` so the drone still has velocity when crossing. |

### Reprojection refinement (per-frame update)

| Constant | Value | Role |
|---|---|---|
| `APPROACH_REFINE_ALPHA` | `0.12` | Low-pass blend factor between the current gate estimate and each new single-view reprojection. Lower = smoother but slower to converge. |
| `CONVERGENCE_WINDOW` | `8` | Window size (frames) for measuring xy standard deviation of recent gate estimates. |
| `CONVERGENCE_MIN_SAMPLES` | `6` | Minimum samples required before convergence can be declared. Avoids triggering on a near-empty window. |
| `CONVERGENCE_STD_THRESHOLD` | `0.04` m | Maximum xy std-dev over the window for the estimate to count as converged. |
| `LOST_GATE_TIMEOUT_S` | `3.0` s | Time without a valid detection during APPROACH before transitioning to FALLBACK_SCAN. |

### Outlier rejection (reprojection)

| Constant | Value | Role |
|---|---|---|
| `REFINE_REJECT_CENTER_DRIFT` | `0.30` m | Maximum xy jump between the current estimate and a new reprojection. Larger jumps are discarded as outliers. |
| `REFINE_REJECT_YAW_DRIFT` | `0.5` rad (~30°) | Same as above for the gate yaw. |
| `REFINE_REJECT_WIDTH_DRIFT` | `0.20` m | Same as above for the gate width. |

### Visual servo (centring controller in APPROACH)

| Constant | Value | Role |
|---|---|---|
| `VISUAL_SERVO_K_H` | `0.4` m | Gain on the left/right edge-length asymmetry. Output is in metres of target offset per unit of dimensionless relative imbalance. |
| `VISUAL_SERVO_K_V` | `1.0` | Pitch-compensated vertical gain. Dimensionless, applied to `pixel_err / FOCAL_PX × distance_to_gate`. |
| `VISUAL_SERVO_K_V_FALLBACK` | `0.5` | Fallback vertical gain used when the gate is very close (`p_cam[2] < 0.1`), based on top/bottom edge asymmetry. |
| `VISUAL_SERVO_MAX_OFFSET` | `0.4` m | Per-axis clamp on the visual-servo correction to prevent runaway commands when the controller misbehaves. |

### High-level commander durations

| Constant | Value | Role |
|---|---|---|
| `HLC_TAKEOFF_DURATION` | `3.0` s | Duration of the firmware takeoff maneuver. Too short = aggressive lift; too long = sluggish. |
| `HLC_LAND_DURATION` | `2.5` s | Duration of the firmware land maneuver. |
| `HLC_TRAVERSE_DURATION` | `1.5` s | Duration of the traverse `go_to` command, sized to keep velocity high through the gate plane. |

### Log paths

| Constant | Value | Role |
|---|---|---|
| `LOG_DIR` | `"logs"` | Subdirectory name for CSV traces. |
| `PLOT_DIR` | `"camera_plots"` | Subdirectory name for camera-related debug plots. |

### Lap 2 & 3 (racing)

| Constant | Value | Role |
|---|---|---|
| `GATE_POS_CSV_PATH` | `racing/gate_pos_csv/gates_info_circle_test.csv` | Path to the CSV of known gate positions used by the racing motion planner. |
| `MOTION_PLANNER_VERBOSE` | `False` | Flag enabling diagnostic prints in `MotionPlanner3D`. |
| `RACE_CONTROLLER` | `2` | Onboard stabilizer to use for the race: `1` = PID, `2` = Mellinger. Mellinger is required to exploit the velocity/acceleration feedforward. |
| `RACE_DRONE_MASS` | `0.042` kg | Vehicle mass pushed to the Mellinger controller (`ctrlMel.mass`). Tune empirically to remove the steady-state z bias during cruise. |
| `VEL_LIM` | `25.0` m/s | Velocity limit fed to the trajectory planner. |
| `ACC_LIM` | `250.0` m/s² | Acceleration limit fed to the trajectory planner. |
| `PTS_PER_METER` | `30.0` | Trajectory sampling density. |
| `MIN_T_F` | `10.0` s | Floor on total trajectory duration. |
| `TARGET_AVG_SPEED` | `0.5` m/s | Target average speed over the planned trajectory. |
| `ADD_POINT_DIST_FROM_GATE` | `0.15` m | Distance of the auxiliary waypoints inserted just before and just after each gate, used to align the trajectory tangent with the gate normal. |

---

## B. Module-level constants in sub-files

### `main.py`

| Constant | Value | Role |
|---|---|---|
| `TICK_PERIOD` | `0.05` s | Period of the main tick loop. 20 Hz is comfortably above the HLC re-issue rate and the AI Deck frame rate. |
| `MISSION_TIMEOUT_S` | `240.0` s | Hard cap on mission duration aligned with the assignment timeout. The drone lands when reached. |
| `MAX_FRAME_AGE_S` | `0.5` s | Maximum age of a camera frame before it is treated as missing. Without this, a frozen stream would silently keep the FSM's "lost gate" timer reset. |

### `planning/fsm.py` — class attributes on `GateFSM`

| Constant | Value | Role |
|---|---|---|
| `REISSUE_POS_THRESHOLD` | `0.05` m | Position drift above which the FSM re-issues an HLC `go_to`. Below this, the previous command is left running. |
| `REISSUE_YAW_THRESHOLD` | `0.05` rad | Same as above for yaw. |
| `MIN_REISSUE_INTERVAL` | `0.1` s | Minimum interval between consecutive `go_to` re-issues (~10 Hz). Throttles radio traffic and matches the AI Deck JPEG frame rate. |
| `ARRIVAL_POS_TOL` | `0.10` m | Distance below which the drone is considered to have reached its xy target. |
| `ARRIVAL_Z_TOL` | `0.08` m | Same for the z target. |
| `ARRIVAL_YAW_TOL` | `0.08` rad (~4.6°) | Same for the yaw target. |
| `ARRIVAL_VEL_TOL` | `0.10` m/s | Total speed below which the drone is considered settled at its target. Required in addition to position to avoid declaring arrival mid-flyby. |
| `NOMINAL_SPEED` | `0.5` m/s | Reference speed used to size HLC `go_to` durations from path length. |
| `YAW_RATE` | `0.6` rad/s | Reference yaw rate used to size HLC durations from yaw delta. |
| `MIN_SEG_DURATION` | `0.6` s | Floor on any HLC `go_to` duration (firmware behaves poorly with very short durations). |
| `SCAN_YAW_RATE` | `0.30` rad/s | Slower yaw rate used during YAW_SCAN so the camera frames are not motion-blurred. |
| `SCAN_MIN_DURATION` | `2.0` s | Floor on a single yaw-scan leg. |
| `MIN_GATE_PX_WIDTH` | `18` px | Minimum gate width in pixels for a detection to be used for reprojection refinement or visual servo. Smaller = unreliable corner localization. |

### `hardware/ai_deck_stream.py` — module constants

| Constant | Value | Role |
|---|---|---|
| `_CPX_HEADER_SIZE` | `4` B | Size of one CPX packet header (`<H B B>`). |
| `_IMG_HEADER_MAGIC` | `0xBC` | First byte of an image-header payload. |
| `_IMG_HEADER_SIZE` | `11` B | Size of the image header payload. |
| `_MIN_JPEG_BYTES` | `5000` | Minimum size of a candidate frame before attempting to decode. Below this, the buffer is rejected outright. |
| `_START_MAGIC` | `b'FER'` | Datagram payload sent to the AI Deck to start the UDP stream. |
| `_LOCAL_PORT` | `5001` | Local UDP port the receiver binds to. |
| `_RECV_BUF` | `1 MB` | Socket receive buffer size, set high enough to absorb bursts on slow consumers. |
| `_OVERRUN_SLACK` | `2048` B | Tolerance above the declared frame size before declaring an overrun (UDP packet loss caused us to miss the next header). |
| `_REPORT_INTERVAL_S` | `5.0` s | Period of the stream-health log line. |
| `_SOCK_TIMEOUT_S` | `1.0` s | Read timeout, short enough that a `stop` request is noticed promptly. |
| `_CONNECT_TIMEOUT_S` | `5.0` s | TCP connect timeout. |
| `_DEBUG_HEADER_DUMP_N` | `60` | Number of CPX headers dumped to the log at startup for protocol debugging. |
| `_DEBUG_SAVE_N` | `6` | Number of raw frame buffers dumped to disk at startup for offline inspection. |

---

## C. Magic numbers (hardcoded in function bodies)

### `planning/fsm.py`

| Where | Value | Role |
|---|---|---|
| `_init_yaw_scan` | `margin = 0.12` rad (~7°) | Extra margin added to the yaw-scan extremes so the sweep slightly overshoots the polygon corners. |
| `_try_detect_gate_during_scan`, `_refine_with_reprojection` | `0.3 < z < 2.5` | Sanity bounds on the optimizer's z output. Reprojection results outside this are rejected. |
| `_try_detect_gate_during_scan` | `dist_to_region < -0.2` | Slack on `cv2.pointPolygonTest`: a detection up to 20 cm outside the expected search polygon is still accepted. |
| `_select_gate_detection` | `p_cam[2] <= 0.01` | Guard for projecting a gate behind the camera; falls back to the largest detection by area. |
| `_select_gate_detection` | `best_dist <= 120.0` px | Maximum pixel distance between a detection centre and the expected reprojected centre for the detection to be accepted as "the right gate". Critical parameter — retune if you change camera resolution or focal. |
| `_refine_with_reprojection` | `0.15 < width < 0.7` | Sanity bounds on the optimizer's width output (slightly wider than `GATE_WIDTH_BOUNDS` to absorb optimizer slack). |
| `_visual_servo_offset` | `p_cam[2] > 0.1` | Cutoff distance above which the pitch-compensated vertical servo is used; below, the edge-asymmetry fallback takes over. |
| `_do_approach`, `_do_align_traverse` | `0.55 * old + 0.45 * new` | Smoothing factor for the persistent xy/z visual-servo setpoint. Heavily damped to suppress per-frame noise. |
| `_do_approach`, `_do_align_traverse` | `pos_tol = 0.15` | Looser positional tolerance during approach/align (vs. the default `ARRIVAL_POS_TOL = 0.10`). |
| `_do_align_traverse` | `_Z_SETTLE_THRESH = 0.06` m | Maximum z servo correction below which the vertical alignment is counted as settled. |
| `_do_align_traverse` | `_Z_SETTLE_FRAMES = 3` | Consecutive settled frames required before committing the traverse (z). |
| `_do_align_traverse` | `_XY_SETTLE_THRESH = 0.08` m | Same as `_Z_SETTLE_THRESH` for the lateral correction. |
| `_do_align_traverse` | `_XY_SETTLE_FRAMES = 3` | Same as `_Z_SETTLE_FRAMES` for xy. |
| `_do_align_traverse` | `lateral < 0.3`, `z_err < 0.3` | Cross-product tolerances on the gate-crossing check. The drone must stay within this lateral/vertical envelope of the gate centre when passing the clear plane. |
| `_do_land` | `< 0.2` m | If the drone is already within 20 cm of the take-off xy, the landing skips the look-yaw alignment to avoid jittery commands. |
| `_do_land` | `+ 0.5` s | Safety margin added to `HLC_LAND_DURATION` before the LAND state declares completion. |

### `perception/gate_detector.py`

| Where | Value | Role |
|---|---|---|
| `build_mask` | open-then-close, `iterations=1` each | De-speckle (open) then bridge LED gaps (close). Kernel sizes are now tunable in `config.py` (`MASK_OPEN_KERNEL` / `MASK_CLOSE_KERNEL`); both kernels are `MORPH_ELLIPSE`. |
| `detect_all_gate_contours` | `0.02 * peri` | Polygon approximation tolerance (fraction of perimeter). Lower = more vertices preserved; higher = more aggressive simplification. |
| `detect_all_gate_contours` | `aspect < 0.4 or > 2.5` | Aspect-ratio bounds on the 4-corner approximation. Filters extreme rectangles that cannot be gates. |
| `detect_all_gate_contours` | `1e-3` | Division-by-zero guard on the edge lengths used for the aspect check. |

### `perception/gate_localizer.py`

| Where | Value | Role |
|---|---|---|
| `localize_gate_reprojection` | `bounds_lo = [-3, -3, 0.3, ...]`, `bounds_hi = [3, 3, 2.5, ...]` | Solver bounds for `[X, Y, Z, yaw, width]`. **Inconsistent** with `X_BOUNDS`, `Y_BOUNDS`, `Z_BOUNDS` in `config.py` — change here too if you re-tune the arena. |
| `localize_gate_reprojection` | `± 0.05` m | Slack added to `GATE_WIDTH_BOUNDS` so the solver has room outside the strict physical bounds. |
| `localize_gate_reprojection` | `max_nfev = 200` | Maximum number of solver function evaluations. Caps optimization time per frame. |
| `_make_reprojection_fn` | `p_cam[2] <= 0.01` | Behind-camera guard; the residual is pushed to a large value to discourage that branch. |
| `_make_reprojection_fn` | `100.0` | Penalty value substituted for behind-camera residuals. |

### `hardware/crazyflie_link.py`

| Where | Value | Role |
|---|---|---|
| `_start_logging` | `period_in_ms = 10` | Lighthouse pose / velocity log period (100 Hz). |
| `__exit__` | `airborne = z > 0.10` m | Emergency-land trigger: only attempts a firmware land if the drone is more than 10 cm off the ground. |
| `__exit__` | `+ 0.8` s | Margin waited after issuing the emergency `land` command before closing the radio link. |
| `_wait_for_estimator_convergence` | `timeout = 10.0` s | Total time to wait for the Kalman variance to settle. Raises if exceeded. |
| `_wait_for_estimator_convergence` | `threshold = 0.001` | Max-minus-min of the variance window below which the estimator is declared converged. |
| `_arm` | `retries = 6`, `delay = 0.5` s | Number of arming attempts and the delay between them. |
| `notify_setpoint_stop` | `remain_valid_ms = 200` | How long the last streamed low-level setpoint remains valid after the racing run ends, bridging the gap until the HLC takes over. |

### `planning/geometry.py`

| Where | Value | Role |
|---|---|---|
| `compute_traversal_direction` | `1e-12` | Division-by-zero guard when normalizing the radial unit vector. |
