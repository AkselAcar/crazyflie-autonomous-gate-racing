"""
Generate randomized gate positions and save to CSV file.

Layout follows the MICRO-502 course directives:
  - The annular arena is divided into 12 segments of 30° each (full 360°).
  - 5 non-consecutive (non-adjacent) segments are selected randomly.
  - One gate is placed per selected segment within the annular ring.

Differences from the official reference map:
  - Origin is offset 10 cm in -X  →  CIRCLE_CENTRE = [-0.10, 0.0]
  - Take-off pad is on the opposite (negative-X) side of the map.
"""

import sys
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

STARTING_POINT = config.STARTING_POINT

# ============================================================================
# GLOBAL BOUNDS AND PARAMETERS
# ============================================================================

# Course geometry (circle centre offset -10 cm in X from true origin)
CIRCLE_CENTRE = np.array([-0.10, 0.0])
INNER_RADIUS  = 0.5   # m
OUTER_RADIUS  = 1.2   # m

# Gate dimensions
GATE_WIDTH_BOUNDS  = [0.3, 0.5]   # m
GATE_HEIGHT_BOUNDS = [0.5, 0.5]   # m  (fixed height)
GATE_Z_BOUNDS      = [0.5, 1.5]   # m  (vertical centre of gate)

# Segment layout -- 12 segments x 30deg = 360deg
NUM_GATES            = 5
NUM_TOTAL_SEGMENTS   = 12
SEGMENT_ANGULAR_SIZE = 2 * np.pi / NUM_TOTAL_SEGMENTS   # pi/6 = 30deg

# Random perturbation of the gate surface angle within a segment
GATE_ORIENTATION_BOUNDS = [-np.pi / 6, np.pi / 6]      # +/-30deg


# ============================================================================
# HELPER: non-consecutive segment selection
# ============================================================================

def pick_non_consecutive_segments(n_total, n_pick):
    """
    Randomly pick n_pick non-adjacent segment indices from n_total segments
    arranged in a circle (segment 0 and segment n_total-1 are also adjacent).

    For n_total=12, n_pick=5 there are C(7,5)=21 valid combinations.
    Returns a sorted list of segment indices.
    """
    valid = []
    for combo in combinations(range(n_total), n_pick):
        ok = True
        for i in range(n_pick):
            # Circular gap between this chosen segment and the next chosen one
            gap = (combo[(i + 1) % n_pick] - combo[i]) % n_total
            if gap < 2:       # adjacent -> invalid
                ok = False
                break
        if ok:
            valid.append(combo)
    # valid is guaranteed non-empty for n_total=12, n_pick=5
    return sorted(random.choice(valid))


# ============================================================================
# GATE CSV GENERATION
# ============================================================================

def generate_gate_csv(
    output_filename="gates_info.csv",
    num_gates=NUM_GATES,
    seed=None,
    verbose=True,
):
    """
    Generate randomized gate positions and save to CSV.

    Gate placement
    --------------
    - 12 segments of 30deg cover the full circle.
    - 5 non-consecutive segments are chosen at random.
    - One gate per segment: angular position drawn uniformly from the
      segment's 30deg arc; radius drawn uniformly from [INNER_RADIUS, OUTER_RADIUS].

    CSV column 'theta'
    ------------------
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    selected_segments = [1, 3, 5, 7, 9] # pick_non_consecutive_segments(NUM_TOTAL_SEGMENTS, num_gates)

    if verbose:
        print(f"Selected segments (0-indexed, seg 0 = [0deg,30deg]): {selected_segments}\n")

    gates_data = []

    for gate_num, seg_idx in enumerate(selected_segments, start=1):
        seg_start = seg_idx       * SEGMENT_ANGULAR_SIZE - np.pi/12 + np.pi
        seg_end   = (seg_idx + 1) * SEGMENT_ANGULAR_SIZE - np.pi/12 + np.pi

        # Random angular position within the segment
        alpha  = random.uniform(seg_start, seg_end)
        radius = random.uniform(INNER_RADIUS, OUTER_RADIUS)

        # Cartesian position
        # Sign convention: negate cos/sin so increasing alpha sweeps CW on map
        x = CIRCLE_CENTRE[0] - radius * np.cos(alpha)
        y = CIRCLE_CENTRE[1] - radius * np.sin(alpha)

        goal_height = np.random.uniform(*GATE_HEIGHT_BOUNDS)
        goal_width  = np.random.uniform(*GATE_WIDTH_BOUNDS)
        goal_z      = np.random.uniform(*GATE_Z_BOUNDS)

        # Surface angle: angle of gate-plane projection w.r.t. the X-axis.
        # For a gate tangent to the circle the natural surface direction is alpha;
        # the random offset perturbs it within +/-30deg.
        orientation_offset = np.random.uniform(*GATE_ORIENTATION_BOUNDS )
        theta = alpha + orientation_offset   # surface angle w.r.t. X-axis (rad)

        gates_data.append({
            "Gate":    gate_num,
            "x":       round(x,          4),
            "y":       round(y,          4),
            "z":       round(goal_z,     4),
            "theta":   (theta + np.pi) % (2*np.pi) - np.pi,   # surface-XY angle (rad); CCW approach normal = theta - pi/2
            "width":   round(goal_width, 4),
            "height":  round(goal_height,4)
        })

        if verbose:
            print(
                f"  Gate {gate_num}  seg={seg_idx:2d}"
                f"  [{np.degrees(seg_start):.0f}deg-{np.degrees(seg_end):.0f}deg]"
                f"  pos=({x:+.3f}, {y:+.3f}, {goal_z:.3f})"
                f"  theta_surface={np.degrees(theta):.1f}deg"
                f"  theta_normal={np.degrees(theta - np.pi/2):.1f}deg"
                f"  w={goal_width:.3f}  h={goal_height:.3f}"
            )

    df = pd.DataFrame(gates_data)
    df.to_csv(output_filename, index=False)

    if verbose:
        print(f"\nGate CSV saved to: {output_filename}")

    return df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_gates(gates_df, show_plot=True):
    """
    Plot gate positions and their approach normals on a 2D top-down map.

    Arrows show the gate NORMAL in the CCW approach direction
    (= theta - pi/2, where theta is the surface angle stored in the CSV).

    All 12 segment boundaries are drawn as dotted radial lines clipped
    strictly between the inner and outer circles.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # -- Arena ring boundaries ---------------------------------------------
    for r, color, label in [
        (INNER_RADIUS, 'royalblue', 'Inner boundary'),
        (OUTER_RADIUS, 'crimson',   'Outer boundary'),
    ]:
        ax.add_patch(plt.Circle(
            CIRCLE_CENTRE, r,
            fill=False, color=color, linestyle='--', linewidth=2, label=label,
        ))

    # -- 12 segment boundary lines (inner ring to outer ring only) ---------
    for i in range(NUM_TOTAL_SEGMENTS):
        alpha = i * SEGMENT_ANGULAR_SIZE - np.pi/12
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        # Apply the same sign convention as gate positions
        xi = CIRCLE_CENTRE[0] - INNER_RADIUS * cos_a
        yi = CIRCLE_CENTRE[1] - INNER_RADIUS * sin_a
        xo = CIRCLE_CENTRE[0] - OUTER_RADIUS * cos_a
        yo = CIRCLE_CENTRE[1] - OUTER_RADIUS * sin_a
        ax.plot([xi, xo], [yi, yo],
                color='dimgray', linestyle=':', linewidth=1.0, alpha=0.55,
                label='Segment boundary' if i == 0 else None)

    # -- Reference markers -------------------------------------------------
    ax.plot(0.0, 0.0,
            'k+', markersize=12, markeredgewidth=2, label='Origin (0, 0)')
    ax.plot(STARTING_POINT[0], STARTING_POINT[1],
            'r+', markersize=12, markeredgewidth=2, label='Take-off pad')

    # -- Gates -------------------------------------------------------------
    arrow_scale = 0.30
    colors = plt.cm.tab10(np.linspace(0, 1, len(gates_df)))

    for idx, (_, gate) in enumerate(gates_df.iterrows()):
        x, y  = gate['x'], gate['y']
        theta = gate['theta']           # surface angle (CSV value)
        gnum  = int(gate['Gate'])

        # Approach normal for CCW flight = surface angle - pi/2
        normal_angle = theta - np.pi / 2
        dx = arrow_scale * np.cos(normal_angle)
        dy = arrow_scale * np.sin(normal_angle)

        ax.plot(x, y, 'o', color=colors[idx], markersize=9,
                label=f"Gate {gnum}")
        ax.arrow(x, y, dx, dy,
                 head_width=0.07, head_length=0.05,
                 fc=colors[idx], ec=colors[idx],
                 linewidth=2, length_includes_head=True)
        ax.annotate(
            f"G{gnum}",
            xy=(x, y), xytext=(x + 0.08, y + 0.08),
            fontsize=9, fontweight='bold', color=colors[idx],
        )

    # -- Labels, legend, limits --------------------------------------------
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(
        'Gate Positions and Orientations\n'
        'Arrows = gate normal',
        fontsize=11, fontweight='bold',
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Extend limits to accommodate the take-off pad and a margin
    margin = 0.4
    all_x = [CIRCLE_CENTRE[0] - OUTER_RADIUS, CIRCLE_CENTRE[0] + OUTER_RADIUS,
             STARTING_POINT[0]]
    all_y = [CIRCLE_CENTRE[1] - OUTER_RADIUS, CIRCLE_CENTRE[1] + OUTER_RADIUS,
             STARTING_POINT[1]]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)
    
    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig, ax


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "gate_pos_csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gates_info_generated.csv"

    print("=" * 65)
    print("  MICRO-502 Gate Configuration Generator")
    print("=" * 65)
    print(f"  Circle centre  : {CIRCLE_CENTRE}")
    print(f"  Inner radius   : {INNER_RADIUS} m")
    print(f"  Outer radius   : {OUTER_RADIUS} m")
    print(f"  Segments       : {NUM_TOTAL_SEGMENTS} x "
          f"{np.degrees(SEGMENT_ANGULAR_SIZE):.0f}deg = 360deg")
    print(f"  Gates          : {NUM_GATES} (non-consecutive segments)")
    print(f"  Gate width     : {GATE_WIDTH_BOUNDS} m")
    print(f"  Gate height    : {GATE_HEIGHT_BOUNDS} m")
    print(f"  Gate Z range   : {GATE_Z_BOUNDS} m")
    print(f"  Take-off pad   : {STARTING_POINT}  (opposite/negative-X side)")
    print("=" * 65 + "\n")

    gates_df = generate_gate_csv(
        output_filename=str(output_path),
        num_gates=NUM_GATES,
        seed=None,
        verbose=True,
    )

    print("\nGenerating 2D plot...")
    plot_gates(gates_df, show_plot=True)