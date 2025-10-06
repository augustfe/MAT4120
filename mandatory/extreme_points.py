import math
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp


def find_feasible_point(A: sp.Matrix, b: sp.Matrix) -> list[sp.Matrix]:
    candidates = []

    for indices in combinations(range(A.rows), A.cols):
        A_sub = A[list(indices), :]
        b_sub = b[list(indices), :]
        if A_sub.rank() < A.cols:
            continue
        x = A_sub.LUsolve(b_sub)
        x.simplify()
        candidates.append(x)

    return candidates


def satisfies_constraints(A: sp.Matrix, b: sp.Matrix, x: sp.Matrix) -> bool:
    return all(A.row(i).dot(x) <= b[i] for i in range(A.rows))


def extreme_points(A: sp.Matrix, b: sp.Matrix) -> list[sp.Matrix]:
    candidates = find_feasible_point(A, b)
    valid_candidates = [x for x in candidates if satisfies_constraints(A, b, x)]

    unique = []
    for v in valid_candidates:
        if v not in unique:
            unique.append(v)

    return unique


def _format_number(value: sp.Expr) -> str:
    simplified = sp.nsimplify(value)
    if simplified.is_Rational or simplified.is_Integer:
        return str(simplified)
    return f"{float(sp.N(value)):.4f}"


def _order_points_counterclockwise(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.arange(points.shape[0])

    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])

    indexed_points = list(enumerate(points))
    indexed_points.sort(key=lambda item: math.atan2(item[1][1] - cy, item[1][0] - cx))
    return np.array([index for index, _ in indexed_points])


def _axis_tick_values(min_val: float, max_val: float) -> list[float]:
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    if np.isclose(max_val, min_val, atol=1e-9):
        return np.array([min_val])

    start = math.floor(min_val)
    end = math.ceil(max_val)
    ticks = list(range(start, end + 1))
    return ticks


def write_tikz_figure(
    A: sp.Matrix,
    b: sp.Matrix,
    extreme_pts: list[sp.Matrix],
    output_path: str | Path | None = None,
) -> Path:
    if A.cols != 2:
        raise ValueError("TikZ export only supported for 2D constraint systems.")

    if not extreme_pts:
        raise ValueError("No extreme points available to illustrate.")

    path = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).with_name("extreme_points.tex")
    )

    point_exprs = [(pt[0], pt[1]) for pt in extreme_pts]
    coords = np.array(
        [[sp.N(pt[0]), sp.N(pt[1])] for pt in extreme_pts],
        dtype=float,
    )

    mins = np.minimum(coords.min(axis=0), np.zeros(2))
    maxs = np.maximum(coords.max(axis=0), np.zeros(2))
    spans = maxs - mins

    margin = 0.2 * max(spans.max(), 1.0)

    clip_min = mins - margin
    clip_max = maxs + margin
    ranges = clip_max - clip_min
    max_span = max(ranges.max(), 1.0)

    ordered_indices = _order_points_counterclockwise(coords)
    polygon_points = " -- ".join(
        f"({_format_number(point_exprs[idx][0])},{_format_number(point_exprs[idx][1])})"
        for idx in ordered_indices
    )

    centroid = coords.mean(axis=0)

    axis_offset = 0.08 * max_span
    axis_base = clip_min + np.clip(axis_offset, 0.0, 0.25 * ranges)
    available = np.maximum(0.0, clip_max - axis_base)
    tick_length = np.minimum(0.03 * max_span, 0.5 * available)
    label_gap = np.minimum(0.04 * max_span, 0.9 * available)

    def _fmt_point(point: np.ndarray) -> str:
        return f"({point[0]:.4f},{point[1]:.4f})"

    axis_configs = [
        {
            "label": "{$x$}",
            "var_idx": 0,
            "const_idx": 1,
            "line_pos": axis_base[1],
            "tick_length": tick_length[1],
            "label_gap": label_gap[1],
            "ticks": _axis_tick_values(mins[0], maxs[0]),
            "tick_anchor": "south",
            "axis_label_anchor": "west",
        },
        {
            "label": "{$y$}",
            "var_idx": 1,
            "const_idx": 0,
            "line_pos": axis_base[0],
            "tick_length": tick_length[0],
            "label_gap": label_gap[0],
            "ticks": _axis_tick_values(mins[1], maxs[1]),
            "tick_anchor": "west",
            "axis_label_anchor": "south",
        },
    ]

    axis_commands: list[str] = []
    tick_commands: list[str] = []

    for config in axis_configs:
        var_idx = config["var_idx"]
        const_idx = config["const_idx"]
        line_pos = config["line_pos"]

        start = clip_min.copy()
        end = clip_min.copy()
        end[var_idx] = clip_max[var_idx]
        start[const_idx] = line_pos
        end[const_idx] = line_pos

        axis_commands.append(
            "\\draw[axis, ->] "
            f"{_fmt_point(start)} -- {_fmt_point(end)} "
            f"node[axis-label, anchor={config['axis_label_anchor']}] {config['label']};"
        )

        tick_len = config["tick_length"]
        label_gap_val = config["label_gap"]

        for tick in config["ticks"]:
            base_point = np.zeros(2)
            base_point[var_idx] = tick
            base_point[const_idx] = line_pos

            tick_end = base_point.copy()
            tick_end[const_idx] += tick_len

            label_point = base_point.copy()
            label_point[const_idx] += tick_len + label_gap_val

            tick_commands.append(
                f"\\draw[tick] {_fmt_point(base_point)} -- {_fmt_point(tick_end)};"
            )

            tick_expr = sp.nsimplify(tick)
            tick_label = f"${_format_number(tick_expr)}$"
            tick_commands.append(
                f"\\node[axis-label, anchor={config['tick_anchor']}] "
                f"at {_fmt_point(label_point)} {{{tick_label}}};"
            )

    constraint_line_commands: list[str] = []
    constraint_label_commands: list[str] = []
    constraint_label_offset = 0.04 * max_span
    constraint_label_margin = 0.02 * max_span

    A_np = np.array(A.tolist(), dtype=float)
    b_np = np.array(b.tolist(), dtype=float).reshape(-1)

    for row, (a_vec, b_val) in enumerate(zip(A_np, b_np, strict=True)):
        if np.allclose(a_vec, 0.0, atol=1e-9):
            continue

        if abs(a_vec[1]) >= 1e-9:
            x_values = np.array([clip_min[0], clip_max[0]])
            y_values = (b_val - a_vec[0] * x_values) / a_vec[1]
            start_point = np.array([x_values[0], y_values[0]])
            end_point = np.array([x_values[1], y_values[1]])
        else:
            x_val = b_val / a_vec[0]
            start_point = np.array([x_val, clip_min[1]])
            end_point = np.array([x_val, clip_max[1]])

        constraint_line_commands.append(
            f"\\draw[constraint] {_fmt_point(start_point)} -- {_fmt_point(end_point)};"
        )

        normal_norm = np.linalg.norm(a_vec)
        if normal_norm < 1e-9:
            continue

        midpoint = 0.5 * (start_point + end_point)
        normal_direction = a_vec / normal_norm
        label_point = midpoint + normal_direction * constraint_label_offset
        lower_bounds = clip_min + constraint_label_margin
        upper_bounds = clip_max - constraint_label_margin
        label_point = np.clip(label_point, lower_bounds, upper_bounds)

        constraint_label_commands.append(
            f"\\node[constraint-label] at {_fmt_point(label_point)} {{$a_{{{row}}}$}};"
        )

    vertex_circle_commands: list[str] = []
    vertex_label_commands: list[str] = []
    vertex_label_offset = 0.05 * max_span
    vertex_label_margin = 0.02 * max_span

    lower_vertex_bounds = clip_min + vertex_label_margin
    upper_vertex_bounds = clip_max - vertex_label_margin

    for idx in ordered_indices:
        expr_x, expr_y = point_exprs[idx]
        point = coords[idx]
        formatted_x = _format_number(expr_x)
        formatted_y = _format_number(expr_y)

        vertex_circle_commands.append(
            f"\\filldraw[vertex] ({formatted_x},{formatted_y}) circle (3pt);"
        )

        direction = point - centroid
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            direction = np.array([1.0, 1.0])
            direction_norm = math.sqrt(2.0)

        label_point = point + (direction / direction_norm) * vertex_label_offset
        label_point = np.clip(label_point, lower_vertex_bounds, upper_vertex_bounds)
        label_text = f"$({formatted_x}, {formatted_y})$"

        vertex_label_commands.append(
            f"\\node[vertex-label] at {_fmt_point(label_point)} {{{label_text}}};"
        )

    style_block = (
        "\\begin{tikzpicture}[\n"
        "  scale=1.2,\n"
        "  axis/.style={color=axis_line, line width=0.8pt},\n"
        "  tick/.style={axis, line width=0.6pt},\n"
        "  axis-label/.style={color=axis_text, font=\\scriptsize},\n"
        "  constraint/.style={color=constraint_line, line width=0.8pt},\n"
        "  constraint-label/.style={constraint, font=\\scriptsize},\n"
        "  vertex-line/.style={color=vertex_point, line width=1pt},\n"
        "  vertex/.style={color=vertex_point, fill=white, line width=0.9pt},\n"
        "  vertex-label/.style={color=vertex_point, font=\\scriptsize}\n"
        "]"
    )

    clip_command = (
        "  \\clip "
        f"({clip_min[0]:.4f},{clip_min[1]:.4f}) "
        f"rectangle ({clip_max[0]:.4f},{clip_max[1]:.4f});"
    )

    fill_command = f"  \\fill[feasible_region] {polygon_points} -- cycle;"

    tikz_lines = [style_block, clip_command, fill_command]

    for command in axis_commands + tick_commands:
        tikz_lines.append(f"  {command}")

    tikz_lines.append(f"  \\draw[vertex-line] {polygon_points} -- cycle;")

    for command in constraint_line_commands + constraint_label_commands:
        tikz_lines.append(f"  {command}")

    for command in vertex_circle_commands + vertex_label_commands:
        tikz_lines.append(f"  {command}")

    tikz_lines.append("\\end{tikzpicture}")

    path.write_text("\n".join(tikz_lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    A = sp.Matrix([[1, -1], [-1, 1], [0, -2], [8, -1], [-1, -1]])
    b = sp.Matrix(5, 1, [0, 1, -5, 16, -4])
    points = extreme_points(A, b)
    write_tikz_figure(A, b, points)
    print(points)
