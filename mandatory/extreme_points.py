import math
from dataclasses import dataclass
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

    if math.isclose(max_val, min_val, rel_tol=1e-9, abs_tol=1e-9):
        return [min_val]

    start = math.floor(min(min_val, 0.0))
    end = math.ceil(max(max_val, 0.0))
    tick_values = {min_val, max_val}
    tick_values.update(float(value) for value in range(start, end + 1))
    return sorted(tick_values)


@dataclass
class GeometryData:
    point_exprs: list[tuple[sp.Expr, sp.Expr]]
    coords: np.ndarray
    ordered_indices: list[int]
    centroid: np.ndarray


@dataclass
class PlotBounds:
    mins: np.ndarray
    maxs: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray
    ranges: np.ndarray
    max_span: float


def _format_point(point: np.ndarray) -> str:
    return f"({point[0]:.4f},{point[1]:.4f})"


def _gather_geometry(extreme_pts: list[sp.Matrix]) -> GeometryData:
    point_exprs = [(pt[0], pt[1]) for pt in extreme_pts]
    coords = np.array(
        [[float(sp.N(pt[0])), float(sp.N(pt[1]))] for pt in extreme_pts],
        dtype=float,
    )
    ordered_indices = _order_points_counterclockwise(coords).tolist()
    centroid = coords.mean(axis=0)
    return GeometryData(point_exprs, coords, ordered_indices, centroid)


def _compute_plot_bounds(geometry: GeometryData) -> PlotBounds:
    mins = np.minimum(geometry.coords.min(axis=0), np.zeros(2))
    maxs = np.maximum(geometry.coords.max(axis=0), np.zeros(2))
    spans = maxs - mins
    margin = 0.2 * max(float(spans.max()), 1.0)

    clip_min = mins - margin
    clip_max = maxs + margin
    ranges = clip_max - clip_min
    max_span = max(float(ranges.max()), 1.0)

    return PlotBounds(mins, maxs, clip_min, clip_max, ranges, max_span)


def _build_feasible_region_polygon(geometry: GeometryData) -> str:
    return " -- ".join(
        f"({_format_number(geometry.point_exprs[idx][0])},"
        f"{_format_number(geometry.point_exprs[idx][1])})"
        for idx in geometry.ordered_indices
    )


def _build_axis_elements(bounds: PlotBounds) -> tuple[list[str], list[str]]:
    x_min, y_min = np.floor(bounds.mins).astype(int)
    x_max, y_max = np.ceil(bounds.maxs).astype(int)

    # Axis lines with labels
    def command(x_min, x_max, y_min, y_max, symbol, axis=True) -> str:
        desc = "axis, ->" if axis else "tick"
        if axis:
            where = "below right" if symbol == "x" else "above left"
        else:
            where = "below" if symbol == "x" else "left"
        return (
            f"\\draw[{desc}] "
            f"({x_min},{y_min}) -- ({x_max},{y_max}) "
            f"node[axis-label, {where}] {{${symbol}$}};"
        )

    axis_commands = [
        command(x_min - 0.5, x_max + 0.5, 0, 0, "x"),
        command(0, 0, y_min - 0.5, y_max + 0.5, "y"),
    ]

    def foreach_block(start: int, end: int, body: str) -> str:
        return f"\\foreach \\i in {{{start},...,{end}}} {{\n{body}\n}}\n"

    x_tick_body = f"\t{command('\\i', '\\i', 0.1, -0.1, '\\i', False)}"
    y_tick_body = f"\t{command(0.1, -0.1, '\\i', '\\i', '\\i', False)}"

    tick_commands = [
        *foreach_block(int(x_min), int(x_max), x_tick_body).splitlines(),
        *foreach_block(int(y_min), int(y_max), y_tick_body).splitlines(),
    ]
    axis = [
        "%" * 10 + " Axis lines " + "%" * 10,
        *axis_commands,
        "",
        "%" * 10 + " Axis ticks " + "%" * 10,
        *tick_commands,
        "",
    ]
    return axis


def _build_constraint_elements(
    A: sp.Matrix, b: sp.Matrix, bounds: PlotBounds
) -> tuple[list[str], list[str]]:
    constraint_line_commands: list[str] = []
    constraint_label_commands: list[str] = []

    constraint_label_offset = 0.04 * bounds.max_span
    constraint_label_margin = 0.02 * bounds.max_span

    A_np = np.array(A.tolist(), dtype=float)
    b_np = np.array(b.tolist(), dtype=float).reshape(-1)

    lower_bounds = bounds.clip_min + constraint_label_margin
    upper_bounds = bounds.clip_max - constraint_label_margin

    for row, (a_vec, b_val) in enumerate(zip(A_np, b_np, strict=True)):
        if np.allclose(a_vec, 0.0, atol=1e-9):
            continue

        if abs(a_vec[1]) >= 1e-9:
            x_values = np.array([bounds.clip_min[0], bounds.clip_max[0]])
            y_values = (b_val - a_vec[0] * x_values) / a_vec[1]
            start_point = np.array([x_values[0], y_values[0]])
            end_point = np.array([x_values[1], y_values[1]])
        else:
            x_val = b_val / a_vec[0]
            start_point = np.array([x_val, bounds.clip_min[1]])
            end_point = np.array([x_val, bounds.clip_max[1]])

        constraint_line_commands.append(
            "\\draw[constraint] "
            f"{_format_point(start_point)} -- {_format_point(end_point)};"
        )

        normal_norm = np.linalg.norm(a_vec)
        if normal_norm < 1e-9:
            continue

        midpoint = 0.5 * (start_point + end_point)
        normal_direction = a_vec / normal_norm
        label_point = midpoint + normal_direction * constraint_label_offset
        label_point = np.clip(label_point, lower_bounds, upper_bounds)

        constraint_label_commands.append(
            "\\node[constraint-label] "
            f"at {_format_point(label_point)} {{$a_{{{row}}}$}};"
        )

    return constraint_line_commands, constraint_label_commands


def _build_vertex_elements(
    geometry: GeometryData, bounds: PlotBounds
) -> tuple[list[str], list[str]]:
    vertex_circle_commands: list[str] = []
    vertex_label_commands: list[str] = []

    vertex_label_offset = 0.05 * bounds.max_span
    vertex_label_margin = 0.02 * bounds.max_span

    lower_vertex_bounds = bounds.clip_min + vertex_label_margin
    upper_vertex_bounds = bounds.clip_max - vertex_label_margin

    for idx in geometry.ordered_indices:
        expr_x, expr_y = geometry.point_exprs[idx]
        point = geometry.coords[idx]
        formatted_x = _format_number(expr_x)
        formatted_y = _format_number(expr_y)

        vertex_circle_commands.append(
            f"\\filldraw[vertex] ({formatted_x},{formatted_y}) circle (3pt);"
        )

        direction = point - geometry.centroid
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            direction = np.array([1.0, 1.0])
            direction_norm = math.sqrt(2.0)

        label_point = point + (direction / direction_norm) * vertex_label_offset
        label_point = np.clip(label_point, lower_vertex_bounds, upper_vertex_bounds)
        label_text = f"$({formatted_x}, {formatted_y})$"

        vertex_label_commands.append(
            f"\\node[vertex-label] at {_format_point(label_point)} {{{label_text}}};"
        )

    return vertex_circle_commands, vertex_label_commands


def _tikz_style_block() -> str:
    args = [
        "scale=1.2",
        ">=stealth",
        "axis/.style={color=axis_line, line width=0.8pt}",
        "tick/.style={axis, line width=0.6pt}",
        "axis-label/.style={color=axis_text, font=\\scriptsize}",
        "constraint/.style={color=constraint_line, line width=0.8pt}",
        "constraint-label/.style={constraint, font=\\scriptsize}",
        "vertex-line/.style={color=vertex_point, line width=1pt}",
        "vertex/.style={color=vertex_point, fill=white, line width=0.9pt}",
        "vertex-label/.style={color=vertex_point, font=\\scriptsize}",
    ]
    style = [
        "\\begin{tikzpicture}[",
        *[f"\t\t{arg}," for arg in args],
        "\t]",
    ]
    # print(style)

    return style


def _tikz_clip_command(bounds: PlotBounds) -> str:
    return (
        "\\clip "
        f"{_format_point(bounds.clip_min)} "
        f"rectangle {_format_point(bounds.clip_max)};"
    )


def _tikz_fill_command(polygon_points: str) -> list[str]:
    return [
        "%" * 10 + " Feasible region " + "%" * 10,
        f"\\fill[feasible_region] {polygon_points} -- cycle;",
        "",
    ]


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

    tikz_lines = _tikz_style_block()

    geometry = _gather_geometry(extreme_pts)
    polygon_points = _build_feasible_region_polygon(geometry)
    tikz_lines.extend([f"\t{cmd}" for cmd in _tikz_fill_command(polygon_points)])

    bounds = _compute_plot_bounds(geometry)
    tikz_lines.extend([f"\t{cmd}" for cmd in _build_axis_elements(bounds)])

    constraint_line_commands, constraint_label_commands = _build_constraint_elements(
        A, b, bounds
    )
    vertex_circle_commands, vertex_label_commands = _build_vertex_elements(
        geometry, bounds
    )

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
