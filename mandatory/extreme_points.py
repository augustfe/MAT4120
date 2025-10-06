import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp

TOL = 1e-9


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


def _unique_points(points: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for candidate in points:
        if not any(np.allclose(candidate, existing, atol=TOL) for existing in unique):
            unique.append(candidate)
    return unique


def _clip_line_to_box(
    a_vec: np.ndarray,
    b_val: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    x_max = y_max = max(x_max, y_max)

    points: list[np.ndarray] = []
    a_x, a_y = a_vec

    if abs(a_y) > TOL:
        for x in (x_min, x_max):
            y = (b_val - a_x * x) / a_y
            if y_min - TOL <= y <= y_max + TOL:
                points.append(np.array([x, y], dtype=float))

    if abs(a_x) > TOL:
        for y in (y_min, y_max):
            x = (b_val - a_y * y) / a_x
            if x_min - TOL <= x <= x_max + TOL:
                points.append(np.array([x, y], dtype=float))

    points = _unique_points(points)
    if len(points) < 2:
        return None

    direction = np.array([-a_y, a_x], dtype=float)
    if np.linalg.norm(direction) <= TOL:
        direction = np.array([1.0, 0.0])

    scores = [np.dot(point, direction) for point in points]
    order = np.argsort(scores)
    return points[order[0]], points[order[-1]]


def _build_axis_elements(bounds: PlotBounds) -> tuple[list[str], list[str]]:
    x_min, y_min = np.floor(bounds.mins).astype(int)
    x_max, y_max = np.ceil(bounds.maxs).astype(int)

    # Axis lines with labels
    def command(x_min, x_max, y_min, y_max, axis=True, on_x=True) -> str:
        symbol = "\\i" if not axis else "x" if on_x else "y"
        desc = "axis, ->" if axis else "tick"
        where = "below" + " right" * axis if on_x else "above " * axis + "left"
        # if axis:
        #     where += " right" if on_x else " above"
        return (
            f"\\draw[{desc}] "
            f"({x_min},{y_min}) -- ({x_max},{y_max}) "
            f"node[axis-label, {where}] {{${symbol}$}};"
        )

    axis_commands = [
        command(x_min - 0.5, x_max + 0.5, 0, 0, on_x=True),
        command(0, 0, y_min - 0.5, y_max + 0.5, on_x=False),
    ]

    def foreach_block(start: int, end: int, body: str) -> str:
        return f"\\foreach \\i in {{{start},...,{end}}} {{\n{body}\n}}\n"

    x_tick_body = f"\t{command('\\i', '\\i', 0.1, -0.1, False, True)}"
    y_tick_body = f"\t{command(0.1, -0.1, '\\i', '\\i', False, False)}"

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
) -> list[str]:
    A_np = np.array(A.tolist(), dtype=float)
    b_np = np.array(b.tolist(), dtype=float).reshape(-1)

    box_min = np.array([0.5, 0.5])
    box_max = np.maximum(bounds.maxs, box_min)

    entries: list[tuple[str, str]] = []

    for row, (a_vec, b_val) in enumerate(zip(A_np, b_np, strict=True)):
        if np.allclose(a_vec, 0.0, atol=1e-9):
            continue

        segment = _clip_line_to_box(
            a_vec, b_val, (box_min[0], box_max[0]), (box_min[1], box_max[1])
        )
        if segment is None:
            continue
        s0, s1 = map(_format_point, segment)
        entries.append((f"{s0} -- {s1}", f"$a_{{{row}}}$"))

    if not entries:
        return []

    formatted_entries = []
    for idx, (segment, label_text) in enumerate(entries):
        separator = "," if idx < len(entries) - 1 else ""
        formatted_entries.append(f"\t{{{segment}}}/{{{label_text}}}{separator}")

    constraint_line = (
        "\t\\draw[constraint] \\segment node[left, constraint-label] {\\labeltext};"
    )

    return [
        "%%%%%%%%%% Constraint lines %%%%%%%%%%",
        "\\foreach \\segment/\\labeltext in {",
        *formatted_entries,
        "} {",
        constraint_line,
        "}",
        "",
    ]


def _build_vertex_elements(geometry: GeometryData, bounds: PlotBounds) -> list[str]:
    vertex_label_offset = 0.05 * bounds.max_span
    vertex_label_margin = 0.02 * bounds.max_span

    lower_vertex_bounds = bounds.clip_min + vertex_label_margin
    upper_vertex_bounds = bounds.clip_max - vertex_label_margin

    entries: list[tuple[str, str, str]] = []

    for idx in geometry.ordered_indices:
        expr_x, expr_y = geometry.point_exprs[idx]
        point = geometry.coords[idx]

        direction = point - geometry.centroid
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            direction = np.array([1.0, 1.0])
            direction_norm = math.sqrt(2.0)

        label_point = point + (direction / direction_norm) * vertex_label_offset
        label_point = np.clip(label_point, lower_vertex_bounds, upper_vertex_bounds)

        f_x = _format_number(expr_x)
        f_y = _format_number(expr_y)
        f_label = _format_point(label_point)
        label_text = f"$({f_x}, {f_y})$"

        entries.append((f"({f_x},{f_y})", f_label, label_text))

    if not entries:
        return []

    formatted_entries = []
    for idx, (vertex_point, label_point, label_text) in enumerate(entries):
        separator = "," if idx < len(entries) - 1 else ""
        formatted_entries.append(
            f"\t{{{vertex_point}}}/{{{label_point}}}/{{{label_text}}}{separator}"
        )

    return [
        "\\def\\VertexRadius{1.5pt}",
        "%%%%%%%%%% Vertices %%%%%%%%%%",
        "\\foreach \\vertexpos/\\labelpos/\\labeltext in {",
        *formatted_entries,
        "} {",
        "\t\\filldraw[vertex] \\vertexpos circle (\\VertexRadius);",
        "\t\\node[vertex-label] at \\labelpos {\\labeltext};",
        "}",
        "",
    ]


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
    return [
        "\\begin{tikzpicture}[",
        *[f"\t\t{arg}," for arg in args],
        "\t]",
    ]


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

    main_block = []

    geometry = _gather_geometry(extreme_pts)
    polygon_points = _build_feasible_region_polygon(geometry)
    main_block.extend(_tikz_fill_command(polygon_points))

    bounds = _compute_plot_bounds(geometry)
    main_block.extend(_build_axis_elements(bounds))
    main_block.append(f"\\draw[vertex-line] {polygon_points} -- cycle;")
    main_block.extend(_build_constraint_elements(A, b, bounds))
    main_block.extend(_build_vertex_elements(geometry, bounds))

    main_block = [f"\t{line}" for line in main_block]

    all_lines = "\n".join(tikz_lines + main_block + ["\\end{tikzpicture}"])
    all_lines = all_lines.replace("\t", " " * 2)

    path.write_text(all_lines + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    A = sp.Matrix([[1, -1], [-1, 1], [0, -2], [8, -1], [-1, -1]])
    b = sp.Matrix(5, 1, [0, 1, -5, 16, -4])
    points = extreme_points(A, b)
    write_tikz_figure(A, b, points)
    print(points)
