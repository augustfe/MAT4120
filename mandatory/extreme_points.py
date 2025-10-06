import math
from itertools import combinations
from pathlib import Path

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


def _order_points_counterclockwise(points: list[tuple[float, float]]) -> list[int]:
    if len(points) <= 1:
        return list(range(len(points)))

    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)

    indexed_points = list(enumerate(points))
    indexed_points.sort(key=lambda item: math.atan2(item[1][1] - cy, item[1][0] - cx))
    return [index for index, _ in indexed_points]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _axis_tick_values(min_val: float, max_val: float) -> list[float]:
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    if math.isclose(max_val, min_val, rel_tol=1e-9, abs_tol=1e-9):
        return [min_val]

    start = math.floor(min_val)
    end = math.ceil(max_val)
    tick_values = {min_val, max_val}
    tick_values.update(float(value) for value in range(start, end + 1))
    filtered = [
        tick for tick in tick_values if min_val - 1e-9 <= tick <= max_val + 1e-9
    ]
    return sorted(filtered)


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
    point_coords = [(float(sp.N(pt[0])), float(sp.N(pt[1]))) for pt in extreme_pts]

    xs = [coord[0] for coord in point_coords]
    ys = [coord[1] for coord in point_coords]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    span_x = x_max - x_min
    span_y = y_max - y_min
    margin = 0.2 * max(span_x, span_y, 1.0)

    clip_x_min = x_min - margin
    clip_x_max = x_max + margin
    clip_y_min = y_min - margin
    clip_y_max = y_max + margin

    range_x = clip_x_max - clip_x_min
    range_y = clip_y_max - clip_y_min
    max_span = max(range_x, range_y, 1.0)

    ordered_indices = _order_points_counterclockwise(point_coords)
    polygon_points = " -- ".join(
        f"({_format_number(point_exprs[idx][0])},{_format_number(point_exprs[idx][1])})"
        for idx in ordered_indices
    )

    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)

    axis_offset = 0.08 * max_span
    x_axis_y = clip_y_min + min(axis_offset, max(0.0, 0.25 * range_y))
    y_axis_x = clip_x_min + min(axis_offset, max(0.0, 0.25 * range_x))

    x_available = max(0.0, clip_y_max - x_axis_y)
    y_available = max(0.0, clip_x_max - y_axis_x)
    x_tick_length = min(0.03 * max_span, x_available * 0.5)
    y_tick_length = min(0.03 * max_span, y_available * 0.5)
    x_label_gap = min(0.04 * max_span, x_available * 0.9)
    y_label_gap = min(0.04 * max_span, y_available * 0.9)

    axis_commands = [
        (
            "\\draw[axis, ->] "
            f"({clip_x_min:.4f},{x_axis_y:.4f}) -- "
            f"({clip_x_max:.4f},{x_axis_y:.4f}) "
            "node[axis-label, anchor=west] {$x$};"
        ),
        (
            "\\draw[axis, ->] "
            f"({y_axis_x:.4f},{clip_y_min:.4f}) -- "
            f"({y_axis_x:.4f},{clip_y_max:.4f}) "
            "node[axis-label, anchor=south] {$y$};"
        ),
    ]

    x_tick_commands: list[str] = []
    for tick in _axis_tick_values(x_min, x_max):
        tick_expr = sp.nsimplify(tick)
        label_text = f"${_format_number(tick_expr)}$"
        x_tick_commands.append(
            "\\draw[tick] "
            f"({tick:.4f},{x_axis_y:.4f}) -- "
            f"({tick:.4f},{x_axis_y + x_tick_length:.4f});"
        )
        x_tick_commands.append(
            "\\node[axis-label, anchor=south] "
            f"at ({tick:.4f},{x_axis_y + x_tick_length + x_label_gap:.4f}) "
            f"{{{label_text}}};"
        )

    y_tick_commands: list[str] = []
    for tick in _axis_tick_values(y_min, y_max):
        tick_expr = sp.nsimplify(tick)
        label_text = f"${_format_number(tick_expr)}$"
        y_tick_commands.append(
            "\\draw[tick] "
            f"({y_axis_x:.4f},{tick:.4f}) -- "
            f"({y_axis_x + y_tick_length:.4f},{tick:.4f});"
        )
        y_tick_commands.append(
            "\\node[axis-label, anchor=west] "
            f"at ({y_axis_x + y_tick_length + y_label_gap:.4f},{tick:.4f}) "
            f"{{{label_text}}};"
        )

    constraint_line_commands: list[str] = []
    constraint_label_commands: list[str] = []
    constraint_label_offset = 0.04 * max_span
    constraint_label_margin = 0.02 * max_span
    for row in range(A.rows):
        a1 = float(sp.N(A[row, 0]))
        a2 = float(sp.N(A[row, 1]))
        b_val = float(sp.N(b[row]))

        if abs(a1) < 1e-9 and abs(a2) < 1e-9:
            continue

        if abs(a2) >= 1e-9:
            start_point = (clip_x_min, (b_val - a1 * clip_x_min) / a2)
            end_point = (clip_x_max, (b_val - a1 * clip_x_max) / a2)
        else:
            x_val = b_val / a1
            start_point = (x_val, clip_y_min)
            end_point = (x_val, clip_y_max)

        line_command = (
            "\\draw[constraint] "
            f"({start_point[0]:.4f},{start_point[1]:.4f}) -- "
            f"({end_point[0]:.4f},{end_point[1]:.4f});"
        )
        constraint_line_commands.append(line_command)

        normal_norm = math.hypot(a1, a2)
        if normal_norm < 1e-9:
            continue

        mid_x = 0.5 * (start_point[0] + end_point[0])
        mid_y = 0.5 * (start_point[1] + end_point[1])
        offset_x = (a1 / normal_norm) * constraint_label_offset
        offset_y = (a2 / normal_norm) * constraint_label_offset
        label_x = _clamp(
            mid_x + offset_x,
            clip_x_min + constraint_label_margin,
            clip_x_max - constraint_label_margin,
        )
        label_y = _clamp(
            mid_y + offset_y,
            clip_y_min + constraint_label_margin,
            clip_y_max - constraint_label_margin,
        )
        label_text = f"$a_{{{row}}}$"
        constraint_label_commands.append(
            "\\node[constraint-label] "
            f"at ({label_x:.4f},{label_y:.4f}) {{{label_text}}};"
        )

    vertex_circle_commands: list[str] = []
    vertex_label_commands: list[str] = []
    vertex_label_offset = 0.05 * max_span
    vertex_label_margin = 0.02 * max_span
    for idx in ordered_indices:
        expr_x, expr_y = point_exprs[idx]
        coord_x, coord_y = point_coords[idx]
        formatted_x = _format_number(expr_x)
        formatted_y = _format_number(expr_y)

        vertex_circle_commands.append(
            f"\\filldraw[vertex] ({formatted_x},{formatted_y}) circle (3pt);"
        )

        direction_x = coord_x - centroid_x
        direction_y = coord_y - centroid_y
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm < 1e-9:
            direction_x, direction_y, direction_norm = 1.0, 1.0, math.sqrt(2.0)

        label_x = coord_x + (direction_x / direction_norm) * vertex_label_offset
        label_y = coord_y + (direction_y / direction_norm) * vertex_label_offset
        label_x = _clamp(
            label_x,
            clip_x_min + vertex_label_margin,
            clip_x_max - vertex_label_margin,
        )
        label_y = _clamp(
            label_y,
            clip_y_min + vertex_label_margin,
            clip_y_max - vertex_label_margin,
        )
        label_text = f"$({formatted_x}, {formatted_y})$"
        vertex_label_commands.append(
            f"\\node[vertex-label] at ({label_x:.4f},{label_y:.4f}) {{{label_text}}};"
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
        f"({clip_x_min:.4f},{clip_y_min:.4f}) "
        f"rectangle ({clip_x_max:.4f},{clip_y_max:.4f});"
    )

    fill_command = f"  \\fill[feasible_region] {polygon_points} -- cycle;"

    tikz_lines = [style_block, clip_command, fill_command]

    for command in axis_commands + x_tick_commands + y_tick_commands:
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
