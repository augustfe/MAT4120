from itertools import combinations

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


if __name__ == "__main__":
    A = sp.Matrix([[1, -1], [-1, 1], [0, -2], [8, -1], [-1, -1]])
    b = sp.Matrix(5, 1, [0, 1, -5, 16, -4])
    print(extreme_points(A, b))
