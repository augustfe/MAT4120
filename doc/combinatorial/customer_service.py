import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.random import Generator, default_rng
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


def generate_locations(
    num_places: int, grid_size: int = 100, rng: Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = default_rng()
    return rng.uniform(0, grid_size, size=(num_places, 2))


def generate_building_costs(
    num_locations: int, rng: Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = default_rng()
    return rng.integers(10, 100, size=num_locations)


def generate_connection_costs(
    centre_loc: np.ndarray, customer_loc: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(customer_loc[:, None, :] - centre_loc[None, :, :], axis=-1)


def make_constraints(
    connection_costs: np.ndarray, building_costs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.concatenate([connection_costs.flatten(), building_costs])

    num_customers, num_centres = connection_costs.shape
    n_x = num_customers * num_centres
    n_y = num_centres

    # Equality constraints: each customer must be connected to exactly one centre
    A_eq = np.hstack(
        [
            np.kron(np.eye(num_customers), np.ones((1, num_centres))),
            np.zeros((num_customers, n_y)),
        ]
    )
    b_eq = np.ones(num_customers)

    # Inequality constraints: connection to a centre only if the centre is built
    # x_ij <= y_j -> x_ij - y_j <= 0
    A_ub = np.hstack(
        [
            np.eye(n_x),
            -np.kron(np.ones((num_customers, 1)), np.eye(num_centres)),
        ]
    )
    b_ub = np.zeros(n_x)

    return c, A_eq, b_eq, A_ub, b_ub


def customer_service_lp(
    connection_costs: np.ndarray, building_costs: np.ndarray
) -> tuple[np.ndarray, float]:
    c, A_eq, b_eq, A_ub, b_ub = make_constraints(connection_costs, building_costs)

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), method="highs")
    if not res.success:
        raise ValueError(
            f"Linear programming failed to find a solution.\nStatus: {res.message}"
        )

    return res.x, res.fun


def customer_service_ilp(
    connection_costs: np.ndarray, building_costs: np.ndarray
) -> tuple[np.ndarray, float]:
    c, A_eq, b_eq, A_ub, b_ub = make_constraints(connection_costs, building_costs)
    bounds = Bounds(0, 1)
    eq_constraint = LinearConstraint(A_eq, b_eq, b_eq)
    ub_constraint = LinearConstraint(A_ub, ub=b_ub)

    res = milp(
        c,
        constraints=(eq_constraint, ub_constraint),
        bounds=bounds,
        integrality=np.ones_like(c, dtype=int),
    )
    if not res.success:
        raise ValueError(
            f"Mixed integer linear programming failed to find a solution.\n"
            f"Status: {res.message}"
        )

    return res.x, res.fun


def plot_locations(
    centre_locations: np.ndarray,
    customer_locations: np.ndarray,
) -> None:
    cel = centre_locations
    cul = customer_locations
    plt.scatter(cel[:, 0], cel[:, 1], c="red", label="Centres", marker="s")
    plt.scatter(cul[:, 0], cul[:, 1], c="blue", label="Customers", marker="o")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Customer and Centre Locations")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_solution(
    centre_locations: np.ndarray,
    customer_locations: np.ndarray,
    solution: np.ndarray,
    ax: plt.Axes | None = None,
) -> None:
    if ax is None:
        plt.figure()
        ax = plt.gca()
    cel = centre_locations
    cul = customer_locations
    num_customers, num_centres = cul.shape[0], cel.shape[0]
    connection_vars = solution[: num_customers * num_centres].reshape(
        (num_customers, num_centres)
    )

    plt.scatter(cel[:, 0], cel[:, 1], c="red", label="Centres", marker="s")
    plt.scatter(cul[:, 0], cul[:, 1], c="blue", label="Customers", marker="o")

    mask = connection_vars > 0.02
    i_idx, j_idx = np.where(mask)

    seg_start = cul[i_idx]
    seg_end = cel[j_idx]
    segments = np.stack([seg_start, seg_end], axis=1)
    alphas = connection_vars[i_idx, j_idx].ravel()
    colors = np.zeros((len(alphas), 4))
    colors[:, 3] = alphas

    lc = LineCollection(segments, colors=colors, linewidths=1)
    ax = plt.gca()
    ax.add_collection(lc)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Customer-Centre Connections")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    rng = default_rng(42)
    num_centres = 50
    num_customers = 200
    grid_size = 300

    centre_locations = generate_locations(num_centres, grid_size=grid_size, rng=rng)
    customer_locations = generate_locations(num_customers, grid_size=grid_size, rng=rng)

    connection_costs = generate_connection_costs(centre_locations, customer_locations)
    building_costs = generate_building_costs(num_centres, rng=rng)

    print(
        f"{'Max Connection Costs:':<20} {connection_costs.max()}",
        f"{'Min Connection Costs:':<20} {connection_costs.min()}",
        f"{'Building Costs:':<20} {building_costs}",
        sep="\n",
    )

    solution, total_cost = customer_service_lp(connection_costs, building_costs)

    print(f"{'Total Cost:':<20} {total_cost}")
    plot_solution(centre_locations, customer_locations, solution)

    # Integral solution
    i_solution, i_total_cost = customer_service_ilp(connection_costs, building_costs)
    print(f"{'Integral Total Cost:':<20} {i_total_cost}")
    plot_solution(centre_locations, customer_locations, i_solution)

    plt.show()
