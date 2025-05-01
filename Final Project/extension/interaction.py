import numpy as np
from itertools import product


# Calculates all allowed interaction tuples for a given dimension and maximum radius
def precompute_interaction_tuples(*, dimension: int, maximum_radius: float) -> np.ndarray[tuple]:
    INTERACTION_RADIUS_SQUARED = maximum_radius ** 2

    # Generate all possible positive offsets within the interaction radius
    interaction_offset_tuples = []
    for offset in product(range(int(maximum_radius)+1), repeat=dimension):
        if sum(offset) == 0: continue
        if sum([i**2 for i in offset]) > INTERACTION_RADIUS_SQUARED: continue
        interaction_offset_tuples.append(offset)

    # Convert to numpy array of tuples and return
    return np.array(interaction_offset_tuples, dtype=int)


# Calculates the corresponding interaction weights for a given set of offsets and interaction power falloff
def precompute_interaction_weights(*, interaction_offsets, power) -> float:

    # Calculate the interaction weighting
    interaction_weighting = []
    for offset in interaction_offsets:
        if power == 0: interaction_weighting.append(1.0)
        else: interaction_weighting.append(pow(np.sqrt(sum([i**2 for i in offset])), -power))

    # Return the interaction weighting
    return interaction_weighting


# Calculates the interactions between grid points in N-dimensional space.
def interaction_products(*, grid: np.ndarray, offset: np.ndarray) -> float:

    # Make sure inputs are correct
    assert isinstance(grid,   np.ndarray) and grid.dtype == np.float64, "Grid must be a numpy array of floats."
    assert isinstance(offset, np.ndarray) and offset.dtype == np.int64, "Offset must be a numpy array of integers."
    assert np.all(offset >= 0), "Offsets must be non-negative integers."
    assert offset.shape[0] == grid.ndim - 1, "Offset must have the same number of elements as grid dimensions."

    # Calculate the interaction
    dimensions = grid.ndim - 1
    if dimensions == 1:
        return interact_1D(grid, offset[0])
    elif dimensions == 2:
        return interact_2D(grid, offset[0], offset[1])
    elif dimensions == 3:
        return interact_3D(grid, offset[0], offset[1], offset[2])
    else:
        raise NotImplementedError("Interaction not implemented for dimensions > 3.")

def interact_1D(grid: np.ndarray, Δx: int) -> float:
    
        # Calculate interactions
        if Δx == 0:
            # Self-interaction term
            return np.sum(grid * grid)
        else:
            # 1 Diagonal per 1D rectangle
            return np.sum(grid[:-Δx] * grid[Δx:])

def interact_2D(grid: np.ndarray, Δx: int, Δy: int) -> float:

    # Calculate interactions
    if Δx == 0 and Δy == 0:
        # Self-interaction term
        return np.sum(grid * grid)
    elif Δx == 0:
        # 1 Diagonal per 1D rectangle
        return np.sum(grid[:, :-Δy] * grid[:, Δy:])
    elif Δy == 0:
        # 1 Diagonal per 1D rectangle
        return np.sum(grid[:-Δx, :] * grid[Δx:, :])
    else:
        # 2 Diagonals per 2D rectangle
        return np.sum(grid[:-Δx, :-Δy] * grid[Δx:, Δy:] + grid[:-Δx, Δy:] * grid[Δx:, :-Δy])

def interact_3D(grid: np.ndarray, Δx: int, Δy: int, Δz: int) -> float:

    # Calculate interactions
    if Δx == 0 and Δy == 0 and Δz == 0:
        # Self-interaction term
        return np.sum(grid * grid)
    elif Δx == 0 and Δy == 0:
        # 1 Diagonal per 1D rectangle
        return np.sum(grid[:, :, :-Δz] * grid[:, :, Δz:])
    elif Δx == 0 and Δz == 0:
        # 1 Diagonal per 1D rectangle
        return np.sum(grid[:, :-Δy, :] * grid[:, Δy:, :])
    elif Δy == 0 and Δz == 0:
        # 1 Diagonal per 1D rectangle
        return np.sum(grid[:-Δx, :, :] * grid[Δx:, :, :])
    elif Δx == 0:
        # 2 Diagonals per 2D rectangle
        return np.sum(grid[:, :-Δy, :-Δz] * grid[:, Δy:, Δz:] + grid[:, :-Δy, Δz:] * grid[:, Δy:, :-Δz])
    elif Δy == 0:
        # 2 Diagonals per 2D rectangle
        return np.sum(grid[:-Δx, :, :-Δz] * grid[Δx:, :, Δz:] + grid[:-Δx, :, Δz:] * grid[Δx:, :, :-Δz])
    elif Δz == 0:
        # 2 Diagonals per 2D rectangle
        return np.sum(grid[:-Δx, :-Δy, :] * grid[Δx:, Δy:, :] + grid[:-Δx, Δy:, :] * grid[Δx:, :-Δy, :])
    else:
        # 4 Diagonals per 3D rectangle (rectanguloid)
        return np.sum(grid[:-Δx, :-Δy, :-Δz] * grid[Δx:, Δy:, Δz:] + grid[:-Δx, :-Δy, Δz:] * grid[Δx:, Δy:, :-Δz] +
                      grid[:-Δx, Δy:, :-Δz] * grid[Δx:, :-Δy, Δz:] + grid[:-Δx, Δy:, Δz:] * grid[Δx:, :-Δy, :-Δz])

# Test the interaction function for all combinations of offsets
def _test_interaction_products(*, size: int, dimension: int, print_intermediate_successes: bool=False):

    # Test the interaction function for all combinations of offsets
    CALCULATED_TOTAL_INTERACTIONS = 0

    # Make a grid of spins all in the same direction
    fill_vector = np.zeros((dimension,), dtype=np.float64)
    fill_vector[0] = 1.0
    grid = np.full((size,) * dimension + fill_vector.shape, fill_vector, dtype=np.float64)

    # Test the interaction function for all combinations of interaction offsets
    for offset in product(*[range(size)], repeat=dimension):

        # Calculate the number of interactions for the given offset
        calculated_interactions = int(interaction_products(grid=grid, offset=np.array(offset)))

        # Calculate expected interactions
        multiplicative_factor = np.clip(2 ** (sum([i != 0 for i in offset]) - 1), 1, 2**(dimension-1))
        expected_interactions = np.prod([size - i for i in offset]) * multiplicative_factor

        # Assert that the calculated interactions match the expected interactions
        assert expected_interactions == calculated_interactions, f"Test failed: {offset} Expected {expected_interactions}, got {calculated_interactions}"
        CALCULATED_TOTAL_INTERACTIONS += calculated_interactions

        # Print success message if the test passes and print_successes is True
        if print_intermediate_successes: print(f"Test passed: interact({offset}) = {calculated_interactions}")

    # Calculate the expected total interactions
    K = size ** dimension
    CALCULATED_TOTAL_INTERACTIONS -= K             # Subtract K for the self-interaction term
    EXPECTED_TOTAL_INTERACTIONS = K * (K - 1) / 2  # K-graph number of vertices

    # Assert that the calculated total interactions match the expected total interactions
    assert CALCULATED_TOTAL_INTERACTIONS == EXPECTED_TOTAL_INTERACTIONS, f"Test failed: Expected {EXPECTED_TOTAL_INTERACTIONS}, got {CALCULATED_TOTAL_INTERACTIONS}"
    if print_intermediate_successes: print(f"Total interactions: {CALCULATED_TOTAL_INTERACTIONS}")  # Subtract N^2 for the self-interaction term
    print("All tests passed!")

    

# Main function to run the tests
if __name__ == "__main__":
    # Test the functions
    _test_interaction_products(size=5, dimension=3, print_intermediate_successes=False)
