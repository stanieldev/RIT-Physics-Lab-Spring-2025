import numpy as np
from vector3 import Vector3


# Function to calculate the interaction of spins in a 2D or 3D grid
def interact_2D(grid: np.ndarray, Δx: int, Δy: int) -> float:

    # Make sure inputs are correct
    assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
    assert grid.ndim == 2,               "Grid must be a 2D array."
    assert grid.dtype == Vector3,        "Grid must be a numpy array of floats."
    assert isinstance(Δx, int), "Δx must be an integer."
    assert isinstance(Δy, int), "Δy must be an integer."
    assert Δx >= 0 and Δy >= 0, "Δx and Δy must be non-negative integers."

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

    # Make sure inputs are correct
    assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
    assert grid.ndim == 3,               "Grid must be a 3D array."
    assert grid.dtype == Vector3,        "Grid must be a numpy array of floats."
    assert isinstance(Δx, int), "Δx must be an integer."
    assert isinstance(Δy, int), "Δy must be an integer."
    assert isinstance(Δz, int), "Δz must be an integer."
    assert Δx >= 0 and Δy >= 0 and Δz >= 0, "Δx, Δy, and Δz must be non-negative integers."

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


# Function tests
def test_interaction_2D(N: int, *, print_intermediate_successes: bool=False):

    # Test the interaction function for all combinations of Δx and Δy
    CALCULATED_TOTAL_INTERACTIONS = 0

    # Make a grid of spins all in the same direction
    grid = np.zeros((N, N), dtype=Vector3)
    for i in range(N):
        for j in range(N):
            grid[i, j] = Vector3(1, 0, 0)

    # Test the interaction function for all combinations of Δx and Δy
    for Δx in range(N):
        for Δy in range(N):
            
            # Calculate the number of interactions for the given Δx and Δy
            calculated_interactions = int(interact_2D(grid, Δx, Δy))

            # Calculate expected interactions
            # if Δx == 0 and Δy == 0:
            #     # Self-interaction term
            #     expected_interactions = (N - Δx) * (N - Δy)
            # elif (Δx == 0) or (Δy == 0):
            #     # 1 Diagonal per 1D rectangle
            #     expected_interactions = (N - Δx) * (N - Δy) * 1
            # else:
            #     # 2 Diagonals per 2D rectangle
            #     expected_interactions = (N - Δx) * (N - Δy) * 2
            multiplicative_factor = np.clip(2 ** (sum([Δx != 0, Δy != 0]) - 1), 1, 2**(2-1))
            expected_interactions = (N - Δx) * (N - Δy) * multiplicative_factor

            # Assert that the calculated interactions match the expected interactions
            assert expected_interactions == calculated_interactions, f"Test failed: Expected {expected_interactions}, got {calculated_interactions}"
            CALCULATED_TOTAL_INTERACTIONS += calculated_interactions

            # Print success message if the test passes and print_successes is True
            if print_intermediate_successes: print(f"Test passed: interact({Δx}, {Δy}) = {calculated_interactions}")

    # Calculate the expected total interactions
    K = N*N
    CALCULATED_TOTAL_INTERACTIONS -= K           # Subtract N^2 for the self-interaction term
    EXPECTED_TOTAL_INTERACTIONS = K*(K - 1) / 2  # K-graph of N*N vertices

    # Assert that the calculated total interactions match the expected total interactions
    assert CALCULATED_TOTAL_INTERACTIONS == EXPECTED_TOTAL_INTERACTIONS, f"Test failed: Expected {EXPECTED_TOTAL_INTERACTIONS}, got {CALCULATED_TOTAL_INTERACTIONS}"
    if print_intermediate_successes: print(f"Total interactions: {CALCULATED_TOTAL_INTERACTIONS}")  # Subtract N^2 for the self-interaction term
    print("All tests passed!")

def test_interaction_3D(N: int, *, print_intermediate_successes: bool=False):

    # Test the interaction function for all combinations of Δx and Δy
    CALCULATED_TOTAL_INTERACTIONS = 0

    # Make a grid of spins all in the same direction
    grid = np.zeros((N, N, N), dtype=Vector3)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                grid[i, j, k] = Vector3(1, 0, 0)

    # Test the interaction function for all combinations of Δx and Δy
    for Δx in range(N):
        for Δy in range(N):
            for Δz in range(N):
                
                # Calculate the number of interactions for the given Δx and Δy
                calculated_interactions = int(interact_3D(grid, Δx, Δy, Δz))

                # Calculate expected interactions
                # if Δx == 0 and Δy == 0 and Δz == 0:
                #     expected_interactions = (N - Δx) * (N - Δy) * (N - Δz)
                # elif (Δx == 0 and Δy == 0) or (Δx == 0 and Δz == 0) or (Δy == 0 and Δz == 0):
                #     expected_interactions = (N - Δx) * (N - Δy) * (N - Δz) * 1  # 1 diagonal per 1-D rectangle
                # elif (Δx == 0) or (Δy == 0) or (Δz == 0):
                #     expected_interactions = (N - Δx) * (N - Δy) * (N - Δz) * 2  # 2 diagonals per rectangle
                # else:
                #     expected_interactions = (N - Δx) * (N - Δy) * (N - Δz) * 4  # 4 diagonals per rectanguloid
                multiplicative_factor = np.clip(2 ** (sum([Δx != 0, Δy != 0, Δz != 0]) - 1), 1, 2**(3-1))
                expected_interactions = (N - Δx) * (N - Δy) * (N - Δz) * multiplicative_factor
                
                # Assert that the calculated interactions match the expected interactions
                assert expected_interactions == calculated_interactions, f"Test failed: ({Δx}, {Δy}, {Δz}) Expected {expected_interactions}, got {calculated_interactions}"
                CALCULATED_TOTAL_INTERACTIONS += calculated_interactions

                # Print success message if the test passes and print_successes is True
                if print_intermediate_successes: print(f"Test passed: interact({Δx}, {Δy}, {Δz}) = {calculated_interactions}")

    # Calculate the expected total interactions
    K = N*N*N
    CALCULATED_TOTAL_INTERACTIONS -= K           # Subtract N^2 for the self-interaction term
    EXPECTED_TOTAL_INTERACTIONS = K*(K - 1) / 2  # K-graph of N*N*N vertices

    # Assert that the calculated total interactions match the expected total interactions
    assert CALCULATED_TOTAL_INTERACTIONS == EXPECTED_TOTAL_INTERACTIONS, f"Test failed: Expected {EXPECTED_TOTAL_INTERACTIONS}, got {CALCULATED_TOTAL_INTERACTIONS}"
    if print_intermediate_successes: print(f"Total interactions: {CALCULATED_TOTAL_INTERACTIONS}")  # Subtract N^2 for the self-interaction term
    print("All tests passed!")


# Main function to run tests
if __name__ == "__main__":
    test_interaction_2D(10, print_intermediate_successes=False)
    test_interaction_3D(10, print_intermediate_successes=False)
