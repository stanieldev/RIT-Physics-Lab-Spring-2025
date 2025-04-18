import numpy as np
from vector3 import Vector3




# Takes in a grid of spins and returns the interactions between them
def interact_2D(grid: np.ndarray, Δx: int, Δy: int):

    print(grid.dtype)

    # Make sure inputs are correct
    assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
    assert grid.ndim == 2,               "Grid must be a 2D array."
    assert grid.dtype == Vector3,        "Grid must be a numpy array of floats."


    assert isinstance(Δx, int), "Δx must be an integer."
    assert isinstance(Δy, int), "Δy must be an integer."
    assert Δx >= 0 and Δy >= 0, "Δx and Δy must be non-negative integers."

    # Calculate interactions
    if Δx == 0 and Δy == 0:
        return np.sum(grid * grid)
    elif Δx == 0:
        return np.sum(grid[:, :-Δy] * grid[:, Δy:])
    elif Δy == 0:
        return np.sum(grid[:-Δx, :] * grid[Δx:, :])
    else:
        return np.sum(grid[:-Δx, :-Δy] * grid[Δx:, Δy:] + grid[:-Δx, Δy:] * grid[Δx:, :-Δy])




def test_interaction_2D(N: int, *, print_intermediate_successes: bool=False):

    # Test the interaction function for all combinations of Δx and Δy
    CALCULATED_TOTAL_INTERACTIONS = 0
    for Δx in range(N):
        for Δy in range(N):
            
            # Calculate the number of interactions for the given Δx and Δy
            calculated_interactions = int(interact_2D(np.ones((N, N)), Δx, Δy))
            expected_interactions = (N - Δx) * (N - Δy) * (1 if Δx == 0 or Δy == 0 else 2)

            # Assert that the calculated interactions match the expected interactions
            assert expected_interactions == calculated_interactions, f"Test failed: Expected {expected_interactions}, got {calculated_interactions}"
            CALCULATED_TOTAL_INTERACTIONS += calculated_interactions

            # Print success message if the test passes and print_successes is True
            if print_intermediate_successes: print(f"Test passed: interact({Δx}, {Δy}) = {calculated_interactions}")

    # Calculate the expected total interactions
    CALCULATED_TOTAL_INTERACTIONS -= N**2                # Subtract N^2 for the self-interaction term
    EXPECTED_TOTAL_INTERACTIONS = (N*N)*((N*N) - 1) / 2  # K-graph of N*N vertices

    # Assert that the calculated total interactions match the expected total interactions
    assert CALCULATED_TOTAL_INTERACTIONS == EXPECTED_TOTAL_INTERACTIONS, f"Test failed: Expected {EXPECTED_TOTAL_INTERACTIONS}, got {CALCULATED_TOTAL_INTERACTIONS}"
    if print_intermediate_successes: print(f"Total interactions: {CALCULATED_TOTAL_INTERACTIONS}")  # Subtract N^2 for the self-interaction term
    print("All tests passed!")



if __name__ == "__main__":
    test_interaction_2D(10, print_intermediate_successes=True)