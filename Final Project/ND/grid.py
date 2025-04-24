# Stanley Goodwin, 4/22/2025
import numpy as np


# Generates a grid of unit vectors in N-dimensional space.
def generate_grid(*, size: int, dimension: int, fill: np.ndarray | None = None) -> np.ndarray:
    """
    Generate a grid of unit vectors in N-dimensional space.

    If fill is None, the grid will be filled with random unit vectors with 'dimension' elements.
    if fill is a numpy array, the grid will be filled with unit vectors with the same direction as fill.

    The grid will have the shape (size, size, ..., size, dimension), where 'dimension' is the number of dimensions.
    """
    # Generate a uniform grid of unit vectors...
    if fill is None:
        # ... randomly distributed in the unit N-sphere
        grid  = np.random.uniform(-1, 1, size=(size,) * dimension + (dimension,))
        norms = np.linalg.norm(grid, axis=-1, keepdims=True)
        grid  = grid / norms
    elif isinstance(fill, np.ndarray):
        # ... with the same direction as fill
        if np.linalg.norm(fill) == 0: raise ValueError("Fill vector cannot be zero.")
        fill = fill / np.linalg.norm(fill)
        grid = np.full((size,) * dimension + fill.shape, fill, dtype=np.float64)
    else:
        raise ValueError("Fill must be a numpy array or None.")
    return grid


# Test the generate_grid function
def _test_generate_grid():
    """
    Test the generate_grid function.
    """
    # Test 1D grid
    grid_1d = generate_grid(size=5, dimension=1)
    assert grid_1d.shape == (5, 1), "Test 1D grid failed."
    assert np.all(np.isclose(np.linalg.norm(grid_1d, axis=-1), 1)), "Test 1D grid failed."

    # Test 2D grid
    grid_2d = generate_grid(size=7, dimension=2)
    assert grid_2d.shape == (7, 7, 2), "Test 2D grid failed."
    assert np.all(np.isclose(np.linalg.norm(grid_2d, axis=-1), 1)), "Test 2D grid failed."

    # Test 3D grid
    grid_3d = generate_grid(size=11, dimension=3)
    assert grid_3d.shape == (11, 11, 11, 3), "Test 3D grid failed."
    assert np.all(np.isclose(np.linalg.norm(grid_3d, axis=-1), 1)), "Test 3D grid failed."

    # Test fill vector
    fill_vector = np.array([1.0, 0.0])
    grid_fill = generate_grid(size=5, dimension=2, fill=fill_vector)
    assert grid_fill.shape == (5, 5, 2), "Test fill vector failed."
    assert np.all(np.isclose(grid_fill[..., :2], fill_vector)), "Test fill vector failed."


# Test the generate_grid function
if __name__ == "__main__":
    # Run the tests
    _test_generate_grid()
    print("All tests passed.")
