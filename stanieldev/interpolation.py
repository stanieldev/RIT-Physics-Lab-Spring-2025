import numpy as np

# Closest point in list of discrete points (Single Variable)
def find_two_closest_points(*, 
    y: np.ndarray, 
    x: np.ndarray, 
    target_x: int | float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Find the two closest points to a target x value in a list of discrete points.
    Assumes the list of points is sorted by x values (will break if not).

    Parameters
    ----------
    y : np.ndarray
        The y values of the discrete points.
    x : np.ndarray
        The x values of the discrete points.
    target_x : int | float
        The target x value.

    Returns
    -------
    
    """
    
    # Type checking
    assert isinstance(y, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert len(y) == len(x), "Lists are not the same length"
    assert isinstance(target_x, (int, float))

    # Find the closest point to x0
    index_1 = np.argmin(np.abs(x - target_x))

    # Find the second closest point to x0
    if index_1 + 1 == len(x):
        index_2 = index_1 - 1
    elif index_1 - 1 == -1:
        index_2 = index_1 + 1
    elif x[index_1 + 1] < x[index_1 - 1]:
        index_2 = index_1 + 1
    elif x[index_1 + 1] > x[index_1 - 1]:
        index_2 = index_1 - 1

    # Return the two closest points
    return (x[index_1], y[index_1]), (x[index_2], y[index_2])

# Linear interpolation between two points (Single Variable)
def linear_interpolation(*, 
    p1: tuple[int|float, int|float],
    p2: tuple[int|float, int|float], 
    target_x: int|float
) -> int | float:
    """
    Linearly interpolate the value at a target x value between two points.

    Parameters
    ----------
    p1 : tuple[int|float, int|float]
        The first point.
    p2 : tuple[int|float, int|float]
        The second point.
    target_x : int | float
        The point at which to find the interpolated value.

    Returns
    -------
    int | float
        The linearly interpolated value at the target x value.
    """

    # Type checking
    assert isinstance(p1, tuple)
    assert len(p1) == 2    
    assert isinstance(p2, tuple)
    assert len(p2) == 2
    assert isinstance(target_x, (int, float))
    assert isinstance(p1[0], (int, float))
    assert isinstance(p1[1], (int, float))
    assert isinstance(p2[0], (int, float))
    assert isinstance(p2[1], (int, float))
    assert p1[0] != p2[0], "Points x-values cannot be the same"

    # Calculate the slope and return the interpolated value
    xa, ya = p1
    xb, yb = p2
    slope = (yb - ya) / (xb - xa)
    return slope * (target_x - xa) + ya
