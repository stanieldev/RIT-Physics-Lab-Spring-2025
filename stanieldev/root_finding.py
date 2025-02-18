import numpy as np
from interpolation import find_two_closest_points, linear_interpolation


# Binary Search Method for Continuous Functions (Single Variable)
def continuous_binary_search_value(*,
    f: callable, 
    lowerbound: int | float, 
    upperbound: int | float, 
    error: int | float, 
    max_iter: int
) -> float:
    """
    Binary Search Method for Continuous Functions (Single Variable)
    Returns the value of x that is the root of the function

    Parameters
    ----------
    f : callable
        The function to find the root of
    lowerbound : int | float
        The lower bound of the search
    upperbound : int | float
        The upper bound of the search
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations

    Returns
    -------
    float
        The value of x that is the root of the function
    """

    # Type Checking
    assert callable(f)
    assert isinstance(lowerbound, (int, float))
    assert isinstance(upperbound, (int, float))
    assert lowerbound < upperbound
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    for _ in range(max_iter):
        midpoint = (lowerbound + upperbound) / 2
        if f(midpoint) == 0 or (upperbound - lowerbound) / 2 < error:
            return midpoint
        elif f(midpoint) < 0:
            lowerbound = midpoint
        else:
            upperbound = midpoint
    else:
        return midpoint

def continuous_binary_search_function(*,
    f: callable, 
    lowerbound: int | float, 
    upperbound: int | float, 
    error: int | float, 
    max_iter: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary Search Method for Continuous Functions (Single Variable)
    Returns the lists of x and f(x) that lead toward the root of the function

    Parameters
    ----------
    f : callable
        The function to find the root of
    lowerbound : int | float
        The lower bound of the search
    upperbound : int | float
        The upper bound of the search
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The lists of x and f(x) that lead toward the root of the function
    """

    # Type Checking
    assert callable(f)
    assert isinstance(lowerbound, (int, float))
    assert isinstance(upperbound, (int, float))
    assert lowerbound < upperbound
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    x = [lowerbound, upperbound]
    y = [f(lowerbound), f(upperbound)]
    for _ in range(max_iter):
        midpoint = (x[-1] + x[-2]) / 2
        x.append(midpoint)
        y.append(f(midpoint))
        if y[-1] == 0 or (x[-1] - x[-2]) / 2 < error:
            return x, y
        elif y[-1] < 0:
            x[-2] = midpoint
        else:
            x[-1] = midpoint
    else:
        return x, y

def discrete_binary_search_value(*,
    y: np.ndarray,
    x: np.ndarray,
    lowerbound: int | float,
    upperbound: int | float,
    error: int | float,
    max_iter: int
) -> float:
    """
    Binary Search Method for Discrete Functions (Single Variable)
    Returns the value of x that is the root of the function

    Parameters
    ----------
    y : np.ndarray
        The y values of the discrete function
    x : np.ndarray
        The x values of the discrete function
    lowerbound : int | float
        The lower bound of the search
    upperbound : int | float
        The upper bound of the search
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations

    Returns
    -------
    float
        The value of x that is the root of the function
    """

    # Type Checking
    assert isinstance(y, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert len(y) == len(x), "Lists are not the same length"
    assert isinstance(lowerbound, (int, float))
    assert isinstance(upperbound, (int, float))
    assert lowerbound < upperbound
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    for _ in range(max_iter):
        midpoint = (lowerbound + upperbound) / 2
        midpoint_y = linear_interpolation(find_two_closest_points(x, midpoint), midpoint)
        if midpoint_y == 0 or (upperbound - lowerbound) / 2 < error:
            return midpoint
        elif midpoint_y < 0:
            lowerbound = midpoint
        else:
            upperbound = midpoint
    else:
        return midpoint

def discrete_binary_search_function():
    raise NotImplementedError("This function is not implemented yet")


# Secant Method for Continuous Functions (Single Variable)
def continuous_secant_method_value(*,
    f: callable, 
    x0: int | float, 
    x1: int | float, 
    error: int | float, 
    max_iter: int
) -> float:
    """
    Secant Method for Continuous Functions (Single Variable)
    Returns the value of x that is the root of the function

    Parameters
    ----------
    f : callable
        The function to find the root of
    x0 : int | float
        The first guess of the root
    x1 : int | float
        The second guess of the root
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations

    Returns
    -------
    float
        The value of x that is the root of the function
    """

    # Type Checking
    assert callable(f)
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert x0 != x1, "Guesses x0 and x1 cannot be the same"
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if np.abs(x2 - x1) < error:
            return x2
        x0, x1 = x1, x2
    else:
        return x2

def continuous_secant_method_function(
    f: callable,
    x0: int | float,
    x1: int | float,
    error: int | float,
    max_iter: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Secant Method for Continuous Functions (Single Variable)
    Returns the lists of x and f(x) that lead toward the root of the function

    Parameters
    ----------
    f : callable
        The function to find the root of
    x0 : int | float
        The first guess of the root
    x1 : int | float
        The second guess of the root
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The lists of x and f(x) that lead toward the root of the function
    """

    # Type Checking
    assert callable(f)
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert x0 != x1, "Guesses x0 and x1 cannot be the same"
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    x = [x0, x1]
    for _ in range(max_iter):
        x.append(x[-1] - f(x[-1]) * (x[-1] - x[-2]) / (f(x[-1]) - f(x[-2])))
        if np.abs(x[-1] - x[-2]) < error:
            return x, [f(i) for i in x]
    else:
        return x, [f(i) for i in x]

def discrete_secant_method_value(*,
    y: np.ndarray,
    x: np.ndarray,
    x0: int | float, 
    x1: int | float, 
    error: int | float, 
    max_iter: int
):
    """
    Secant Method for Discrete Functions (Single Variable)
    Returns the value of x that is the root of the function

    Parameters
    ----------
    y : np.ndarray
        The y values of the discrete function
    x : np.ndarray
        The x values of the discrete function
    x0 : int | float
        The first guess of the root
    x1 : int | float
        The second guess of the root
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations
    
    Returns
    -------
    float
        The value of x that is the root of the function
    """

    # Type Checking
    assert isinstance(y, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert len(y) == len(x), "Lists are not the same length"
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert x0 != x1, "Guesses x0 and x1 cannot be the same"
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    for _ in range(max_iter):

        # Approximate the y-value at x0
        p1, p2 = find_two_closest_points(x, x0)
        y0 = linear_interpolation(p1, p2, x0)

        # Approximate the y-value at x1
        p1, p2 = find_two_closest_points(x, x1)
        y1 = linear_interpolation(p1, p2, x1)

        # Secant Step
        x2 = x1 - y1 * (x1 - x0) / (y1 - y0)
        if np.abs(x2 - x1) < error:
            return x2
        x0, x1 = x1, x2
    else:
        return x2

def discrete_secant_method_function(*,
    y: np.ndarray,
    x: np.ndarray,
    x0: int | float, 
    x1: int | float, 
    error: int | float, 
    max_iter: int
):
    """
    Secant Method for Discrete Functions (Single Variable)
    Returns the lists of x and f(x) that lead toward the root of the function

    Parameters
    ----------
    y : np.ndarray
        The y values of the discrete function
    x : np.ndarray
        The x values of the discrete function
    x0 : int | float
        The first guess of the root
    x1 : int | float
        The second guess of the root
    error : int | float
        The error tolerance
    max_iter : int
        The maximum number of iterations
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The lists of x and f(x) that lead toward the root of the function
    """

    # Type Checking
    assert isinstance(y, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert len(y) == len(x), "Lists are not the same length"
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert x0 != x1, "Guesses x0 and x1 cannot be the same"
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    x = [x0, x1]
    y = [linear_interpolation(find_two_closest_points(x, x0)), linear_interpolation(find_two_closest_points(x, x1))]
    for _ in range(max_iter):

        # Approximate the y-value at x0
        p1, p2 = find_two_closest_points(x, x[-2])
        y0 = linear_interpolation(p1, p2, x[-2])

        # Approximate the y-value at x1
        p1, p2 = find_two_closest_points(x, x[-1])
        y1 = linear_interpolation(p1, p2, x[-1])

        # Secant Step
        x.append(x[-1] - y1 * (x[-1] - x[-2]) / (y1 - y0))
        y.append(linear_interpolation(find_two_closest_points(x, x[-1]), x[-1]))
        if np.abs(x[-1] - x[-2]) < error:
            return x, y
    else:
        return x, y


# Newton's Method for Continuous Functions (Single Variable)
def continuous_newton_method_value(*,
    f: callable,
    f_prime: callable,
    x0: int | float,
    error: int | float,
    max_iter: int
) -> float:
    """
    """

    # Type Checking
    assert callable(f)
    assert callable(f_prime)
    assert isinstance(x0, (int, float))
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0

    # Implementation
    for _ in range(max_iter):
        x1 = x0 - f(x0) / f_prime(x0)
        if np.abs(x1 - x0) < error:
            return x1
        x0 = x1
    else:
        return x1

def continuous_newton_method_function():
    raise NotImplementedError("This function is not implemented yet")

def discrete_newton_method_value():
    raise NotImplementedError("This function is not implemented yet")

def discrete_newton_method_function():
    raise NotImplementedError("This function is not implemented yet")
