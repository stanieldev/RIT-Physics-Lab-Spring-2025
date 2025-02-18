import numpy as np
from stanieldev.interpolation import find_two_closest_points, linear_interpolation


# Binary Search Method for Continuous Functions (Single Variable)
def continuous_binary_search_value():
    raise NotImplementedError("This function is not implemented yet")

def continuous_binary_search_function():
    raise NotImplementedError("This function is not implemented yet")

def discrete_binary_search_value():
    raise NotImplementedError("This function is not implemented yet")

def discrete_binary_search_function():
    raise NotImplementedError("This function is not implemented yet")


# Secant Method for Continuous Functions (Single Variable)
def continuous_secant_method_value():
    raise NotImplementedError("This function is not implemented yet")

def continuous_secant_method_function():
    raise NotImplementedError("This function is not implemented yet")

def discrete_secant_method_value():
    raise NotImplementedError("This function is not implemented yet")

def discrete_secant_method_function():
    raise NotImplementedError("This function is not implemented yet")


# Newton's Method for Continuous Functions (Single Variable)
def continuous_newton_method_value():
    raise NotImplementedError("This function is not implemented yet")

def continuous_newton_method_function():
    raise NotImplementedError("This function is not implemented yet")

def discrete_newton_method_value():
    raise NotImplementedError("This function is not implemented yet")

def discrete_newton_method_function():
    raise NotImplementedError("This function is not implemented yet")










# Secant Method for Continuous Functions (Single Variable)
# Returns the value of x that is the root of the function
def secant_method_continuous_value(*, f: callable, x0: int | float, x1: int | float, error: int | float, max_iter: int) -> float:
    assert callable(f)
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert x0 != x1, "Guesses x0 and x1 cannot be the same"
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if np.abs(x2 - x1) < error:
            return x2
        x0, x1 = x1, x2
    else:
        return x2

# Secant Method for Continuous Functions (Single Variable)
# Returns the lists of x and f(x) that lead toward the root of the function
def secant_method_continuous_plot(*, f: callable, x0: int | float, x1: int | float, error: int | float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """
    blah blah blah
    """
    assert callable(f)
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert x0 != x1, "Guesses x0 and x1 cannot be the same"
    assert isinstance(error, (int, float))
    assert error > 0
    assert isinstance(max_iter, int)
    assert max_iter > 0
    x = [x0, x1]
    for _ in range(max_iter):
        x.append(x[-1] - f(x[-1]) * (x[-1] - x[-2]) / (f(x[-1]) - f(x[-2])))
        if np.abs(x[-1] - x[-2]) < error:
            return x, [f(i) for i in x]
    else:
        return x, [f(i) for i in x]

# Secant Method for Discrete Functions (Single Variable)
# Returns the value of x that is the root of the function
def secant_method_discrete_value(*, y: np.ndarray, x: np.ndarray, x0: int | float, x1: int | float, error: int | float, max_iter: int) -> float:
    pass

# Secant Method for Discrete Functions (Single Variable)
# Returns the lists of x and f(x) that lead toward the root of the function
def secant_method_discrete_plot(*, y: np.ndarray, x: np.ndarray, x0: int | float, x1: int | float, error: int | float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    pass



















# Fix all below








f = lambda x: x**3 - 2*x - 5
val = secant_method_continuous_value(f=f, x0=2, x1=3, error=1e-5, max_iter=5)
print(val)
print(f(val))
val = secant_method_continuous_plot(f=f, x0=2, x1=3, error=1e-5, max_iter=5)
print(val[0])
print(val[1])







# secant_method_continuous_value
# secant_method_continuous_plot
# secant_method_discrete_value
# secant_method_discrete_plot

# Secant Method Implementation
DEBUG = False
def secant_method_discrete(*, y: np.ndarray, x: np.ndarray, x0: int | float, x1: int | float, error: int | float, max_iter: int) -> float:
    
    # Forced assertions
    assert isinstance(y, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert len(y) == len(x), "Lists are not the same length"
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert isinstance(error, (int, float))
    assert isinstance(max_iter, int)
    assert np.any(y[:-1] * y[1:] < 0)
    assert x0 < x1

    # Find the root
    for _ in range(max_iter):
        if DEBUG: print(f"x0={x0}, x1={x1}")

        # Find the closest x values to x0 and linearly interpolate the value at x0
        xa, ya, xb, yb = find_two_closest_points(y=y, x=x, target_x=x0)
        y0 = linear_interpolation(xa=xa, ya=ya, xb=xb, yb=yb, target_x=x0)
        if DEBUG: print(f"Closests 2 points to x0: ({xa}, {ya}), ({xb}, {yb})")
        if DEBUG: print(f"Linear Approximation: ({x0}, {y0})")

        # Find the closest x values to x1 and linearly interpolate the value at x1
        xa, ya, xb, yb = find_two_closest_points(y=y, x=x, target_x=x1)
        y1 = linear_interpolation(xa=xa, ya=ya, xb=xb, yb=yb, target_x=x1)
        if DEBUG: print(f"Closests 2 points to x1: ({xa}, {ya}), ({xb}, {yb})")
        if DEBUG: print(f"Linear Approximation: ({x1}, {y1})")

        # Find the next x value in the secant line
        x2 = x1 - y1 * (x1 - x0) / (y1 - y0)
        if np.abs(x2 - x1) < error:
            print(f"Converged after {_} iterations)")
            return x2
        x0, x1 = x1, x2
        if DEBUG: print("\n")

    return x2































# Bisection Method Implementation
DEBUG = False
def bisection_method_discrete(*, y: np.ndarray, x: np.ndarray, lowerbound: int | float, upperbound: int | float, error: int | float, max_iter: int) -> float:
    
    # Forced assertions
    assert isinstance(y, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert len(y) == len(x), "Lists are not the same length"

    assert isinstance(lowerbound, (int, float))
    assert lowerbound > np.min(x), "Lower bound is less than the minimum x value"
    assert isinstance(upperbound, (int, float))
    assert upperbound < np.max(x), "Upper bound is greater than the maximum x value"
    
    assert isinstance(error, (int, float))
    assert isinstance(max_iter, int)
    assert np.any(y[:-1] * y[1:] < 0)
    assert lowerbound < upperbound

    # Find the root
    for _ in range(max_iter):

        # Find the midpoint
        midpoint = (lowerbound + upperbound) / 2
        if DEBUG: print(f"a={lowerbound}, b={upperbound=}")
        if DEBUG: print(f"{midpoint=}")

        # Find the closest x value to the midpoint
        closest_x_index_1 = np.argmin(np.abs(x - midpoint))
        x1 = x[closest_x_index_1]
        y1 = y[closest_x_index_1]
        if DEBUG: print(f"x1_index: {closest_x_index_1}")
        if DEBUG: print(f"x1_point: ({float(x[closest_x_index_1])}, {float(y[closest_x_index_1])})")

        # Find the next closest x value to the midpoint
        # Check +1 and -1 and see which one is closer to the midpoint
        val1 = np.abs(x[closest_x_index_1 + 1] - midpoint)
        val2 = np.abs(x[closest_x_index_1 - 1] - midpoint)
        if val1 < val2: closest_x_index_2 = closest_x_index_1 + 1
        if val1 > val2: closest_x_index_2 = closest_x_index_1 - 1
        x2 = x[closest_x_index_2]
        y2 = y[closest_x_index_2]
        if DEBUG: print(f"x2_index: {closest_x_index_2}")
        if DEBUG: print(f"x2_point: ({float(x[closest_x_index_2])}, {float(y[closest_x_index_2])})")

        # Linearly interpolate the two points at to find the value at the midpoint
        slope = (y2 - y1) / (x2 - x1)
        if DEBUG: print(f"slope: {slope}")
        midpoint_value = slope * (midpoint - x1) + y1
        if DEBUG: print(f"midpoint: ({float(midpoint)}, {float(midpoint_value)})")

        # Check if the midpoint is the root
        if np.abs(midpoint_value) < error:
            print(f"Converged after {_} iterations)")
            return midpoint
        elif midpoint_value < 0:
            lowerbound = midpoint
        elif midpoint_value > 0:
            upperbound = midpoint
        if DEBUG: print("\n")

    return midpoint







def bisection_method(f: callable, a: int | float, b: int | float, *, error: int | float, max_iter: int) -> float:
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert isinstance(error, (int, float))
    assert isinstance(max_iter, int)
    assert a < b
    assert f(a) * f(b) < 0
    for _ in range(max_iter):
        c = (a + b) / 2
        if np.abs(f(c)) < error:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c

# val = bisection_method(lambda x: x**3 - 2*x - 5, a=2, b=3, error=1e-5, max_iter=1_000_000)
# print(val)


def newton_method(*, f: callable, f_prime: callable, x0: int | float, error: int | float, max_iter: int) -> float:
    assert callable(f)
    assert callable(f_prime)
    assert isinstance(x0, (int, float))
    assert isinstance(error, (int, float))
    assert isinstance(max_iter, int)
    assert error > 0
    assert max_iter > 0
    for _ in range(max_iter):
        x1 = x0 - f(x0) / f_prime(x0)
        if np.abs(x1 - x0) < error:
            return x1
        x0 = x1
    return x1

# val = newton_method(f=lambda x: x**3 - 2*x - 5, f_prime=lambda x: 3*x**2 - 2, x0=2, error=1e-5, max_iter=1_000_000)
# print(val)



