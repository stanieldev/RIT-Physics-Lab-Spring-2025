import numpy as np


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

val = bisection_method(lambda x: x**3 - 2*x - 5, a=2, b=3, error=1e-5, max_iter=1_000_000)
print(val)


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

val = newton_method(f=lambda x: x**3 - 2*x - 5, f_prime=lambda x: 3*x**2 - 2, x0=2, error=1e-5, max_iter=1_000_000)
print(val)


def secant_method(*, f: callable, x0: int | float, x1: int | float, error: int | float, max_iter: int) -> float:
    assert callable(f)
    assert isinstance(x0, (int, float))
    assert isinstance(x1, (int, float))
    assert isinstance(error, (int, float))
    assert isinstance(max_iter, int)
    assert error > 0
    assert max_iter > 0
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if np.abs(x2 - x1) < error:
            return x2
        x0, x1 = x1, x2
    return x2

val = secant_method(f=lambda x: x**3 - 2*x - 5, x0=2, x1=3, error=1e-5, max_iter=1_000_000)
print(val)
