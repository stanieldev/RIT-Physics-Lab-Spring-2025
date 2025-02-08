import numpy as np


# Value Summation Functions
def eulers_method(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> float:
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    t = np.arange(t0, tf+dt, dt)
    x = x0
    for i in range(len(t)-1):
        x = x + f(t[i], x)*dt
    return x

def midpoint_method(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> float:
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    t = np.arange(t0, tf+dt, dt)
    x = x0
    for i in range(len(t)-1):
        x = x + f(t[i] + dt/2, x + f(t[i], x)*dt/2)*dt
    return x

def runge_kutta_method(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> float:
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    t = np.arange(t0, tf+dt, dt)
    x = x0
    for i in range(len(t)-1):
        k1 = f(t[i], x)
        k2 = f(t[i] + dt/2, x + k1*dt/2)
        k3 = f(t[i] + dt/2, x + k2*dt/2)
        k4 = f(t[i] + dt, x + k3*dt)
        x = x + (k1 + 2*k2 + 2*k3 + k4)*dt/6
    return x


# Integrating Functions
def eulers_method_function(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> tuple[np.ndarray, np.ndarray]:
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    t = np.arange(t0, tf+dt, dt)
    x = [x0]
    for i in range(len(t)-1):
        x.append(x[-1] + f(t[i], x[-1])*dt)
    return t, x

def midpoint_method_function(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> tuple[np.ndarray, np.ndarray]:
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    t = np.arange(t0, tf+dt, dt)
    x = [x0]
    for i in range(len(t)-1):
        x.append(x[-1] + f(t[i] + dt/2, x[-1] + f(t[i], x[-1])*dt/2)*dt)
    return t, x

def runge_kutta_method_function(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> tuple[np.ndarray, np.ndarray]:
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    t = np.arange(t0, tf+dt, dt)
    x = [x0]
    for i in range(len(t)-1):
        k1 = f(t[i], x[-1])
        k2 = f(t[i] + dt/2, x[-1] + k1*dt/2)
        k3 = f(t[i] + dt/2, x[-1] + k2*dt/2)
        k4 = f(t[i] + dt, x[-1] + k3*dt)
        x.append(x[-1] + (k1 + 2*k2 + 2*k3 + k4)*dt/6)
    return t, x
