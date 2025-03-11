import numpy as np


# Runge-Kutta Method (Single Variable, Vectorized)
def continuous_runge_kutta_method_value(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> float:
    """
    Runge-Kutta Method for solving ordinary differential equations.
    This function solves the ODE using the Runge-Kutta Method and returns the final value.
    
    Parameters:
        f: callable - The function equal to the derivative of the unknown function
        t0: float | int - The initial time
        x0: float | int - The initial value(s)
        dt: float | int - The time step
        tf: float | int - The final time
    
    Returns:
        float - The value of the unknown function at the final time
    """
    
    # Input Checking
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))

    # Implementation
    T = np.arange(t0, tf+dt, dt)
    x = x0
    for t in T[:-1]:
        k1 = f(t       , x          )
        k2 = f(t + dt/2, x + k1*dt/2)
        k3 = f(t + dt/2, x + k2*dt/2)
        k4 = f(t + dt  , x + k3*dt  )
        x += (k1 + 2*k2 + 2*k3 + k4)*dt/6
    return x

def continuous_runge_kutta_method_function(
    f: callable,
    t0: float | int,
    x0: float | int | np.ndarray | tuple | list,
    dt: float | int,
    tf: float | int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runge-Kutta Method for solving ordinary differential equations.
    This function solves the ODE using the Runge-Kutta Method and returns the time and value arrays.
    
    Parameters:
        f: callable - The function equal to the derivative of the unknown function
        t0: float | int - The initial time
        x0: float | int - The initial value(s)
        dt: float | int - The time step
        tf: float | int - The final time

    Returns:
        tuple[np.ndarray, np.ndarray] - The time and value arrays
    """

    # Input Checking
    assert callable(f)
    assert isinstance(t0, (int, float))
    assert isinstance(x0, (int, float, np.ndarray, tuple, list))
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))

    # Implementation
    T = np.arange(t0, tf+dt, dt)
    x = [x0]
    for t in T[:-1]:
        k1 = f(t       , x[-1]          )
        k2 = f(t + dt/2, x[-1] + k1*dt/2)
        k3 = f(t + dt/2, x[-1] + k2*dt/2)
        k4 = f(t + dt  , x[-1] + k3*dt  )
        x.append(x[-1] + (k1 + 2*k2 + 2*k3 + k4)*dt/6)
    return T, x

def discrete_runge_kutta_method_value():
    raise NotImplementedError("This function is not yet implemented")

def discrete_runge_kutta_method_function():
    raise NotImplementedError("This function is not yet implemented")
