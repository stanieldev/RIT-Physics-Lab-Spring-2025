import numpy as np


# Class for Initial Conditions
class InitialCondition:
    def __init__(self, t0: float|int, x0: float|int|tuple|list|np.ndarray):
        assert isinstance(t0, (int, float))
        assert isinstance(x0, (int, float, tuple, list, np.ndarray))
        if isinstance(x0, (int, float, tuple, list)): x0 = np.array(x0)
        self.t0 = t0  # Initial Independent Variable
        self.x0 = x0  # Initial Dependent Variable


# Euler's Method (Single Differential Parameter, Vectorized)
def eulers_method_value(
    functional: callable,
    initial_condition: InitialCondition, *,
    dt: float | int,
    tf: float | int
) -> float:
    """
    Euler's Method for solving ordinary differential equations.
    This function solves the ODE using Euler's Method and returns the final value.
    Solve the following ODE: `dx/dt = f(t, x)` with the initial condition `x(t0) = x0`.

    Parameters:
        f: callable - The function equal to the derivative of the unknown function
        initial_condition: InitialCondition - The initial condition of the unknown function
        dt: float | int - The time step
        tf: float | int - The final time
        
    Returns:
        float - The value of the unknown function at the final time
    """

    # Input Checking
    assert callable(functional)
    assert isinstance(initial_condition, InitialCondition)
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    assert dt > 0, "dt must be greater than 0"
    assert abs(tf - initial_condition.t0) > dt, "dt is too large for the given time frame"

    # Determine if loop is done backwards or forwards
    if initial_condition.t0 < tf: 
        time = np.arange(initial_condition.t0, tf, dt)
    else:
        time = np.arange(tf, initial_condition.t0, dt)[::-1]
    
    # Implementation
    x = initial_condition.x0
    for t in time: x += functional(t, x)*dt
    return x


def eulers_method_function(
    functional: callable,
    initial_condition: InitialCondition, *,
    dt: float | int,
    tf: float | int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euler's Method for solving ordinary differential equations.
    This function solves the ODE using Euler's Method and returns the final value.
    Solve the following ODE: `dx/dt = f(t, x)` with the initial condition `x(t0) = x0`.

    Parameters:
        f: callable - The function equal to the derivative of the unknown function
        initial_condition: InitialCondition - The initial condition of the unknown function
        dt: float | int - The time step
        tf: float | int - The final time
        
    Returns:
        tuple[np.ndarray, np.ndarray] - The time and value arrays
    """

    # Input Checking
    assert callable(functional)
    assert isinstance(initial_condition, InitialCondition)
    assert isinstance(dt, (int, float))
    assert isinstance(tf, (int, float))
    assert dt > 0, "dt must be greater than 0"
    assert abs(tf - initial_condition.t0) > dt, "dt is too large for the given time frame"

    # Determine if loop is done backwards or forwards
    if initial_condition.t0 < tf: 
        time = np.arange(initial_condition.t0, tf+dt, dt)
    else:
        time = np.arange(tf, initial_condition.t0+dt, dt)[::-1]
    
    # Implementation
    x = [initial_condition.x0]
    for t in time[:-1]: x.append(x[-1] + functional(t, x[-1])*dt)
    return time, x


def discrete_eulers_method_value():
    raise NotImplementedError("This function is not yet implemented")


def discrete_eulers_method_function():
    raise NotImplementedError("This function is not yet implemented")
