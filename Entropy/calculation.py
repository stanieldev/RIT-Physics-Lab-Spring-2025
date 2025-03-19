# Make a trial wavefunction:
import numpy as np


# Functions I made for this problem
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


# Norm of a function
def norm(f: np.ndarray, X: np.ndarray) -> float:
    return np.sqrt(np.trapezoid(f**2, X))

# Entropy of a function
def entropy(psi: np.ndarray, X: np.ndarray) -> float:
    return -np.trapezoid(abs(psi)**2 * np.log(abs(psi)**2), X)


# Problem Parameters
Δx = 2
x0: float = -Δx
Hamiltonian = lambda E, x, psi: np.array([psi[1], (x**2-2*E)*psi[0]])
psi0 = np.array([1, 0])


ENERGIES = np.linspace(0, 10, 1000)
entropies = []
for _ in ENERGIES:
    # Solve the Schrodinger Equation
    wavefunction = lambda E: continuous_runge_kutta_method_function(
        f=lambda x, psi: Hamiltonian(_, x, psi),
        t0=0, x0=psi0, dt=0.001, tf=Δx
    )

    # Normalize the wavefunction
    solution = wavefunction(ENERGIES)
    position = np.append(-solution[0][::-1], solution[0])
    psi = np.append(np.array([v[0] for v in solution[1]])[::-1], np.array([v[0] for v in solution[1]]))
    psi /= norm(psi, position)

    # Calculate the entropy
    entropies.append(entropy(psi, position))




# Plot  
import matplotlib.pyplot as plt

# plt.plot(ENERGIES, entropies, label="S")
# plt.plot(ENERGIES, np.gradient(entropies, ENERGIES), label="dS/dE")
plt.plot(ENERGIES, np.gradient(np.gradient(entropies, ENERGIES), ENERGIES), label="d^2S/dE^2")
plt.xlabel("Energy")
plt.ylabel("Entropy")
plt.title("Entropy vs Energy")

true_energies = [float((0.5+n)*np.sqrt(2)) for n in range(10)]
[plt.axvline(_, color='black', linestyle='--') for _ in true_energies]
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.show()