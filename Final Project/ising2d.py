import numpy as np
import matplotlib.pyplot as plt
from interaction import interact_2D
from vector3 import Vector3

DEBUG = True  # Set to True to enable debug mode













class IsingModel2D:

    # Initialize the Ising model with a grid of spins
    def __init__(self, *, grid_size: int, 
                 interaction_strength: int | float, 
                 interaction_radius:   int | float, 
                 interaction_falloff:  int | float,
                 magnetic_field: Vector3 = Vector3(0, 0, 0)
        ) -> None:
        # Input validation
        assert isinstance(grid_size, int) and grid_size > 0,                                "Grid size must be a positive integer."
        assert isinstance(interaction_strength, (int, float)) and interaction_strength > 0, "Interaction strength must be a positive float."
        assert isinstance(interaction_radius,   (int, float)) and interaction_radius  >= 0, "Interaction radius must be a positive float."
        assert isinstance(interaction_falloff,  (int, float)) and interaction_falloff >= 0, "Interaction falloff must be a positive float."
        assert isinstance(magnetic_field, Vector3),                                         "Magnetic field must be a Vector3 object."

        # Class variables
        self.grid_size:              int = grid_size             # Size of the grid (number of spins in each dimension)
        self.interaction_strength: float = interaction_strength  # Interaction strength (coupling constant $J$)
        self.interaction_radius:   float = interaction_radius    # Interaction radius (0 = nearest neighbor, 1 = next nearest neighbor, etc.)
        self.interaction_falloff:  float = interaction_falloff   # Interaction falloff exponent (0 = constant, 1 -> 1/x, 2 = 1/x², etc.)
        self.magnetic_field:     Vector3 = magnetic_field        # Magnetic field vector (external field $h$)

        # Initialize spin node grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=Vector3)
        self._initialize_grid(fill=Vector3(1,0,0))

        # Precompute interaction parameters
        self.interaction_offsets: list[tuple[int, int]] = []  # List of valid offsets for interaction (Δx, Δy)
        self.interaction_coefficients:      list[float] = []  # List of coefficients for interaction (weights for each offset)
        self.interaction_global: bool = False                 # Flag for global interaction
        self._initialize_interaction()                        # (True if interaction radius is large enough to cover the entire grid)


    # Initialize the grid with random or uniform vectors
    def _initialize_grid(self, *, fill: Vector3 | None = None) -> None:
        if fill is None:
            # Make a 2D grid of random vectors
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.grid[i, j] = Vector3(np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)).normalized
        else:
            # Make a 2D grid of vectors with the same direction as fill
            assert isinstance(fill, Vector3), "Fill must be a Vector3 object."
            fill = fill.normalized  # Normalize the fill vector
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.grid[i, j] = fill

    # Initialize the interaction parameters
    def _initialize_interaction(self) -> None:
        
        # Interaction radius is large enough to (effectively) cover the entire grid
        if self.interaction_radius > self.grid_size * np.sqrt(2):
            self.interaction_radius = self.grid_size * np.sqrt(2)
            self.interaction_global = True
        if DEBUG: print(f"Interaction radius: {self.interaction_radius}, Global interaction: {self.interaction_global}")

        # Pre-calculate all offset tuples (Δx, Δy) within interaction radius (excluding self-interaction)
        INTERACTION_RADIUS_SQUARED = self.interaction_radius**2
        for i in range(int(self.interaction_radius)+1):
            for j in range(int(self.interaction_radius)+1):
                if i == 0 and j == 0: continue                           # Self-interaction
                if (i**2 + j**2) > INTERACTION_RADIUS_SQUARED: continue  # Interactions outside the interaction radius
                self.interaction_offsets.append((i, j))
        self.interaction_offsets = np.array(self.interaction_offsets, dtype=tuple)
        if DEBUG: print(f"Interaction offsets: {self.interaction_offsets}")

        # Pre-calculate the interaction coefficients for each offset tuple (Δx, Δy)
        self.interaction_coefficients = []
        for Δx, Δy in self.interaction_offsets:
            coupling_coefficient = self.interaction_strength / pow(np.sqrt(Δx**2 + Δy**2), self.interaction_falloff)
            self.interaction_coefficients.append(coupling_coefficient)
        self.interaction_coefficients = np.array(self.interaction_coefficients, dtype=float)
        if DEBUG: print(f"Interaction coefficients: {self.interaction_coefficients}")


    # Total energy of the grid
    def total_energy(self):
        ENERGY: float = 0.0

        # Calculate the interaction energy for each pair of spins in the grid
        for (Δx, Δy), coeff in zip(self.interaction_offsets, self.interaction_coefficients):
            ENERGY += -coeff * interact_2D(self.grid, Δx, Δy)

        # Calculate the magnetic field energy
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ENERGY += self.magnetic_field * self.grid[i, j]

        # Return the total energy
        return ENERGY
    
    # Total magnetization of the grid
    def magnetization(self):
        MAGNETIZATION = np.sum(self.grid)
        return MAGNETIZATION / self.grid.size


    # Energy of a single point in the grid
    # The sum of all points should be equal to twice the total energy of the grid (double counting)
    def point_energy(self, i: int, j: int) -> float:
        ENERGY: float = 0.0
        INTERACTION_RADIUS_SQUARED = self.interaction_radius**2

        # Calculate the interaction energy of a single point in the grid
        for Δx in range(-self.interaction_radius, self.interaction_radius+1):
            for Δy in range(-self.interaction_radius, self.interaction_radius+1):

                # Skip self-interaction and interactions outside the interaction radius
                if Δx == 0 and Δy == 0: continue                           # Self-interaction
                if (Δx**2 + Δy**2) > INTERACTION_RADIUS_SQUARED: continue  # Interactions outside the interaction radius

                # Add the energies of the neighbors
                coupling_coefficient = self.interaction_strength / pow(np.sqrt(i**2 + j**2), self.interaction_falloff)
                interaction = self.grid[i, j] * self.grid[(i + Δx), (j + Δy)] if (0 <= i + Δx < self.grid_size and 0 <= j + Δy < self.grid_size) else 0

                # Calculate the interaction energy
                ENERGY += -coupling_coefficient * interaction

        # Calculate the magnetic field energy
        ENERGY += self.magnetic_field * self.grid[i, j]

        # Return the total energy
        return ENERGY

    # Mean spin of a single point in the grid excluding the spin itself
    def point_mean_spin(self, i: int, j: int) -> Vector3:
        SPIN = Vector3(0, 0, 0)
        INTERACTION_RADIUS_SQUARED = self.interaction_radius**2

        # Calculate the mean spin of a single point in the grid
        for Δx in range(-self.interaction_radius, self.interaction_radius+1):
            for Δy in range(-self.interaction_radius, self.interaction_radius+1):

                # Skip self-interaction and interactions outside the interaction radius
                if Δx == 0 and Δy == 0: continue                           # Self-interaction
                if (Δx**2 + Δy**2) > INTERACTION_RADIUS_SQUARED: continue  # Interactions outside the interaction radius

                # Add the spins of the neighbors weighted by coupling coefficient
                coupling_coefficient = 1 / pow(np.sqrt(i**2 + j**2), self.interaction_falloff)
                try:               magnetization = self.grid[(i + Δx), (j + Δy)]
                except IndexError: magnetization = Vector3(0, 0, 0)

                # Fix for later:
                magnetization.x *= coupling_coefficient
                magnetization.y *= coupling_coefficient
                magnetization.z *= coupling_coefficient
                SPIN += magnetization

        # Return the mean spin
        return SPIN.normalized

    # Iteration function (Metropolis algorithm)
    def iterate(self, beta: float) -> None:

        # How big does the interaction need to be before we just calculate the whole grid?
        # TODO: Make this a ratio of areas weighted by the interaction power for criterion
        INTERACTION_THRESHOLD = 0.1  # 10% of the grid size
        CALCULATE_WHOLE_GRID = (self.interaction_radius < np.sqrt(2 * (INTERACTION_THRESHOLD * self.grid_size) ** 2))

        # Pick a random point in the grid
        i, j = np.random.randint(0, self.grid.shape[0]), np.random.randint(0, self.grid.shape[1])

        # Pre-Flip State
        current_magnetization = self.grid[i, j]
        if CALCULATE_WHOLE_GRID: current_energy = self.point_energy(i, j)
        else:                    current_energy = self.total_energy()

        # Find a new magnetization vector (randomly)
        # TODO: Make the sigma based on the vectors in the grid
        SIGMA = 0.1
        new_magnetization = self.point_mean_spin(i, j)
        new_magnetization.x += np.random.normal(0, SIGMA)
        new_magnetization.y += np.random.normal(0, SIGMA)
        new_magnetization.z += np.random.normal(0, SIGMA)
        new_magnetization.normalize()

        # Post-Flip Energy
        self.grid[i, j] = new_magnetization
        if CALCULATE_WHOLE_GRID: new_energy = self.point_energy(i, j)
        else:                    new_energy = self.total_energy()

        # Calculate the metropolis acceptance probability
        ΔE = new_energy - current_energy
        p_accept = min(1, np.exp(-ΔE * beta))

        # Accept or reject the flip based on the acceptance probability
        if np.random.rand() < p_accept:
            ΔM = new_magnetization - current_magnetization
            return ΔE, ΔM
        else:
            self.grid[i, j] = current_magnetization  # Revert to the original state
            return 0, Vector3(0, 0, 0)


x1 = IsingModel2D(grid_size=10, 
                  interaction_strength=1.0, 
                  interaction_radius=100, 
                  interaction_falloff=0.0,
                  magnetic_field=Vector3(0, 0, 0)
                  )

print(x1.total_energy())







# Draw a 3D plot of the grid with a 2D plane of spins represented as arrows
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Each spin in the 2D grid, draw a point at its position and an arrow in the direction of the spin
for i in range(x1.grid_size):
    for j in range(x1.grid_size):
        # Get the spin vector
        spin = x1.grid[i, j]

        # Draw a point at the position of the spin
        ax.scatter(i, j, 0, color='b', s=100)

        # Draw an arrow in the direction of the spin
        ax.quiver(i, j, 0, spin.x, spin.y, spin.z, length=0.5, normalize=True)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Spin Configuration')

# Set the limits of the axes
ax.set_xlim(0, x1.grid_size)
ax.set_ylim(0, x1.grid_size)
ax.set_zlim(-1, 1)

plt.show()








# # Program
# STEPS = 25_000  # Number of steps to iterate
# T = 1.0  # Temperature (arbitrary value for this example)
# BETA = 1 / T  # 1/kT


# model = IsingModel(grid_size=10, interaction_strength=1.0, magnetic_field=Vector3(1, 0, 0), interaction_radius=3, initialize_random=True)
# model_energy = [model.total_energy()]
# model_magnetization = [model.magnetization()]
# for i in range(STEPS):
#     ΔE, ΔM = model.iterate(beta=BETA)
#     model_energy.append(model_energy[-1] + ΔE)
#     model_magnetization.append(model_magnetization[-1] + ΔM)
#     print(f"Step {i+1}/{STEPS}: Energy = {model_energy[-1]}, Magnetization = {model.magnetization()}")


# # Plot the results with 2 axes
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# # Energy plot
# ax1.plot(model_energy, 'g-')
# ax1.set_ylabel('Energy', color='g')
# ax1.tick_params(axis='y', labelcolor='g')

# # Magnetization plot
# ax2.plot([i.norm for i in model_magnetization], 'b-')
# ax2.set_ylabel('Normalized Magnetization', color='b')
# ax2.tick_params(axis='y', labelcolor='b')
# # ax2.set_ylim(-1.1, 1.1)

# # Final plot settings
# plt.xticks(np.arange(0, len(model_energy), step=int(STEPS/10)), rotation=45)
# plt.title('Energy and Magnetization over Time')
# plt.xlabel('Iteration')
# plt.legend()
# plt.show()