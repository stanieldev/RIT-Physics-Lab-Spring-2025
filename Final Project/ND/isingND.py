import numpy as np
import matplotlib.pyplot as plt
from ND.grid import generate_grid
from ND.interaction import interaction_products, allowed_spherical_interaction_tuples, allowed_interaction_weighting
from itertools import product


DEBUG = False  # Set to True to enable debug mode





# class Vector3(np.ndarray):
#     def __new__(cls, x: float, y: float, z: float) -> 'Vector3':
#         obj = np.array([x, y, z], dtype=float).view(cls)
#         return obj
    
#     @property
#     def norm(self) -> float:
#         return np.linalg.norm(self)

#     @property
#     def normalized(self) -> 'Vector3':
#         norm = np.linalg.norm(self)
#         if norm == 0:
#             return self
#         return self / norm
    
#     def __repr__(self) -> str:
#         return f"Vector3({self[0]}, {self[1]}, {self[2]})"

#     def __str__(self) -> str:
#         return f"Vector3({self[0]}, {self[1]}, {self[2]})"



class IsingModel:
    def __init__(self, *,
                 grid_size: int,
                 dimension: int,
                 interaction_strength: int | float = 1.0,
                 interaction_radius:   int | float = 3.0,
                 interaction_falloff:  int | float = 0.0,
                 magnetic_field: np.ndarray = None
        ) -> None:

        # Setting defaults if not provided
        if magnetic_field is None: magnetic_field = np.zeros(dimension, dtype=float)

        # Input validation
        assert isinstance(grid_size, int) and grid_size > 0,                                "Grid size must be a positive integer."
        assert isinstance(dimension, int) and dimension > 0,                                "Dimension must be a positive integer."
        assert isinstance(interaction_strength, (int, float)) and interaction_strength > 0, "Interaction strength must be a positive float."
        assert isinstance(interaction_radius,   (int, float)) and interaction_radius  >= 0, "Interaction radius must be a positive float."
        assert isinstance(interaction_falloff,  (int, float)) and interaction_falloff >= 0, "Interaction falloff must be a positive float."
        assert isinstance(magnetic_field, np.ndarray),                                      "Magnetic field must be a numpy array."

        # Shape consistency checks
        assert magnetic_field.ndim == 1,             "Magnetic field must be a 1D array."
        assert magnetic_field.dtype == np.float64,   "Magnetic field must be a numpy array of floats."
        assert magnetic_field.shape == (dimension,), "Magnetic field must have the same dimension as the spin vectors."
        interaction_radius = np.clip(interaction_radius, 0, grid_size * np.sqrt(dimension))  # Don't allow interaction radius to exceed the grid size
        
        # Class variables
        self.grid_size:              int = grid_size             # Size of the grid (number of spins in each dimension)
        self.dimension:              int = dimension             # Dimension of the grid (1D, 2D, 3D, etc.)
        self.interaction_strength: float = interaction_strength  # Interaction strength (coupling constant $J$)
        self.interaction_radius:   float = interaction_radius    # Interaction radius (0 = nearest neighbor, 1 = next nearest neighbor, etc.)
        self.interaction_falloff:  float = interaction_falloff   # Interaction falloff exponent (0 = constant, 1 -> 1/x, 2 = 1/x², etc.)
        self.magnetic_field:  np.ndarray = magnetic_field        # Magnetic field vector (external field $h$)

        # Initialize spin node grid
        self.grid = generate_grid(size=self.grid_size, dimension=self.dimension, fill=np.array([1, 0]))  # fill=np.array([1, 0, 0])

        # Precompute interaction parameters
        self.interaction_offsets: np.ndarray = allowed_spherical_interaction_tuples(dimension=self.dimension, maximum_radius=self.interaction_radius)
        self.interaction_weights: np.ndarray = allowed_interaction_weighting(interaction_offsets=self.interaction_offsets, power=self.interaction_falloff)


    # Calculate the interaction energy for each pair of spins in the grid within interaction radii
    def total_energy(self) -> float:
        ENERGY: float = 0.0

        # Calculate the interaction energy for each pair of spins in the grid
        for offset, weight in zip(self.interaction_offsets, self.interaction_weights):
            ENERGY += -weight * interaction_products(grid=self.grid, offset=offset)

        # Calculate the magnetic field energy
        ENERGY += sum(self.magnetic_field * np.sum(self.grid, axis=tuple(range(self.dimension))))

        # Return the total energy
        return ENERGY

    # Calculate the magnetization of the grid
    def total_magnetization(self) -> float:
        MAGNETIZATION: float = np.sum(self.grid, axis=tuple(range(self.dimension)))
        return MAGNETIZATION / (self.grid_size ** self.dimension)
    

    # Calculate the point energy of a single spin in the grid
    # TODO THIS IS BROKEN
    def point_energy(self, index: tuple[int]) -> float:
        ENERGY: float = 0.0
        INTERACTION_RADIUS_SQUARED = self.interaction_radius**2

        # Calculate the interaction energy of a single point in the grid
        for indices in product(range(-int(self.interaction_radius), int(self.interaction_radius)+1), repeat=self.dimension):

            # Skip self-interaction and interactions outside the interaction radius
            if all(i == 0 for i in indices): continue
            if sum(i**2 for i in indices) > INTERACTION_RADIUS_SQUARED: continue

            # Add the energies of the neighbors
            coupling_coefficient = self.interaction_strength / pow(np.sqrt(sum(i**2 for i in indices)), self.interaction_falloff)
            interaction = self.grid[index] * self.grid[tuple(np.add(index, indices))] if all(0 <= i < self.grid_size for i in np.add(index, indices)) else 0

            # Calculate the interaction energy
            ENERGY += -coupling_coefficient * interaction

        # Calculate the magnetic field energy
        ENERGY += self.magnetic_field * self.grid[index]

        # Return the total energy
        return ENERGY

    # Calculate the mean spin at a single spin in the grid
    def point_magnetization(self, index: tuple[int]) -> float:
        SPIN = np.zeros(self.dimension, dtype=float)  # Initialize the spin vector
        INTERACTION_RADIUS_SQUARED = self.interaction_radius**2

        # Calculate the mean spin of a single point in the grid
        for indices in product(range(-int(self.interaction_radius), int(self.interaction_radius+1)), repeat=self.dimension):
                
            # Skip self-interaction and interactions outside the interaction radius
            if all(i == 0 for i in indices): continue
            if sum(i**2 for i in indices) > INTERACTION_RADIUS_SQUARED: continue

            # Add the spins of the neighbors weighted by coupling coefficient
            coupling_coefficient = 1 / pow(np.sqrt(sum(i**2 for i in indices)), self.interaction_falloff)
            try:               magnetization = self.grid[index] * self.grid[tuple(np.add(index, indices))]
            except IndexError: continue

        # Return the mean spin
        return SPIN / np.linalg.norm(SPIN) if np.linalg.norm(SPIN) != 0 else SPIN
        

    # Iterate the model for a single step
    def iterate(self, *, beta: float) -> tuple[float, float]:
        
        # Pick a random point in the grid
        random_index = np.random.randint(0, self.grid_size, size=self.dimension)

        # Pre-Flip State
        current_magnetization = self.grid[tuple(random_index)]
        current_energy = self.point_energy(random_index)

        # Find a new magnetization vector (randomly)
        p: float = 0.5
        new_magnetization = p * self.point_magnetization(random_index) + (1-p) * current_magnetization
        new_magnetization = new_magnetization / np.linalg.norm(new_magnetization)

        # Post-Flip State
        self.grid[tuple(random_index)] = new_magnetization  # Update the grid with the new magnetization
        new_energy = self.point_energy(random_index)  # Calculate the new energy of the grid

        # Calculate the metropolis acceptance probability
        ΔE = new_energy - current_energy
        print(f"{new_energy=}")
        print(f"{current_energy=}")
        p_accept = min(1, np.exp(-ΔE * beta))

        # Accept or reject the flip based on the acceptance probability
        if np.random.rand() < p_accept:
            ΔM = new_magnetization - current_magnetization
            return ΔE, ΔM
        else:
            self.grid[tuple(random_index)] = current_magnetization  # Revert to the original state
            return 0, np.zeros(self.dimension, dtype=float)         # No change in magnetization




model = IsingModel(grid_size=10, dimension=2,
                   interaction_strength=1.0,
                   interaction_radius=100,
                   interaction_falloff=0.0,
                   magnetic_field=np.array([0, 0], dtype=float)
)

print("Total Magnetization:", model.total_magnetization())
print("Total Energy:", model.total_energy())


BETA = 1.0  # Temperature (arbitrary value for this example)
STEPS = 25_000  # Number of steps to iterate
for _ in range(STEPS):
    ΔE, ΔM = model.iterate(beta=BETA)
    print(f"Step {_+1}/{STEPS}: Energy = {model.total_energy()}, Magnetization = {model.total_magnetization()}")




# # Draw a 3D plot of the grid with a 2D plane of spins represented as arrows
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Each spin in the 2D grid, draw a point at its position and an arrow in the direction of the spin
# for i in range(model.grid_size):
#     for j in range(model.grid_size):
#         for k in range(model.grid_size):
#             # Get the spin vector
#             spin = model.grid[i, j, k]

#             # Draw a point at the position of the spin (make it small)
#             ax.scatter(i, j, k, color='b', s=10)

#             # Draw an arrow in the direction of the spin
#             ax.quiver(i, j, k, spin[0], spin[1], spin[2], length=0.5, normalize=True)

# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.set_title('3D Spin Configuration')

# # Set the limits of the axes
# ax.set_xlim(-1, model.grid_size + 1)
# ax.set_ylim(-1, model.grid_size + 1)
# ax.set_zlim(-1, model.grid_size + 1)

# plt.show()








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