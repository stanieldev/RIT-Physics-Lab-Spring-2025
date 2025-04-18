import numpy as np
import matplotlib.pyplot as plt
from interaction import interact_2D
from vector3 import Vector3















class IsingModel:

    # Initialize the Ising model with a grid of spins
    def __init__(self, *, grid_size: int, 
                 magnetic_field: Vector3 = Vector3(0, 0, 0),
                 interaction_strength: int | float = 1.0, 
                 interaction_radius:   int | float = 100, 
                 interaction_falloff:  int | float = 0.0,
                 initialize_random: bool = False,
        ) -> None:
        # Input validation
        assert isinstance(grid_size, int) and grid_size > 0,                                "Grid size must be a positive integer."
        assert isinstance(magnetic_field, Vector3),                                         "Magnetic field must be a Vector3 object."
        assert isinstance(interaction_strength, (int, float)) and interaction_strength > 0, "Interaction strength must be a positive float."
        assert isinstance(interaction_radius, (int, float)) and interaction_radius >= 0,    "Interaction radius must be a positive float."
        assert isinstance(interaction_falloff, (int, float)) and interaction_falloff >= 0,  "Interaction falloff must be a positive float."

        # Class variables
        self.grid_size: int = grid_size
        self.magnetic_field: Vector3 = magnetic_field
        self.interaction_strength: float = interaction_strength
        self.interaction_radius:   float = interaction_radius
        self.interaction_falloff:  float = interaction_falloff

        # Preliminary Optimization
        if self.interaction_radius > int(np.sqrt(2) * grid_size):
            self.interaction_radius = int(np.sqrt(2) * grid_size)  # Reduces the number of interactions to be calculated
        
        # Initialize Node grid
        self.grid = np.zeros((grid_size, grid_size), dtype=Vector3)
        if initialize_random:  # Make a 2D grid of random vectors
            for i in range(grid_size):
                for j in range(grid_size):
                    self.grid[i, j] = Vector3(np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)).normalized
        else:  # Make a 2D grid of vectors with the same direction (1,0,0)
            for i in range(grid_size):
                for j in range(grid_size):
                    self.grid[i, j] = Vector3(1, 0, 0)

        # Calculate the offsets that interact given the interaction radius
        self.interaction_offsets = []
        for i in range(self.interaction_radius+1):
            for j in range(self.interaction_radius+1):
                if i == 0 and j == 0: continue                           # Self-interaction
                if (i**2 + j**2) > self.interaction_radius**2: continue  # Interactions outside the interaction radius
                self.interaction_offsets.append((i, j))

        # Calculate the interaction coefficients for the offsets
        self.interaction_coefficients = []
        for Δx, Δy in self.interaction_offsets:
            coupling_coefficient = self.interaction_strength / pow(np.sqrt(Δx**2 + Δy**2), self.interaction_falloff)
            self.interaction_coefficients.append(coupling_coefficient)
    

    # Total energy of the grid
    def total_energy(self):

        # Calculate the interaction energy of the grid
        ENERGY: float = 0.0

        # Calculate the interaction energy for each pair of spins in the grid
        for (Δx, Δy), coeff in zip(self.interaction_offsets, self.interaction_coefficients):
            ENERGY += -coeff * interact_2D(self.grid, Δx, Δy)

        # Calculate the magnetic field energy
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ENERGY += self.magnetic_field * self.grid[i, j]



        # for i in range(int(self.interaction_radius)+1):
        #     for j in range(int(self.interaction_radius)+1):

        #         # Skip self-interaction and interactions outside the interaction radius
        #         if i == 0 and j == 0: continue                           # Self-interaction
        #         if (i**2 + j**2) > self.interaction_radius**2: continue  # Interactions outside the interaction radius

        #         # Calculate the coupling coefficient and interactions
        #         coupling_coefficient = self.interaction_strength / pow(np.sqrt(i**2 + j**2), self.interaction_falloff)
        #         interactions = interact_2D(self.grid, i, j)

        #         # Calculate the interaction energy
        #         ENERGY += -coupling_coefficient * interactions

        

        # Calculate the total energy
        return ENERGY
    
    # Total magnetization of the grid
    def magnetization(self):
        return np.sum(self.grid) / self.grid.size


    # Energy of a single point in the grid
    # The sum of all points should be equal to twice the total energy of the grid (double counting)
    def point_energy(self, i: int, j: int) -> float:
        
        # Calculate the interaction energy of a single point in the grid
        ENERGY = 0.0
        for Δx in range(-self.interaction_radius, self.interaction_radius+1):
            for Δy in range(-self.interaction_radius, self.interaction_radius+1):

                # Skip self-interaction and interactions outside the interaction radius
                if Δx == 0 and Δy == 0: continue                           # Self-interaction
                if (Δx**2 + Δy**2) > self.interaction_radius**2: continue  # Interactions outside the interaction radius

                # Add the energies of the neighbors
                coupling_coefficient = self.interaction_strength / pow(np.sqrt(i**2 + j**2), self.interaction_falloff)
                interaction = self.grid[i, j] * self.grid[(i + Δx), (j + Δy)] if (0 <= i + Δx < self.grid_size and 0 <= j + Δy < self.grid_size) else 0

                # Calculate the interaction energy
                ENERGY += -coupling_coefficient * interaction

        # Calculate the magnetic field energy
        ENERGY += self.magnetic_field * self.grid[i, j]

        # Calculate the total energy
        return ENERGY

    # Unused, use later
    def point_mean_spin(self, i: int, j: int) -> Vector3:
        # Calculate the mean spin of a single point in the grid
        SPIN = Vector3(0, 0, 0)
        for Δx in range(-self.interaction_radius, self.interaction_radius+1):
            for Δy in range(-self.interaction_radius, self.interaction_radius+1):

                # Skip self-interaction and interactions outside the interaction radius
                if Δx == 0 and Δy == 0: continue                           # Self-interaction
                if (Δx**2 + Δy**2) > self.interaction_radius**2: continue  # Interactions outside the interaction radius

                # Add the spins of the neighbors
                coupling_coefficient = 1 / pow(np.sqrt(i**2 + j**2), self.interaction_falloff)
                try:
                    magnetization = self.grid[(i + Δx), (j + Δy)]
                except IndexError:
                    magnetization = Vector3(0, 0, 0)

                # Fix for later:
                magnetization.x *= coupling_coefficient
                magnetization.y *= coupling_coefficient
                magnetization.z *= coupling_coefficient
                SPIN += magnetization

        return SPIN.normalized


    # Iteration function (Metropolis algorithm)
    def iterate(self, beta: float) -> None:
        INTERACTION_THRESHOLD = 0.1  # How big does the interaction need to be before we just calculate the whole grid?
        CALCULATE_WHOLE_GRID = (self.interaction_radius < np.sqrt(2 * (INTERACTION_THRESHOLD * self.grid_size) ** 2))

        # Pick a random point in the grid
        i = np.random.randint(0, self.grid.shape[0])
        j = np.random.randint(0, self.grid.shape[1])

        # Pre-Flip State
        current_magnetization = self.grid[i, j]
        if CALCULATE_WHOLE_GRID: current_energy = self.point_energy(i, j)
        else:                    current_energy = self.total_energy()

        # Find a new magnetization vector (randomly)
        SIGMA = 0.1
        new_magnetization = self.point_mean_spin(i, j)
        new_magnetization.x += np.random.normal(0, SIGMA)
        new_magnetization.y += np.random.normal(0, SIGMA)
        new_magnetization.z += np.random.normal(0, SIGMA)

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





# Program
STEPS = 25_000  # Number of steps to iterate
T = 1.0  # Temperature (arbitrary value for this example)
BETA = 1 / T  # 1/kT


model = IsingModel(grid_size=10, interaction_strength=1.0, magnetic_field=Vector3(1, 0, 0), interaction_radius=3, initialize_random=True)
model_energy = [model.total_energy()]
model_magnetization = [model.magnetization()]
for i in range(STEPS):
    ΔE, ΔM = model.iterate(beta=BETA)
    model_energy.append(model_energy[-1] + ΔE)
    model_magnetization.append(model_magnetization[-1] + ΔM)
    print(f"Step {i+1}/{STEPS}: Energy = {model_energy[-1]}, Magnetization = {model.magnetization()}")


# Plot the results with 2 axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Energy plot
ax1.plot(model_energy, 'g-')
ax1.set_ylabel('Energy', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Magnetization plot
ax2.plot([i.norm for i in model_magnetization], 'b-')
ax2.set_ylabel('Normalized Magnetization', color='b')
ax2.tick_params(axis='y', labelcolor='b')
# ax2.set_ylim(-1.1, 1.1)

# Final plot settings
plt.xticks(np.arange(0, len(model_energy), step=int(STEPS/10)), rotation=45)
plt.title('Energy and Magnetization over Time')
plt.xlabel('Iteration')
plt.legend()
plt.show()