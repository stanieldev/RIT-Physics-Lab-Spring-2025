import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from extension.grid import SpinGrid
from extension.interaction import precompute_interaction_tuples, precompute_interaction_weights, interaction_products
from matplotlib.animation import FuncAnimation, PillowWriter


class MagneticField:
    def __init__(self, vector: np.ndarray) -> None:
        # Input validation
        assert isinstance(vector, np.ndarray), "Magnetic field must be a numpy array."
        assert vector.ndim == 1,               "Magnetic field must be a 1D array."
        assert vector.dtype == np.float64,     "Magnetic field must be a numpy array of floats."

        # Assigning values to class variables
        self.vector = vector
        self.dimension = vector.shape[0]


class SpinInteraction:
    def __init__(self, strength: float, radius: float, falloff: float) -> None:
        
        # Input validation
        assert isinstance(strength, (int, float)) and strength > 0, "Interaction strength must be a positive float."
        assert isinstance(radius,   (int, float)) and radius  >= 0, "Interaction radius must be a positive float."
        assert isinstance(falloff,  (int, float)) and falloff >= 0, "Interaction falloff must be a positive float."

        # Assigning values to class variables
        self.strength = strength
        self.radius = radius
        self.falloff = falloff


class IsingModel:
    def __init__(self, *,
                 spin_grid: SpinGrid,
                 interaction: SpinInteraction,
                 magnetic_field: MagneticField = None
        ) -> None:
        # Input validation
        assert isinstance(spin_grid, SpinGrid),                    "Grid must be a SpinGrid object."
        assert isinstance(interaction, SpinInteraction),           "Interaction must be an SpinInteraction object."
        assert isinstance(magnetic_field, (MagneticField, None)),  "Magnetic field must be a MagneticField object or None."

        # Additional constraints for the model
        assert spin_grid.spin_dimension == magnetic_field.dimension, "Magnetic field dimension must match the spin's dimension."
        interaction.radius = np.clip(interaction.radius, 1, sum([i**2 for i in spin_grid.grid_shape]) ** 0.5)

        # Assigning values to class variables
        self.spin_grid: SpinGrid = spin_grid
        self.interaction: SpinInteraction = interaction
        self.magnetic_field: MagneticField = magnetic_field if magnetic_field is not None else MagneticField(np.zeros(spin_grid.spin_dimension, dtype=float))

        # Precompute interaction parameters
        self.interaction_offsets: np.ndarray = precompute_interaction_tuples(dimension=self.spin_grid.grid_dimension, maximum_radius=self.interaction.radius)
        self.interaction_weights: np.ndarray = precompute_interaction_weights(interaction_offsets=self.interaction_offsets, power=self.interaction.falloff)


    # Calculate the interaction energy for each pair of spins in the grid within interaction radii
    def total_energy(self) -> float:
        ENERGY: float = 0.0

        # Calculate the interaction energy for each pair of spins in the grid
        for offset, weight in zip(self.interaction_offsets, self.interaction_weights):
            ENERGY += -weight * interaction_products(grid=self.spin_grid.grid, offset=offset)

        # Calculate the magnetic field energy
        ENERGY += sum(self.magnetic_field.vector * np.sum(self.spin_grid.grid, axis=tuple(range(self.spin_grid.spin_dimension))))

        # Return the total energy
        return ENERGY

    # Calculate the magnetization of the grid
    def total_magnetization(self) -> float:
        MAGNETIZATION: float = np.sum(self.spin_grid.grid, axis=tuple(range(self.spin_grid.spin_dimension)))
        return MAGNETIZATION / self.spin_grid.grid_volume
    

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
        SPIN = np.zeros(self.spin_grid.spin_dimension, dtype=float)  # Initialize the spin vector
        INTERACTION_RADIUS_SQUARED = self.interaction.radius**2

        # Calculate the mean spin of a single point in the grid
        for indices in product(range(-int(self.interaction.radius), int(self.interaction.radius+1)), repeat=len(self.spin_grid.grid_shape)):
                
            # Skip self-interaction and interactions outside the interaction radius
            if all(i == 0 for i in indices): continue
            if sum(i**2 for i in indices) > INTERACTION_RADIUS_SQUARED: continue

            # Add the spins of the neighbors weighted by coupling coefficient
            coupling_coefficient = 1 / pow(np.sqrt(sum(i**2 for i in indices)), self.interaction.falloff)
            try: SPIN += coupling_coefficient * self.spin_grid.grid[index] * self.spin_grid.grid[tuple(np.add(index, indices))]
            except IndexError: continue

        # Return the mean spin
        return SPIN / np.linalg.norm(SPIN) if np.linalg.norm(SPIN) != 0 else SPIN
        

    # Iterate the model for a single step
    def iterate(self, *, beta: float) -> tuple[float, float]:
        
        # Pick a random point in the grid
        random_index = tuple(np.random.randint(0, high) for high in self.spin_grid.grid_shape)

        # Pre-Flip State
        preflip_spin = self.spin_grid.grid[tuple(random_index)]
        preflip_energy = self.total_energy()

        # Find a new magnetization vector (randomly)
        p: float = 0.5
        postflip_spin = p * self.total_magnetization() + (1-p) * preflip_spin
        postflip_spin = postflip_spin / np.linalg.norm(postflip_spin)

        # Post-Flip State
        self.spin_grid.grid[random_index] = postflip_spin  # Update the grid with the new magnetization
        postflip_energy = self.total_energy()

        # Calculate the metropolis acceptance probability
        ΔE = postflip_energy - preflip_energy
        p_accept = min(1, np.exp(-ΔE * beta))

        # Accept or reject the flip based on the acceptance probability
        if np.random.rand() < p_accept:
            ΔM = postflip_spin - preflip_spin
            return ΔE, ΔM
        else:
            self.spin_grid.grid[tuple(random_index)] = preflip_spin         # Revert to the original state
            return 0, np.zeros(self.spin_grid.spin_dimension, dtype=float)  # No change in spin state



SPIN_GRID = SpinGrid(resolution=(10, 8, 6), dimension=3, fill=None)
SPIN_INTERACTION = SpinInteraction(strength=1.0, radius=4.0, falloff=2.0)
MAGNETIC_FIELD = MagneticField(vector=np.array([0.0, 0.0, 0.0], dtype=float))

ISING_MODEL = IsingModel(spin_grid=SPIN_GRID, interaction=SPIN_INTERACTION, magnetic_field=MAGNETIC_FIELD)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ISING_MODEL.iterate(beta=1.0)  # Update internal state
    ISING_MODEL.spin_grid.render_animation(ax)
    return []

ani = FuncAnimation(fig, update, frames=1000, interval=100, blit=False)

ani.save("spin_grid.gif", writer=PillowWriter(fps=10))

plt.show()
