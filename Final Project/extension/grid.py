# Stanley Goodwin, 4/30/2025
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SpinGrid:

    # Converts "size" to a list of integers
    def _convert_size(self, resolution) -> list[int]:
        if not isinstance(resolution, (int, np.integer, list, tuple, np.ndarray)):
            raise TypeError("Resolution must be a list/tuple/array.")
        if isinstance(resolution, (int, np.integer)):
            resolution = [resolution]
        if isinstance(resolution, (list, tuple, np.ndarray)):
            if len(resolution) == 0:
                raise TypeError("Resolution must be a non-empty list/tuple/array.")
            if not all(isinstance(s, (int, np.integer)) for s in resolution):
                raise TypeError("Resolution must be a list/tuple/array of integers.")
            if not all(s > 0 for s in resolution):
                raise TypeError("Resolution must be a list/tuple/array of positive integers.")
            
        # There only exists a non-empty, positive-integer list left.
        return list(resolution)

    # Converts "dimension" to an integer or None
    def _convert_dimension(self, dimension) -> int:
        if not isinstance(dimension, (int, np.integer, type(None))):
            raise TypeError("Dimension must be an integer or None.")
        if isinstance(dimension, (int, np.integer)) and not dimension > 0:
            raise TypeError("Dimension must be a positive integer.")
        return int(dimension) if dimension is not None else len(self.grid_shape)

    # Converts "fill" to a list of floats
    def _convert_fill(self, fill) -> np.ndarray | None:

        # Check its type and convert if necessary
        if not isinstance(fill, (int, float, np.integer, np.floating, list, tuple, np.ndarray, type(None))):
            raise TypeError("Fill must be a list/tuple/array or None.")
        if isinstance(fill, (int, float, np.integer, np.floating)):
            fill = np.array([fill], dtype=np.float64)
        if isinstance(fill, (list, tuple)):
            fill = np.array(fill, dtype=np.float64)
        if isinstance(fill, np.ndarray): 
            fill = np.asarray(fill, dtype=np.float64)

        # Check that fill is a valid spin vector
        if fill is not None:
            if fill.ndim != 1:            raise ValueError("Fill must be a 1D numpy array.")
            if np.linalg.norm(fill) == 0: raise ValueError("Fill vector cannot be a zero vector.")
            if fill.shape[0] != self.spin_dimension:
                raise ValueError("Fill vector must have the same dimension as the spin dimension.")
            fill = fill / np.linalg.norm(fill)  # Normalize the fill vector

        # Return the fill vector
        return fill if fill is not None else None


    # Initialize the SpinGrid object
    def __init__(self, *,
        resolution: int | list[int] | tuple[int] | np.ndarray[np.integer], 
        dimension:  int | None = None,
        fill: list[float] | tuple[float] | np.ndarray | None = None
    ):
        """
        Holds all the information pertaining to the spin grid.
        The grid is a multi-dimensional array of spins, each represented as a unit vector.

        Parameters:
        - grid_dimensions: int or list/tuple/array of integers specifying the size of the grid.
        - spin_dimension:  int specifying the number of dimensions for the spins.
        - fill: list/tuple/array of floats specifying the fill vector. 
                If None, random unit vectors are used.
        """

        # Do preliminary checks on the inputs
        self.grid_shape      = self._convert_size(resolution)
        self.grid_volume     = np.prod(self.grid_shape)
        self.grid_dimension  = len(self.grid_shape)
        self.spin_dimension  = self._convert_dimension(dimension)
        _fill = self._convert_fill(fill)

        # Generate the grid
        if _fill is None:
            # Randomly distributed in the unit N-sphere
            grid = np.random.uniform(-1, 1, size=tuple(self.grid_shape) + (self.spin_dimension,))
            norms = np.linalg.norm(grid, axis=-1, keepdims=True)
            self.grid = grid / norms
        else:
            # With the same direction as fill
            self.grid = np.full(tuple(self.grid_shape) + _fill.shape, _fill, dtype=np.float64)


    def render(self, 
        force_spin_z: bool = False,
        force_yz_plane: bool = False
    ) -> None:
        """ Render the grid as a 3D plot """
        
        # Return if the grid is more than 3D
        if self.spin_dimension > 3 or len(self.grid_shape) > 3: 
            raise ValueError("Dimensions must be 3 or less for rendering.")

        # Related misc settings
        if force_spin_z and (self.spin_dimension != 1):
            raise ValueError("force_spin_z is only valid for 1D spins.")
        if force_yz_plane and (self.spin_dimension != 2 or len(self.grid_shape) != 1):
            raise ValueError("force_yz_plane is only valid for 2D spins on 1D spin chains.")


        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        # Plot the points and spins
        NODE_DIMENSION = len(self.grid_shape)
        SPIN_DIMENSION = self.spin_dimension

        # Plot the nodes
        if NODE_DIMENSION == 1:
            positions = np.arange(self.grid_shape[0]), np.zeros(self.grid_shape[0]), np.zeros(self.grid_shape[0])
        elif NODE_DIMENSION == 2:
            positions = np.mgrid[0:self.grid_shape[0], 0:self.grid_shape[1]]
            positions = positions[0].flatten(), positions[1].flatten(), np.zeros(self.grid_shape[0] * self.grid_shape[1])
        elif NODE_DIMENSION == 3:   
            positions = np.mgrid[0:self.grid_shape[0], 0:self.grid_shape[1], 0:self.grid_shape[2]]
            positions = positions[0].flatten(), positions[1].flatten(), positions[2].flatten()
        ax.scatter(positions[0], positions[1], positions[2], s=100, color='red', marker='o')

        # Plot the spins
        ARROW_LENGTH = 1.0
        spin_x = self.grid[..., 0].flatten()
        spin_y = self.grid[..., 1].flatten() if SPIN_DIMENSION > 1 else np.zeros_like(spin_x)
        spin_z = self.grid[..., 2].flatten() if SPIN_DIMENSION > 2 else np.zeros_like(spin_x)
        if force_spin_z: spin_x, spin_y, spin_z = np.zeros_like(spin_x), np.zeros_like(spin_y), spin_x
        if force_yz_plane: spin_x, spin_y, spin_z = np.zeros_like(spin_x), spin_x, spin_y
        ax.quiver(
            positions[0], positions[1], positions[2],
            spin_x, spin_y, spin_z,
            length=ARROW_LENGTH, normalize=True, color='blue'
        )

        # Set up the plot
        ax.set_title('Spin Grid')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
        # Make all axes equal
        if NODE_DIMENSION >= 1:
            HALF_SIZE = self.grid_shape[0] // 2
            ax.set_xlim([-1, self.grid_shape[0]])
            ax.set_ylim([-HALF_SIZE, HALF_SIZE])
            ax.set_zlim([-HALF_SIZE, HALF_SIZE])
        if NODE_DIMENSION >= 2:
            HALF_SIZE = max(self.grid_shape[0], self.grid_shape[1]) // 2
            ax.set_xlim([-1, self.grid_shape[0]])
            ax.set_ylim([-1, self.grid_shape[1]])
            ax.set_zlim([-HALF_SIZE, HALF_SIZE])
        if NODE_DIMENSION >= 3:
            ax.set_xlim([-1, self.grid_shape[0]])
            ax.set_ylim([-1, self.grid_shape[1]])
            ax.set_zlim([-1, self.grid_shape[2]])
        
        # Remove ticks for higher dimensions
        if NODE_DIMENSION >= 1: ax.set_xticks([])
        if NODE_DIMENSION >= 2: ax.set_yticks([])
        if NODE_DIMENSION >= 3: ax.set_zticks([])

        # Show the plot
        plt.show()


    def render_animation(self, ax,
        force_spin_z: bool = False,
        force_yz_plane: bool = False,
    ) -> None:
        """ Render the grid on a provided 3D Axes object (ax). Suitable for use in animation. """
    
        # Return if the grid is more than 3D
        if self.spin_dimension > 3 or len(self.grid_shape) > 3: 
            raise ValueError("Dimensions must be 3 or less for rendering.")

        if force_spin_z and (self.spin_dimension != 1):
            raise ValueError("force_spin_z is only valid for 1D spins.")
        if force_yz_plane and (self.spin_dimension != 2 or len(self.grid_shape) != 1):
            raise ValueError("force_yz_plane is only valid for 2D spins on 1D spin chains.")

        # Clear previous plot
        ax.clear()
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        # Determine positions
        NODE_DIMENSION = len(self.grid_shape)
        SPIN_DIMENSION = self.spin_dimension
        if NODE_DIMENSION == 1:
            x = np.arange(self.grid_shape[0])
            y = np.zeros_like(x)
            z = np.zeros_like(x)
        elif NODE_DIMENSION == 2:
            pos = np.mgrid[0:self.grid_shape[0], 0:self.grid_shape[1]]
            x = pos[0].flatten()
            y = pos[1].flatten()
            z = np.zeros_like(x)
        elif NODE_DIMENSION == 3:
            pos = np.mgrid[0:self.grid_shape[0], 0:self.grid_shape[1], 0:self.grid_shape[2]]
            x = pos[0].flatten()
            y = pos[1].flatten()
            z = pos[2].flatten()

        ax.scatter(x, y, z, s=100, color='red', marker='o')

        # Determine spin vectors
        spin_x = self.grid[..., 0].flatten()
        spin_y = self.grid[..., 1].flatten() if SPIN_DIMENSION > 1 else np.zeros_like(spin_x)
        spin_z = self.grid[..., 2].flatten() if SPIN_DIMENSION > 2 else np.zeros_like(spin_x)
        if force_spin_z:
            spin_x, spin_y, spin_z = np.zeros_like(spin_x), np.zeros_like(spin_y), spin_x
        if force_yz_plane:
            spin_x, spin_y, spin_z = np.zeros_like(spin_x), spin_x, spin_y

        # Plot spin vectors
        ax.quiver(x, y, z, spin_x, spin_y, spin_z,
                length=1.0, normalize=True, color='blue')

        # Set labels
        ax.set_title('Spin Grid')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Set axis limits
        if NODE_DIMENSION == 1:
            HALF_SIZE = self.grid_shape[0] // 2
            ax.set_xlim([-1, self.grid_shape[0]])
            ax.set_ylim([-HALF_SIZE, HALF_SIZE])
            ax.set_zlim([-HALF_SIZE, HALF_SIZE])
        elif NODE_DIMENSION == 2:
            HALF_SIZE = max(self.grid_shape[0], self.grid_shape[1]) // 2
            ax.set_xlim([-1, self.grid_shape[0]])
            ax.set_ylim([-1, self.grid_shape[1]])
            ax.set_zlim([-HALF_SIZE, HALF_SIZE])
        elif NODE_DIMENSION == 3:
            ax.set_xlim([-1, self.grid_shape[0]])
            ax.set_ylim([-1, self.grid_shape[1]])
            ax.set_zlim([-1, self.grid_shape[2]])

        # Remove axis ticks for higher dims
        if NODE_DIMENSION >= 1: ax.set_xticks([])
        if NODE_DIMENSION >= 2: ax.set_yticks([])
        if NODE_DIMENSION >= 3: ax.set_zticks([])



# Test the generate_grid function
if __name__ == "__main__":

    # Test with a grid
    test = SpinGrid(resolution=(3,4,5), dimension=3, fill=None)
    print("Grid size:", test.grid_shape)
    print("Grid dimension:", test.spin_dimension)
    print("Grid shape:", test.grid.shape)
    test.render()
    # test.render(force_yz_plane=True)
    # test.render(force_spin_z=True)