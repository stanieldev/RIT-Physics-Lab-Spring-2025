# Stanley Goodwin, 4/30/2025
import numpy as np
from itertools import product



class SpinGrid:
    def __init__(self, *,
        resolution: int | list[int] | tuple[int] | np.ndarray[np.integer], 
        dimension: int | None = None,
        fill: list[float] | tuple[float] | np.ndarray | None = None
    ):
        """
        Initialize the SpinGrid object.

        Parameters:
        - resolution: int or list/tuple/array of integers specifying the size of the grid.
        - dimension:  int specifying the number of dimensions for the grid.
        - fill: list/tuple/array of floats specifying the fill vector. 
                If None, random unit vectors are used.
        """
        self.size      = None
        self.dimension = None
        self.grid      = None

        # Enforce type hints
        if not isinstance(resolution, (int, np.integer, list, tuple, np.ndarray)):
            raise TypeError("Resolution must be a list/tuple/array.")
        if isinstance(resolution, (int, np.integer)) and not resolution > 0:
            raise TypeError("Resolution must be a positive integer.")
        if isinstance(resolution, (list, tuple, np.ndarray)):
            if len(resolution) == 0:
                raise TypeError("Resolution must be a non-empty list/tuple/array.")
            if not all(isinstance(s, (int, np.integer)) for s in resolution):
                raise TypeError("Resolution must be a list/tuple/array of integers.")
            if not all(s > 0 for s in resolution):
                raise TypeError("Resolution must be a list/tuple/array of positive integers.")

        if not isinstance(dimension, (int, np.integer, type(None))):
            raise TypeError("Dimension must be an integer or None.")
        if isinstance(dimension, (int, np.integer)):
            if not dimension > 0:
                raise TypeError("Dimension must be a positive integer.")
            
        if not isinstance(fill, (list, tuple, np.ndarray, type(None))):
            raise TypeError("Fill must be a list, tuple, numpy array, or None.")
        if isinstance(fill, (list, tuple, np.ndarray)):
            if len(fill) == 0:
                raise TypeError("Fill must be a non-empty list, tuple, or numpy array.")
            if not all(isinstance(f, (int, float, np.integer, np.floating)) for f in fill):
                raise TypeError("Fill must be a list, tuple, or numpy array of numbers.")

        # Check consistency between parameters
        if isinstance(resolution, (int, np.integer)) and dimension is None:
            # Assume a 1D grid with "resolution" points
            self.size = (resolution,)
            self.dimension = 1
        elif isinstance(resolution, (int, np.integer)) and dimension is not None:
            # Assume a 1D grid with "resolution" points and "dimension" dimensions
            self.size = (resolution,)
            self.dimension = dimension
        elif isinstance(resolution, (list, tuple, np.ndarray)) and dimension is None:
            # Assume the dimension is the length of the resolution list/tuple/array
            self.size = tuple(resolution)
            self.dimension = len(self.size)
        elif isinstance(resolution, (list, tuple, np.ndarray)) and dimension is not None:
            # TODO: Better fix (Raise an error if the length of size and dimension do not match)
            # if len(resolution) != dimension: raise ValueError("Size and dimension must have the same length.")
            self.size = tuple(resolution)
            self.dimension = dimension

        # Check that fill is a valid spin vector
        if fill is not None:

            # Convert the fill vector to a numpy array
            if isinstance(fill, (list, tuple)):
                fill = np.array(fill, dtype=np.float64)
            elif isinstance(fill, np.ndarray): 
                fill = np.asarray(fill, dtype=np.float64)
            elif isinstance(fill, (int, float, np.integer, np.floating)):
                fill = np.array([fill], dtype=np.float64)
            else:
                raise ValueError("Fill must be a vector as a list, tuple, numpy array, or None.")
            if fill.ndim != 1:            raise ValueError("Fill must be a 1D numpy array.")
            if np.linalg.norm(fill) == 0: raise ValueError("Fill vector cannot be a zero vector.")
            fill = fill / np.linalg.norm(fill)  # Normalize the fill vector
            
            # Test that the fill vector has the same dimension as the grid
            if fill.shape[0] != self.dimension:
                raise ValueError("Fill vector must have the same dimension as the grid.")

        # Generate the grid
        if fill is None:
            # Randomly distributed in the unit N-sphere
            grid = np.random.uniform(-1, 1, size=self.size + (self.dimension,))
            norms = np.linalg.norm(grid, axis=-1, keepdims=True)
            self.grid = grid / norms
        else:
            # With the same direction as fill
            self.grid = np.full(self.size + fill.shape, fill, dtype=np.float64)

    def render(self):
        """ Render the grid as a 3D plot """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Return if the grid is more than 3D
        if self.dimension > 3: raise ValueError("Grid dimension must be 3 or less for rendering.")    

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        # Plot the points and spins
        ARROW_LENGTH = 0.25
        NODE_DIMENSION = len(self.size)
        SPIN_DIMENSION = self.dimension

        # Plot the nodes
        if NODE_DIMENSION == 1:
            positions = np.arange(self.size[0]), np.zeros(self.size[0]), np.zeros(self.size[0])
        elif NODE_DIMENSION == 2:
            positions = np.mgrid[0:self.size[0], 0:self.size[1]]
            positions = positions[0].flatten(), positions[1].flatten(), np.zeros(self.size[0] * self.size[1])
        elif NODE_DIMENSION == 3:   
            positions = np.mgrid[0:self.size[0], 0:self.size[1], 0:self.size[2]]
            positions = positions[0].flatten(), positions[1].flatten(), positions[2].flatten()
        ax.scatter(positions[0], positions[1], positions[2], s=100, color='red', marker='o')

        # Plot the spins
        spin_x = self.grid[..., 0].flatten()
        spin_y = self.grid[..., 1].flatten() if SPIN_DIMENSION > 1 else np.zeros_like(spin_x)
        spin_z = self.grid[..., 2].flatten() if SPIN_DIMENSION > 2 else np.zeros_like(spin_x)

        ax.quiver(
            positions[0], positions[1], positions[2],
            spin_x, spin_y, spin_z,
            length=ARROW_LENGTH, normalize=True, color='blue'
        )
        

        plt.show()





        





# def plot_spin_chain(spins, position):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the points, spins, and spin-chain line
#     ax.scatter(position, np.zeros(N), np.zeros(N), s=100, color='red', marker='o')
#     ax.quiver(position, np.zeros(N), np.zeros(N), spins[:, 0], spins[:, 1], spins[:, 2], length=0.25, normalize=True, label='Spin Directions')
#     ax.plot(position, np.zeros(N), np.zeros(N), color='black', linewidth=2, label='Spin Chain')

#     # Set up the plot
#     ax.set_title('Spin Chain after Monte Carlo Step')
#     ax.set_xlabel('X-axis')
#     ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
#     ax.set_xlim([0, N])
#     ax.set_ylim([-0.75, 0.75])
#     ax.set_zlim([-0.75, 0.75])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     plt.show()



# Test the SpinGrid class
def _test_spingrid_initialization():

    """ Resolution input test cases """
    fail_cases = [
        -1, 0, 0.5, "1", None, True, 2+1j, 
        [], (), np.array([]),
        [-1], [0], [0.5], ["0.5"],
        (-1), (0), (0.5), ("0.5"),
        [3, -1], [3, 0], [3, 0.5], [3, "0.5"],
        (3, -1), (3, 0), (3, 0.5), (3, "0.5"),
        [3, 5, -1], [3, 5, 0], [3, 5, 0.5], [3, 5, "0.5"],
        (3, 5, -1), (3, 5, 0), (3, 5, 0.5), (3, 5, "0.5"),
        np.array([-1]), np.array([0]), np.array([0.5]), np.array(["0.5"]),
        np.array([3, -1]), np.array([3, 0]), np.array([3, 0.5]), np.array([3, "0.5"]),
        np.array([3, 5, -1]), np.array([3, 5, 0]), np.array([3, 5, 0.5]), np.array([3, 5, "0.5"]),
    ]
    success_cases = [
        3,
        [3], (3), np.array([3]),
        [3, 2], (3, 2), np.array([3, 2]),
        [3, 2, 1], (3, 2, 1), np.array([3, 2, 1]),
        [3, 2, 1, 4], (3, 2, 1, 4), np.array([3, 2, 1, 4]),
    ]
    for case in fail_cases:
        try: SpinGrid(resolution=case)
        except TypeError: pass
        else: raise AssertionError(f"Expected TypeError for resolution={case}.")
    for case in success_cases:
        try: SpinGrid(resolution=case)
        except: AssertionError(f"Expected no error for resolution={case}.")
        else: pass


    """ Dimension input test cases """
    fail_cases = [
        -1, 0, 0.5, "1", True, 2+1j, [], (), np.array([]),
    ]
    success_cases = [
        None, 3, 14, 1, 2, 4, 5, 6, 7, 8, 9, 10,
    ]
    for case in fail_cases:
        try: SpinGrid(resolution=3, dimension=case)
        except TypeError: pass
        else: raise AssertionError(f"Expected TypeError for dimension={case}.")
    for case in success_cases:
        try: SpinGrid(resolution=3, dimension=case)
        except: AssertionError(f"Expected no error for dimension={case}.")
        else: pass


    """ Fill input test cases """
    fail_cases = [
        -1, 0, 0.5, "1", True, 2+1j, [], (), np.array([]),
        ["0.5"], ("0.5"), np.array(["0.5"]),
        [0, 0], (0, 0), np.array([0, 0]),
        [1, 2, 3], (1, 2, 3), np.array([1, 2, 3]),
        [1, 2, 3, 4], (1, 2, 3, 4), np.array([1, 2, 3, 4]),
    ]
    success_cases = [
        None, [1, 0], (1, 0), np.array([1, 0]),
    ]
    for case in fail_cases:
        try: SpinGrid(resolution=3, dimension=2, fill=case)
        except TypeError: pass
        except ValueError: pass
        else: raise AssertionError(f"Expected TypeError for fill={case}.")
    else:
        for case in success_cases:
            try: SpinGrid(resolution=3, dimension=2, fill=case)
            except: AssertionError(f"Expected no error for fill={case}.")
            else: pass

    """ Fill and resolution/size consistency test cases """
    # TODO



# Test the generate_grid function
if __name__ == "__main__":
    # Run the tests
    # _test_spingrid_initialization()
    # print("All tests passed.")

    test = SpinGrid(resolution=(8, 5, 3), dimension=3, fill=[1,0,0])
    print("Grid size:", test.size)
    print("Grid dimension:", test.dimension)
    print("Grid shape:", test.grid.shape)
    test.render()
    

