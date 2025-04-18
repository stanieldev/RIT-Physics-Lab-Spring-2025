import numpy as np


# Vector class
class Vector3:
    def __init__(self, x: float, y: float, z: float):
        assert isinstance(x, (int, float)), "x must be an int or float."
        assert isinstance(y, (int, float)), "y must be an int or float."
        assert isinstance(z, (int, float)), "z must be an int or float."
        self.x = x
        self.y = y
        self.z = z

    # Calculate the norm of the vector (property)
    @property
    def norm(self) -> float:
        # TODO: Complex numbers and other types of numbers
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    # Calculate the normalized vector (property)
    @property
    def normalized(self) -> "Vector3":
        _norm: float = self.norm
        if _norm == 0: raise ValueError("Cannot normalize a zero vector.")
        return Vector3(self.x / _norm, self.y / _norm, self.z / _norm)

    # Normalize the instance of vector
    def normalize(self) -> None:
        _norm: float = self.norm
        if _norm == 0: raise ValueError("Cannot normalize a zero vector.")
        self.x /= _norm
        self.y /= _norm
        self.z /= _norm

    # Unary Operations
    def __pos__(self):  return self
    def __neg__(self):  return Vector3(-self.x, -self.y, -self.z)
    def __abs__(self):  return self.norm
    def __bool__(self): return self.norm != 0
    def __hash__(self): return hash((self.x, self.y, self.z))

    # Binary Addition (Both Forward and Reverse)
    def  __add__(self, other):
        if isinstance(other, Vector3): return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        else: raise NotImplementedError(f"Addition with {type(other)} is not implemented.")
    def __radd__(self, other): return self. __add__(other)
    def __iadd__(self, other):
        if isinstance(other, Vector3): self.x += other.x; self.y += other.y; self.z += other.z
        else: raise NotImplementedError(f"In-place addition with {type(other)} is not implemented.")
        return self
    def  __sub__(self, other): return self. __add__(other * -1)
    def __rsub__(self, other): return self. __add__(other * -1)
    def __isub__(self, other): return self.__iadd__(other * -1)

    # Binary Multiplication (Both Forward and Reverse)
    def  __mul__(self, other):        
        if isinstance(other, (int, float, np.float64)): return Vector3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3):    return self.x * other.x + self.y * other.y + self.z * other.z
        else: raise NotImplementedError(f"Multiplication with {type(other)} is not implemented.")
    def __rmul__(self, other): return self.__mul__(other)
    def __imul__(self, other):
        if isinstance(other, (int, float)): self.x *= other; self.y *= other; self.z *= other
        elif isinstance(other, Vector3):    raise ValueError("In-place multiplication with another vector is not a defined operation.")
        else: raise NotImplementedError(f"In-place multiplication with {type(other)} is not implemented.")
    def  __truediv__(self, other):
        if isinstance(other, (int, float)): return Vector3(self.x / other, self.y / other, self.z / other)
        else: raise NotImplementedError(f"Division with {type(other)} is not implemented.")
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)): return Vector3(other / self.x, other / self.y, other / self.z)
        else: raise NotImplementedError(f"Reverse division with {type(other)} is not implemented.")
    def __itruediv__(self, other): self.__imul__(1 / other); return self
    
    # Python Well Ordering and Comparison
    def __eq__(self, other): return isinstance(other, Vector3) and self.x == other.x and self.y == other.y and self.z == other.z
    def __ne__(self, other): return not self.__eq__(other)
    def __lt__(self, other): return isinstance(other, Vector3) and self.norm <  other.norm
    def __le__(self, other): return isinstance(other, Vector3) and self.norm <= other.norm
    def __gt__(self, other): return isinstance(other, Vector3) and self.norm >  other.norm
    def __ge__(self, other): return isinstance(other, Vector3) and self.norm >= other.norm

    # Python Iteration
    def __len__(self):
        return 3
    def __iter__(self):
        return iter((self.x, self.y, self.z))
    def __sizeof__(self):
        return 3 * float.__sizeof__()  # Size of the vector in bytes
    def __getitem__(self, index):
        if index == 0: return self.x
        elif index == 1: return self.y
        elif index == 2: return self.z
        else: raise IndexError("Index out of range.")
    def __setitem__(self, index, value):
        if index == 0: self.x = value
        elif index == 1: self.y = value
        elif index == 2: self.z = value
        else: raise IndexError("Index out of range.")

    # Python Printing and Formatting
    def __repr__(self): return f"Vector3({self.x}, {self.y}, {self.z})"
    def __str__(self):  return f"Vector3({self.x}, {self.y}, {self.z})"

    # Python Copying and Deleting
    def __copy__(self):            # Shallow copy
        return Vector3(self.x, self.y, self.z)
    def __deepcopy__(self, memo):  # Deep copy
        return Vector3(self.x, self.y, self.z)
    def __del__(self):             # Destructor
        del self.x; del self.y; del self.z  
