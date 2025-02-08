import numpy as np


# Value Summation Functions
def riemann_left(f: callable, 
                 a: float|int, 
                 b: float|int, *, 
                 dx: float|int|None = None,
                 N: int|None = None
                 ) -> float:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a, b-(b-a)/N, N)
        return np.sum(f(x)) * (b-a)/N
    elif dx is not None:
        x = np.arange(a, b, dx)
        return np.sum(f(x)) * dx
    else:
        raise Exception("Something went wrong...")

def riemann_right(f: callable,
                  a: float|int,
                  b: float|int, *,
                  dx: float|int|None = None,
                  N: int|None = None
                  ) -> float:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a+(b-a)/N, b, N)
        return np.sum(f(x)) * (b-a)/N
    elif dx is not None:
        x = np.arange(a+dx, b+dx, dx)
        return np.sum(f(x)) * dx
    else:
        raise Exception("Something went wrong...")

def riemann_middle(f: callable,
                   a: float|int,
                   b: float|int, *,
                   dx: float|int|None = None,
                   N: int|None = None
                   ) -> float:
     # Function requirements
     assert callable(f)
     assert isinstance(a, (int, float))
     assert isinstance(b, (int, float))
     assert N is not None or dx is not None
     assert not (N is not None and dx is not None)
     if N is not None: assert isinstance(N, int)
     if dx is not None: assert isinstance(dx, (int, float))
    
     # Calculate the integral
     if N is not None:
          x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
          print(x)
          return np.sum(f(x)) * (b-a)/N
     elif dx is not None:
          x = np.arange(a+dx/2, b, dx)
          print(x)
          return np.sum(f(x)) * dx
     else:
          raise Exception("Something went wrong...")

def trapezoid(f: callable,
              a: float|int,
              b: float|int, *,
              dx: float|int|None = None,
              N: int|None = None
              ) -> float:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a, b, N+1)
        return np.sum(f(x[1:]) + f(x[:-1])) * (b-a)/(2*N)
    elif dx is not None:
        x = np.arange(a, b+dx, dx)
        return np.sum(f(x[1:]) + f(x[:-1])) * dx/2
    else:
        raise Exception("Something went wrong...")

def simpson(f: callable,
            a: float|int,
            b: float|int, *,
            dx: float|int|None = None,
            N: int|None = None
            ) -> float:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a, b, N+1)
        return np.sum(f(x[:-2:2]) + 4*f(x[1:-1:2]) + f(x[2::2])) * (b-a)/(3*N)
    elif dx is not None:
        x = np.arange(a, b+dx, dx)
        return np.sum(f(x[:-2:2]) + 4*f(x[1:-1:2]) + f(x[2::2])) * dx/3
    else:
        raise Exception("Something went wrong...")


# Integrating Functions
def riemann_left_function(f: callable, 
                             a: float|int, 
                             b: float|int, *, 
                             dx: float|int|None = None,
                             N: int|None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a, b-(b-a)/N, N)
        y = []
        integral = 0
        for i in range(N):
            integral += f(x[i])
            y.append(integral)
        return x, np.array(y) * (b-a)/N
    elif dx is not None:
        x = np.arange(a, b, dx)
        y = []
        integral = 0
        for i in range(len(x)):
            integral += f(x[i])
            y.append(integral)
        return x, np.array(y) * dx
    else:
        raise Exception("Something went wrong...")

def riemann_right_function(f: callable,
                              a: float|int,
                              b: float|int, *,
                              dx: float|int|None = None,
                              N: int|None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a+(b-a)/N, b, N)
        y = []
        integral = 0
        for i in range(N):
            integral += f(x[i])
            y.append(integral)
        return x, np.array(y) * (b-a)/N
    elif dx is not None:
        x = np.arange(a+dx, b+dx, dx)
        y = []
        integral = 0
        for i in range(len(x)):
            integral += f(x[i])
            y.append(integral)
        return x, np.array(y) * dx
    else:
        raise Exception("Something went wrong...")

def riemann_middle_function(f: callable,
                               a: float|int,
                               b: float|int, *,
                               dx: float|int|None = None,
                               N: int|None = None
                               ) -> tuple[np.ndarray, np.ndarray]:
     # Function requirements
     assert callable(f)
     assert isinstance(a, (int, float))
     assert isinstance(b, (int, float))
     assert N is not None or dx is not None
     assert not (N is not None and dx is not None)
     if N is not None: assert isinstance(N, int)
     if dx is not None: assert isinstance(dx, (int, float))
    
     # Calculate the integral
     if N is not None:
          x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
          y = []
          integral = 0
          for i in range(N):
                integral += f(x[i])
                y.append(integral)
          return x, np.array(y) * (b-a)/N
     elif dx is not None:
          x = np.arange(a+dx/2, b, dx)
          y = []
          integral = 0
          for i in range(len(x)):
                integral += f(x[i])
                y.append(integral)
          return x, np.array(y) * dx
     else:
          raise Exception("Something went wrong...")

def trapezoid_function(f: callable,
                          a: float|int,
                          b: float|int, *,
                          dx: float|int|None = None,
                          N: int|None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
        # Function requirements
        assert callable(f)
        assert isinstance(a, (int, float))
        assert isinstance(b, (int, float))
        assert N is not None or dx is not None
        assert not (N is not None and dx is not None)
        if N is not None: assert isinstance(N, int)
        if dx is not None: assert isinstance(dx, (int, float))
    
        # Calculate the integral
        if N is not None:
            x = np.linspace(a, b, N+1)
            y = []
            integral = 0
            for i in range(N):
                integral += f(x[i])
                y.append(integral)
            return x, np.array(y) * (b-a)/(2*N)
        elif dx is not None:
            x = np.arange(a, b+dx, dx)
            y = []
            integral = 0
            for i in range(len(x)):
                integral += f(x[i])
                y.append(integral)
            return x, np.array(y) * dx/2
        else:
            raise Exception("Something went wrong...")
        
def simpson_function(f: callable,
                        a: float|int,
                        b: float|int, *,
                        dx: float|int|None = None,
                        N: int|None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    # Function requirements
    assert callable(f)
    assert isinstance(a, (int, float))
    assert isinstance(b, (int, float))
    assert N is not None or dx is not None
    assert not (N is not None and dx is not None)
    if N is not None: assert isinstance(N, int)
    if dx is not None: assert isinstance(dx, (int, float))

    # Calculate the integral
    if N is not None:
        x = np.linspace(a, b, N+1)
        y = []
        integral = 0
        for i in range(N):
            integral += f(x[i])
            y.append(integral)
        return x, np.array(y) * (b-a)/(3*N)
    elif dx is not None:
        x = np.arange(a, b+dx, dx)
        y = []
        integral = 0
        for i in range(len(x)):
            integral += f(x[i])
            y.append(integral)
        return x, np.array(y) * dx/3
    else:
        raise Exception("Something went wrong...")
