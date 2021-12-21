import numpy as np
import matplotlib.pyplot as plt
import pde as pde 
import numba as nb

class TuringPDE(pde.PDEBase):
    """Turing patterns"""

    def __init__(self, k=-0.005, diffusivity = [2.8e-4, 5e-3], bc="natural"):
        self.k = k
        self.diffusivity = diffusivity  # spatial mobility
        self.bc = bc  # boundary condition

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        u = pde.ScalarField(grid, 0.5, label="activatior field")
        v = pde.ScalarField.random_normal(grid, label="Field $v$")
        return pde.FieldCollection([u, v])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        u, v = state
        rhs = state.copy()
        d0, d1 = self.diffusivity
        rhs[0] = d0 * u.laplace(self.bc) + self.k + u - u**3 - v 
        rhs[1] = d1 * v.laplace(self.bc) + u - v
        return rhs

    def _make_pde_rhs_numba(self, state):
        """nunmba-compiled implementation of the PDE"""
        d0, d1 = self.diffusivity
        k = self.k
        laplace = state.grid.make_operator("laplace", bc=self.bc)

        @nb.jit
        def pde_rhs(state_data, t):
            u = state_data[0]
            v = state_data[1]

            rate = np.empty_like(state_data)
            rate[0] = d0 * laplace(u) + k + u - u**3 - v 
            rate[1] = d1 * laplace(v) + u - v 
            return rate

        return pde_rhs


# initialize state
grid = pde.CartesianGrid([[-1, 1],[-1,1]],100)
eq = TuringPDE(diffusivity=[3e-4, 5e-3])
state = eq.get_initial_state(grid)

# simulate the pde
#tracker = pde.PlotTracker(interval=1, plot_args={"vmin": 0, "vmax": 0.3})
sol = eq.solve(state, t_range=10, dt=1e-3)
sol[0].plot()
plt.show()