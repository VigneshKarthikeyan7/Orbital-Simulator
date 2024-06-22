import numpy as np
from scipy.integrate import solve_ivp



class propagation_tools:

    def keplerian_propagator(self, init_r, init_v, tof, steps):
        """
        Function to propagate a given orbit
        """
        # Time vector
        tspan = [0, tof]
        # Array of time values
        tof_array = np.linspace(0,tof, num=steps)
        init_state = np.concatenate((init_r,init_v))
        # Do th integration
        sol = solve_ivp(fun = lambda t,x:self.keplerian_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

        # Return everything
        return sol.y, sol.t


    def keplerian_eoms(self, t, state):
        """
        Equation of motion for 2body orbits
        """
        earth_nu = 398600.441500000
        # Extract values from init
        x, y, z, vx, vy, vz = state
        r_dot = np.array([vx, vy, vz])
        
        # Define r
        r = (x**2 + y**2 + z**2)**.5

        ax = -earth_nu*x/(r**3)
        ay = -earth_nu*y/(r**3)
        az = -earth_nu*z/(r**3)

        v_dot = np.array([ax, ay, az])

        dx = np.append(r_dot, v_dot)

        return dx
