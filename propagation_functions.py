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

    def threebody_propagator(self, init_r, init_v, tof, steps, init_epic):
        """
        Function to propagate a given orbit
        """
        et=spice.str2et(init_epic)
        p={"ep0":et}
        # Time vector
        tspan = [0, tof]
        # Array of time values
        tof_array = np.linspace(0,tof, num=steps)
        init_state = np.concatenate((init_r,init_v))
        # Do th integration
        sol = solve_ivp(fun = lambda t,x:self.threebody_eoms(t,x,p), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

        # Return everything
        return sol.y, sol.t
    
    def threebody_eoms(self, t, state, p):
        """
        Equation of motion for 2body orbits
        """
        earth_nu = 398600.441500000
        moon_nu = 4902.80058214780
        sun_nu = 132712440017.990
        # Extract values from init
        x, y, z, vx, vy, vz = state
        r_dot = np.array([vx, vy, vz])
        init_epic=p["ep0"]
        current_epic=init_epic+t
        # Define r
        r = (x**2 + y**2 + z**2)**.5
        res=[1,1,1]
        rem=[1,1,1]
        rssc=state[0:3]-res
        rmsc=state[0:3]-rem
        ax = -earth_nu*x/(r**3)-sun_nu*(rssc[0]/(np.linalg.norm(rssc)**3) +res[0]/(np.linalg.norm(res)**3))-moon_nu*(rmsc[0]/(np.linalg.norm(rmsc)**3) + rem[0]/(np.linalg.norm(rem)**3))

        ay = -earth_nu*y/(r**3)-sun_nu*(rssc[1]/(np.linalg.norm(rssc)**3) +res[1]/(np.linalg.norm(res)**3))-moon_nu*(rmsc[1]/(np.linalg.norm(rmsc)**3) + rem[1]/(np.linalg.norm(rem)**3))

        az = -earth_nu*z/(r**3)-sun_nu*(rssc[2]/(np.linalg.norm(rssc)**3) +res[2]/(np.linalg.norm(res)**3))-moon_nu*(rmsc[2]/(np.linalg.norm(rmsc)**3) + rem[2]/(np.linalg.norm(rem)**3))

        v_dot = np.array([ax, ay, az])

        dx = np.append(r_dot, v_dot)

        return dx