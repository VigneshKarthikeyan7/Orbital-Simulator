import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp


class propagation_tools:
    def init_ephem(self):
        """
        Initialize the ephemeris files for EMS
        """
        # This is for the Cassini example, comment out later
        #spice.furnsh("./Ephemeris/cassMetaK.txt")
        # Furnish the kernals we actually need
        spice.furnsh("./Ephemeris/ephemMeta.text")

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
        self.init_ephem()
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
        # Get vector from E to S
        res, _ = spice.spkpos('Sun', current_epic, 'J2000', 'LT', 'Earth')
        # Get vector from E to M
        rem, _ = spice.spkpos('Moon', current_epic, 'J2000', 'LT', 'Earth')

        rssc = state[0:3]-res
        rmsc = state[0:3]-rem

        ax = -earth_nu*x/(r**3)-sun_nu*(rssc[0]/(np.linalg.norm(rssc)**3) + res[0]/(np.linalg.norm(res)**3))-moon_nu*(rmsc[0]/(np.linalg.norm(rmsc)**3) + rem[0]/(np.linalg.norm(rem)**3))

        ay = -earth_nu*y/(r**3) - sun_nu*(rssc[1]/(np.linalg.norm(rssc)**3) +res[1]/(np.linalg.norm(res)**3))-moon_nu*(rmsc[1]/(np.linalg.norm(rmsc)**3) + rem[1]/(np.linalg.norm(rem)**3))

        az = -earth_nu*z/(r**3) - sun_nu*(rssc[2]/(np.linalg.norm(rssc)**3) +res[2]/(np.linalg.norm(res)**3))-moon_nu*(rmsc[2]/(np.linalg.norm(rmsc)**3) + rem[2]/(np.linalg.norm(rem)**3))

        v_dot = np.array([ax, ay, az])

        dx = np.append(r_dot, v_dot)

        return dx
    
    def Jtwo_propagator(self, init_r, init_v, tof, steps, init_theta): 
        """
        Propogates accounting for Earth Oblateness
        """
        # Time vector
        tspan = [0, tof]
        # Array of time values
        tof_array = np.linspace(0,tof, num=steps)
        # Need to add an extra term which is the rotation angle
        init_state = np.concatenate((init_r,init_v,[init_theta]))
        # Do th integration
        sol = solve_ivp(fun = lambda t,x:self.Jtwo_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

        # Return everything
        return sol.y, sol.t

    def Jtwo_eoms(self, t, state):
        """
        Equations of Motion that account for Earth Oblateness  
        """
        earth_nu = 398600.441500000
        J2 = 0.00108264
        earth_radius = 6378.00
        # First need to rotate our current position
        # Only need to rotate position
        r_ecef = self.rot3(state[0:3],state[6])
        # Define variables
        x, y, z = r_ecef
        r_dot = state[3:6]
        
        # Define r
        r = np.linalg.norm(r_ecef)
        # Creating a matrix
        s=x/r
        t=y/r
        u=z/r
        # U vector
        u_ECEF = np.array([s, t, u])
        # UJ2 term
        U_J2 = (3/2)*np.array([5*s*u**2 - s, 5*t*u**2 - t, 5*u**3 - 3*u])
        # Acceleration with everything
        ag_ECEF = (-earth_nu/r**2)*(u_ECEF) + (earth_nu*J2/r**2)*(earth_radius/r)**2*U_J2
        # Need to rotate acceleration back to ECI
        A_ECI = self.rot3(ag_ECEF,-state[6])
        # Also need the rotation rate of the Earth
        theta_dot = 2*np.pi/(24*60*60)
        temp = np.append(r_dot,A_ECI)
        dx = np.append(temp, theta_dot)
    
        return dx
    
    def rot3(self, vec, xval):
        """
        Function to rotate a vector about the z axis by the angle theta
        WARNING: Does a clockwise rotation instead of CCW
        """
        temp= vec[1]
        c = np.cos( xval )
        s = np.sin( xval )
        # Predefine
        outvec = np.array([0.0,0.0,0.0])
        # Save the actual slots
        outvec[1] = c*vec[1] - s*vec[0]
        outvec[0] = c*vec[0] + s*temp
        outvec[2] = vec[2]
        # Return rotated vector
        return outvec
    
    def drag_eoms(self, init_r, init_v, state):
        """
        Function that calculates the effect of atmospheric drag on
        non-equatorial orbits
        """
        # C value chosen as 2
        C = 2
        # Altitude selected was 100 km
        p_r = 0.000000461
        # Pick a spherical area value
        A = 10
        # Pick a mass
        m = 5
        v_prime = r_dot
        # Extract values from init
        x, y, z, vx, vy, vz = state
        r_dot = np.array([vx, vy, vz])
        a_drag=-(C/2)*p_r*(A/m)*(v_prime)**2*(v_prime/abs(v_prime))
        
        return a_drag