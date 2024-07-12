# Main set of code that we will use to run everything
import numpy as np
from propagation_functions import propagation_tools
import mathplotlib.pyplot as plt

propagation_tools = propagation_tools()
def main():
    # Main function
    print("Starting Main Function")
    initial_position = np.array([7000,0,0])
    
    initial_velocity = np.array([0, 7.72, 5])
    integration_time = 5*60*60
    steps = 100
 
    theta = 30 # defined theta value as 30 for example. is it right?

    U_J2 = propagation_tools.Jtwo_propogator(initial_position, initial_velocity,theta)

    trajectory, times = propagation_tools.keplerian_propagator(initial_position, initial_velocity, integration_time, steps)

    init_epic="Jan 1, 2020"
    trajectory, times = propagation_tools.threebody_propagator(initial_position, initial_velocity, integration_time, steps, init_epic)

    # What does this look like                      
    # Plot it
    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes(projection='3d',computed_zorder=False)
    # Plot x, y, z
    ax.plot(trajectory[0],trajectory[1],trajectory[2],zorder=5)
    plt.title("Keplerian Orbit")
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    ax.set_zlabel("Z-axis (km)")
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    main()