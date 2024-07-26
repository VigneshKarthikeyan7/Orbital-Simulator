# Main set of code that we will use to run everything
import numpy as np
from propagation_functions import propagation_tools
import matplotlib.pyplot as plt

propagation_tools = propagation_tools()
def main():
    # Main function
    print("Starting Main Function")
    initial_position = np.array([7000,0,0])
    
    initial_velocity = np.array([0, 7.72, 5])
    integration_time = 24*60*60
    steps = 100                                                                     
 
    theta = 90

    
    trajectory, times = propagation_tools.keplerian_propagator(initial_position, initial_velocity, integration_time, steps)

    init_epic="Jan 1, 2020"
    trajectory_threebody, times_threebody = propagation_tools.threebody_propagator(initial_position, initial_velocity, integration_time, steps, init_epic)

    # Initial rotation relative to the Earth
    theta = np.pi/2
    trajectory_J2, times_J2 = propagation_tools.Jtwo_propagator(initial_position, initial_velocity, integration_time, steps, theta)
    
    # With Drag
    trajectory_drag, times_drag = propagation_tools.drag_propagator(initial_position, initial_velocity, integration_time, steps)

    # What does this look like                      
    # Plot it
    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes(projection='3d',computed_zorder=False)
    # Plot x, y, z
    ax.plot(trajectory[0],trajectory[1],trajectory[2],zorder=5)
    ax.plot(trajectory_threebody[0],trajectory_threebody[1],trajectory_threebody[2],zorder=5)
    ax.plot(trajectory_J2[0],trajectory_J2[1],trajectory_J2[2],zorder=5)
    ax.plot(trajectory_drag[0],trajectory_drag[1],trajectory_drag[2],zorder=5)
    plt.title("All Orbits")
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    ax.set_zlabel("Z-axis (km)")
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    # Over short timespans, that plot should look like 1 orbit because the trajectories are so close
    # Go ahead and try different integration times to see what happens
    # With this plot, it should be easier to see the differences
    fig = plt.figure(1)
    ax = plt.axes()
  
    labels = ['X Error 4B', 'Y Error 4B', 'Z Error 4B']
    # Get the x, y, and z error
    for i in range(3):
        three_body_dif = np.abs(trajectory[i] - trajectory_threebody[i])
        ax.plot(times,three_body_dif,label=labels[i])
    # Get error from J2

    # COMMENT OUT THIS BLOCK TO COMPARE
    labels = ['X Error J2', 'Y Error J2', 'Z Error J2']
    for i in range(3):
        J2_dif = np.abs(trajectory[i] - trajectory_J2[i])
        ax.plot(times,J2_dif,label=labels[i])

    labels = ['X Error Drag', 'Y Error Drag', 'Z Error Drag']
    for i in range(3):
        Drag_dif = np.abs(trajectory[i] - trajectory_drag[i])
        ax.plot(times,Drag_dif,label=labels[i])

    plt.title("Error vs Time")
    plt.legend()
    ax.set_ylabel("Error (km)")
    ax.set_xlabel("Time [sec]")
    plt.grid()
    plt.show()
    plt.waitforbuttonpress




if __name__ == '__main__':
    main()