import numpy as np
import matplotlib.pyplot as plt


from pydrake.all import (DiagramBuilder, PlanarSceneGraphVisualizer, SceneGraph, 
                         Simulator, AddMultibodyPlantSceneGraph, Parser, ConnectPlanarSceneGraphVisualizer, ConstantVectorSource)
from pydrake.trajectories import Trajectory, PiecewisePolynomial

def visualize_pendulum_trajectory(trajectory_state_array, time_array):
    # Ensure the trajectory_state_array and time_array have the same length.
    assert len(trajectory_state_array) == len(time_array)

    # Create a pendulum model and a SceneGraph.
    builder = DiagramBuilder()
    pendulum, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.)
    parser = Parser(pendulum)
    parser.AddModels(
        url="package://drake/examples/pendulum/Pendulum.urdf")
    pendulum.Finalize()


    # Connect a constant zero input to the actuation input port.
    actuation_input = builder.AddSystem(ConstantVectorSource([0]))
    builder.Connect(actuation_input.get_output_port(0), pendulum.get_actuation_input_port())


    T_VW = np.array([[1., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    visualizer = ConnectPlanarSceneGraphVisualizer(
        builder, scene_graph, T_VW=T_VW, xlim=[-2.5, 2.5],
        ylim=[-2.5, 2.5], show=True)
    #builder.Connect(scene_graph.get_query_output_port(), visualizer.get_input_port(0))

    # Build the Diagram.
    diagram = builder.Build()

    # Create a trajectory from the trajectory_state_array and time_array.
    trajectory = PiecewisePolynomial.FirstOrderHold(time_array, trajectory_state_array.T)

    # Set up the simulator and context.
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    pendulum_context = diagram.GetMutableSubsystemContext(pendulum, context)
    simulator.set_target_realtime_rate(1)

    # Visualize the trajectory.
    for t in time_array:
        # Advance the simulator and visualize.
        simulator.AdvanceTo(t)

        # Set the pendulum state from the trajectory.
        pendulum_context.get_mutable_continuous_state_vector().SetFromVector(trajectory.value(t))

        
        #diagram.Publish(context)
        #visualizer.draw(context)
        print(trajectory.value(t))

    visualizer.stop_recording()
    ani = visualizer.get_recording_as_animation()
    plt.show()
    return ani
    # Show the visualization.
    #

def interpolate_trajectory_state_array(trajectory_state_array, time_array, new_time_array):
    # Create a PiecewisePolynomial trajectory from the trajectory_state_array and time_array.
    trajectory = PiecewisePolynomial.FirstOrderHold(time_array, trajectory_state_array.T)

    # Interpolate the trajectory for the new_time_array.
    interpolated_trajectory_state_array = np.array([trajectory.value(t) for t in new_time_array]).squeeze()

    return interpolated_trajectory_state_array

# Example usage:
trajectory_state_array = np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.7], [0.3, 0.5], [0.4, 0.3], [0.5, 0.1]])
time_array = np.linspace(0, 5, len(trajectory_state_array))
new_time_array = np.linspace(0, 5, 50*len(trajectory_state_array))  # New time array with more points

interpolated_trajectory_state_array = interpolate_trajectory_state_array(trajectory_state_array, time_array, new_time_array)

# # Example usage:
# trajectory_state_array = np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.7], [0.3, 0.5], [0.4, 0.3], [0.5, 0.1]])
# time_array = np.linspace(0, 5, 20*len(trajectory_state_array))

visualize_pendulum_trajectory(interpolated_trajectory_state_array, new_time_array)