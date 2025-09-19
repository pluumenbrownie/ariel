# Third-party libraries
from typing import Any, Self
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import matplotlib
import random as rd

from tqdm import tqdm

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.runners import simple_runner
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from Neural_Net import Brain, Layer, UniformBrain, SelfAdaptiveBrain
from plotters import FitnessPlotter

import json

# Keep track of data / history
HISTORY = []


def save_brain(brain: Brain) -> None:
    """Save the brain to a file."""
    with open(f"__data__/{type(brain).__name__}.json", "w") as f:
        json.dump(brain.export(), f, indent=4)


def load_brain(filename: str) -> Brain:
    """Load a brain from a file."""
    with open(filename, "r") as f:
        data = json.load(f)
        layers = []
        for layer_data in data["layers"]:
            function = None
            if layer_data["function"] == "sigmoid":
                function = sigmoid
            elif layer_data["function"] == "sigmoid_output":
                function = sigmoid_output
            else:
                raise ValueError(
                    f"Unknown activation function: {layer_data['function']}"
                )
            layer = Layer(
                layer_data["input_size"],
                layer_data["output_size"],
                function,
            )
            layer.weights = np.array(layer_data["weights"])
            layers.append(layer)
        if data["name"] == "UniformBrain":
            return UniformBrain(layers)
        elif data["name"] == "SelfAdaptiveBrain":
            return SelfAdaptiveBrain(layers, mutation_rate=data["mutation_rate"])

def save_fitness(weights: np.ndarray, brains: list[Brain]) -> None:
    """Save the fitness values to a file."""
    np.save(f"__data__/{type(brains[0]).__name__}_fitness.npy", weights)

def load_fitness(filename: str) -> np.ndarray:
    """Load the fitness values from a file."""
    return np.load(filename)


def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.

    The mujoco.set_mjcb_control() function will always give
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu

    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi / 2
    rand_moves = np.random.uniform(
        low=-hinge_range, high=hinge_range, size=num_joints  # -pi/2  # pi/2
    )

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    #
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################


def show_qpos_history(history: list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # plt.show()
    plt.savefig("__data__/plot.png")
    plt.close()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_output(x):
    return np.pi * (sigmoid(x) - 0.5)


def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    population = [
        SelfAdaptiveBrain(
            [
                Layer(15, 50, sigmoid),
                Layer(50, 30, sigmoid),
                Layer(30, 8, sigmoid_output),
            ],
            mutation_rate=rd.random(),
        ).random()
        for _ in range(100)
    ]

    plotter = FitnessPlotter()

    # Initialise world
    model, data, to_track = compile_world()

    max_gens = 20
    gen_iterator = tqdm(range(max_gens), desc="Generation")
    fitness = np.zeros((max_gens, len(population)))
    for gen in gen_iterator:
        for controller in tqdm(population, desc="Individual", leave=False):
            test_controller(controller, model, data, to_track)
            # show_qpos_history(controller.history)
        population.sort(key=lambda c: c.fitness(), reverse=True)
        gen_iterator.set_description_str(
            f"Highest fitness: {round(population[0].fitness(), 3)} -- Average fitness: {round(np.mean([c.fitness() for c in population]), 3)} --",
            refresh=False,
        )
        plotter.add([c.fitness() for c in population], gen)
        plotter.savefig()

        fitness[gen, :] = [c.fitness() for c in population]

        scaled_fitnesses = np.array(
            [c.fitness() - population[-1].fitness() for c in population]
        )
        scaled_fitnesses /= sum(scaled_fitnesses)

        next_gen = []
        for _ in range(round(len(population) / 4)):
            p1, p2 = rd.choices(population, weights=scaled_fitnesses, k=2)
            c1, c2 = p1.crossover(p2)
            c1.mutate()
            c2.mutate()
            next_gen.append(c1)
            next_gen.append(c2)

        population = population[: len(population) // 2]
        population.extend(next_gen)

    for controller in tqdm(population, desc="Individual", leave=False):
        test_controller(controller, model, data, to_track)

    population.sort(key=lambda c: c.fitness())
    save_brain(population[0])
    save_fitness(fitness, population)


def compile_world() -> tuple[Any, Any, Any]:
    """
    Return a flat world with gecko.

    :return: The compiled world model and the to_track.
    :rtype: tuple[Any, Any]
    """
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()  # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    return model, data, to_track


def test_controller(controller: Brain, model: Any, data: Any, to_track: Any):
    """
    A function to test gecko controllers.

    :param controller: The neural network to test.
    :type controller: Brain
    """
    # Set the control callback function
    # This is called every time step to get the next action.
    mujoco.set_mjcb_control(lambda m, d: controller.control(m, d, to_track))

    simple_runner(model, data, duration=10)
    # If you want to record a video of your simulation, you can use the video renderer.
    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    # viewer.launch(
    #     model=model,  # type: ignore
    #     data=data,
    # )


if __name__ == "__main__":
    main()
