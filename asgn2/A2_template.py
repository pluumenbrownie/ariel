# Third-party libraries
from typing import Any, Self, Type
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import matplotlib
import random as rd
from enum import Enum

from tqdm import tqdm

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.runners import simple_runner
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from Neural_Net import Brain, Layer, UniformBrain, SelfAdaptiveBrain, NoBrain
from plotters import FitnessPlotter

import json

# Keep track of data / history
HISTORY = []


class BrainType(Enum):
    ADAPTIVE = SelfAdaptiveBrain
    UNIFORM = UniformBrain
    NOBRAIN = NoBrain


def save_brain(
    brain: Brain,
    save_dir: str = "__data__",
    postfix: str = "",
) -> None:
    """Save the brain to a file."""
    with open(
        f"{save_dir}/{type(brain).__name__}{"_" if postfix else ""}{postfix}.json", "w"
    ) as f:
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
        elif data["name"] == "NoBrain":
            return NoBrain()
        else:
            raise ValueError(f"{data["name"]}: Invalid brain type.")


def save_fitness(
    weights: np.ndarray,
    brain: Brain,
    save_dir: str = "__data__",
    postfix: str = "",
) -> None:
    """Save the fitness values to a file."""
    np.save(
        f"{save_dir}/{type(brain).__name__}_fitness{"_" if postfix else ""}{postfix}.npy",
        weights,
    )


def load_fitness(filename: str) -> np.ndarray:
    """Load the fitness values from a file."""
    return np.load(filename)


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


def main(
    brain_type: BrainType,
    pop_size: int,
    max_gens: int,
    save_dir: str = "__data__",
    postfix: str = "",
):
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    brain = brain_type.value
    population = [
        brain(
            [
                Layer(15, 50, sigmoid),
                Layer(50, 30, sigmoid),
                Layer(30, 8, sigmoid_output),
            ],
            mutation_rate=rd.random(),
        ).random()
        for _ in range(pop_size)
    ]

    plotter = FitnessPlotter()

    # Initialise world
    model, data, to_track = compile_world()

    gen_iterator = tqdm(range(max_gens), desc="Generation")
    fitness = np.zeros((max_gens, len(population)))
    best_brain = population[0]
    for gen in gen_iterator:
        for controller in tqdm(population, desc="Individual", leave=False):
            test_controller(controller, model, data, to_track)
            # show_qpos_history(controller.history)
        population.sort(key=lambda c: c.fitness(), reverse=True)
        best_brain = population[0]
        gen_iterator.set_description_str(
            f"Highest fitness: {round(population[0].fitness(), 3)} -- Average fitness: {round(np.mean([c.fitness() for c in population]), 3)} --",
            refresh=False,
        )
        plotter.add([c.fitness() for c in population], gen)
        plotter.savefig(save_dir=save_dir, postfix=postfix)

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

        next_gen.extend([c.copy() for c in population[: len(population) // 2]])
        population = next_gen

    save_brain(best_brain, save_dir=save_dir, postfix=postfix)
    save_fitness(fitness, best_brain, save_dir=save_dir, postfix=postfix)
    plotter.savedata(save_dir=save_dir, postfix=postfix)
    print(fitness)


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
    main(SelfAdaptiveBrain, 100, 20)
