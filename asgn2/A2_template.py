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

# Keep track of data / history
HISTORY = []

CROSSOVER_THRESHOLD = 0.5
MUTATION_THRESHOLD = 0.05

rng = np.random.default_rng()


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


class Brain:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.zeros(shape=(input_size, hidden_size))
        self.W2 = np.zeros(shape=(hidden_size, hidden_size))
        self.W3 = np.zeros(shape=(hidden_size, output_size))

        self.history: list[list[float]] = []

        self._fitness: None | float = None

    def random(self) -> Self:
        self.W1 = rng.standard_normal((self.input_size, self.hidden_size))
        self.W2 = rng.standard_normal((self.hidden_size, self.hidden_size))
        self.W3 = rng.standard_normal((self.hidden_size, self.output_size))
        return self

    def control(self, model: Any, data: Any, to_track: Any):
        # Get outputs, in this case the positions of the actuator motore (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network
        layer1 = sigmoid(np.dot(inputs, self.W1))
        layer2 = sigmoid(np.dot(layer1, self.W2))
        outputs = np.pi * (sigmoid(np.dot(layer2, self.W3)) - 0.5)

        # Scale the outputs to [-pi/2, pi/2]
        data.ctrl = np.clip(outputs, -np.pi / 2, np.pi / 2)

        # Save movement to history
        self.history.append(to_track[0].xpos.copy())

    def crossover(self, other: "Brain") -> list["Brain"]:
        """
        Create two children using crossover with `other`. Uses uniform crossover
        with a probability of `CROSSOVER_THRESHOLD`.

        :param self: Description
        :param other: The other parent
        :type other: "Brain"
        :return: A list containing the two children.
        :rtype: list[Brain]
        """
        left = Brain(self.input_size, self.hidden_size, self.output_size)
        right = Brain(self.input_size, self.hidden_size, self.output_size)

        P = CROSSOVER_THRESHOLD

        w1selection = rng.random(size=self.W1.shape)
        w2selection = rng.random(size=self.W2.shape)
        w3selection = rng.random(size=self.W3.shape)

        left.W1[w1selection > P] = self.W1[w1selection > P]
        left.W1[w1selection < P] = other.W1[w1selection < P]
        right.W1[w1selection < P] = self.W1[w1selection < P]
        right.W1[w1selection > P] = other.W1[w1selection > P]

        left.W2[w2selection > P] = self.W2[w2selection > P]
        left.W2[w2selection < P] = other.W2[w2selection < P]
        right.W2[w2selection < P] = self.W2[w2selection < P]
        right.W2[w2selection > P] = other.W2[w2selection > P]

        left.W3[w3selection > P] = self.W3[w3selection > P]
        left.W3[w3selection < P] = other.W3[w3selection < P]
        right.W3[w3selection < P] = self.W3[w3selection < P]
        right.W3[w3selection > P] = other.W3[w3selection > P]

        return [left, right]

    def mutate(self):
        P = MUTATION_THRESHOLD

        w1selection = rng.random(size=self.W1.shape) < P
        w2selection = rng.random(size=self.W2.shape) < P
        w3selection = rng.random(size=self.W3.shape) < P

        self.W1[w1selection] += rng.normal(scale=0.1, size=self.W1.shape)[w1selection]
        self.W2[w2selection] += rng.normal(scale=0.1, size=self.W2.shape)[w2selection]
        self.W3[w3selection] += rng.normal(scale=0.1, size=self.W3.shape)[w3selection]

    def fitness(self) -> float:
        if self._fitness:
            return self._fitness
        self._fitness = self.history[0][1] - self.history[-1][1]
        return self._fitness


def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    population = [Brain(15, 8, 8).random() for _ in range(100)]

    # Initialise world
    model, data, to_track = compile_world()

    max_gens = 20
    gen_iterator = tqdm(range(max_gens), desc="Generation")
    for gen in gen_iterator:
        for controller in tqdm(population, desc="Individual", leave=False):
            test_controller(controller, model, data, to_track)
            # show_qpos_history(controller.history)
        population.sort(key=lambda c: c.fitness(), reverse=True)
        gen_iterator.set_description_str(
            f"Highest fitness: {round(population[0].fitness(), 3)} -- Average fitness: {round(np.mean([c.fitness() for c in population]), 3)} --",
            refresh=False,
        )

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
