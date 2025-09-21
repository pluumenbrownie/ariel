import numpy as np
from typing import Any, Self

CROSSOVER_THRESHOLD = 0.5
MUTATION_THRESHOLD = 0.05

rng = np.random.default_rng()


class Layer:
    def __init__(self, input_size: int, output_size: int, function) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros(shape=(input_size, output_size))
        self.function = function

    def random(self) -> Self:
        self.weights = rng.standard_normal((self.input_size, self.output_size))
        return self

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.function(np.dot(inputs, self.weights))


class Brain:
    def __init__(self, layers: list[Layer], mutation_rate: float) -> None:
        self.layers = layers
        self.history: list[list[float]] = []
        self._fitness: None | float = None

    def random(self) -> Self:
        self.layers = [layer.random() for layer in self.layers]
        return self

    def control(self, model: Any, data: Any, to_track: Any):
        # Get outputs, in this case the positions of the actuator motore (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)

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
        left = self.copy()
        right = self.copy()

        P = CROSSOVER_THRESHOLD

        selection = []
        for l in self.layers:
            selection.append(rng.random(size=l.weights.shape))
        for i in range(len(self.layers)):
            left.layers[i].weights[selection[i] > P] = self.layers[i].weights[
                selection[i] > P
            ]
            left.layers[i].weights[selection[i] < P] = other.layers[i].weights[
                selection[i] < P
            ]
            right.layers[i].weights[selection[i] < P] = self.layers[i].weights[
                selection[i] < P
            ]
            right.layers[i].weights[selection[i] > P] = other.layers[i].weights[
                selection[i] > P
            ]

        return [left, right]

    def fitness(self) -> float:
        if self._fitness:
            return self._fitness
        self._fitness = self.history[0][1] - self.history[-1][1]
        return self._fitness

    def mutate(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def export(self) -> dict:
        raise NotImplementedError()


class UniformBrain(Brain):
    def __init__(self, layers: list[Layer], mutation_rate: float) -> None:
        super().__init__(layers, mutation_rate)

    def mutate(self):
        P = MUTATION_THRESHOLD

        for layer in self.layers:
            mutation_mask = rng.random(size=layer.weights.shape) < P
            layer.weights[mutation_mask] += rng.normal(
                scale=0.1, size=layer.weights.shape
            )[mutation_mask]

    def copy(self) -> "UniformBrain":
        return UniformBrain(
            [Layer(l.input_size, l.output_size, l.function) for l in self.layers], 0.0
        )

    def export(self) -> dict:
        return {
            "name": type(self).__name__,
            "layers": [
                {
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "weights": layer.weights.tolist(),
                    "function": layer.function.__name__,
                }
                for layer in self.layers
            ],
        }


class SelfAdaptiveBrain(Brain):
    """This is for a brain variant which uses self-adaptive mutation rates"""

    def __init__(self, layers: list[Layer], mutation_rate: float) -> None:
        super().__init__(layers, mutation_rate)
        self.mutation_rate = mutation_rate

    def mutate(self):
        n = sum(layer.weights.size for layer in self.layers)

        tau = 1 / np.sqrt(n)
        self.mutation_rate *= np.exp(rng.normal(scale=tau))

        P = MUTATION_THRESHOLD

        for layer in self.layers:
            mutation_mask = rng.random(size=layer.weights.shape) < P
            layer.weights[mutation_mask] += rng.normal(
                scale=self.mutation_rate, size=layer.weights.shape
            )[mutation_mask]

    def copy(self) -> "SelfAdaptiveBrain":
        return SelfAdaptiveBrain(
            [Layer(l.input_size, l.output_size, l.function) for l in self.layers],
            self.mutation_rate,
        )

    def export(self) -> dict:
        return {
            "name": type(self).__name__,
            "mutation_rate": self.mutation_rate,
            "layers": [
                {
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "weights": layer.weights.tolist(),
                    "function": layer.function.__name__,
                }
                for layer in self.layers
            ],
        }


""" here we reuse the given random_move function and then create a NoBrain baseline which its controller calls"""

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
    to_track:
        The locations of the parts of the robot used as inputs for
        a neural network, if the robot has one.

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


class NoBrain(Brain):
    """A brain controlled by random movements only - serves as our baseline"""

    def __init__(self, layers: list[Layer], mutation_rate: float) -> None:
        super().__init__(layers=[], mutation_rate=mutation_rate)

    def random(self) -> Self:
        return self

    def control(self, model: Any, data: Any, to_track: Any):
        # for the controller we utilize the given random_move() function
        random_move(model, data, to_track)
        # Keep track of movement history, using the same data as the global HISTORY
        self.history.append(to_track[0].xpos.copy())

    def fitness(self) -> float:
        if self._fitness is not None:
            return self._fitness
        if len(self.history) < 2:
            self._fitness = 0.0
        else:
            self._fitness = self.history[0][1] - self.history[-1][1]
        return self._fitness

    def mutate(self):
        # no mutation possible
        pass

    def copy(self) -> "NoBrain":
        return NoBrain([], 0.0)

    def export(self) -> dict:
        return {
            "name": type(self).__name__,
            "layers": [],
            "description": "Random baseline using random_move",
        }
