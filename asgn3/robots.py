from copy import deepcopy
from typing import Any, Self
from collections.abc import Callable
import mujoco
from networkx import DiGraph
import numpy as np
from numpy.typing import NDArray
from rich.traceback import install

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker

from rng import NP_RNG


install(width=180, show_locals=False)


class RobotBody:
    def __init__(
        self, body_genotype: list[NDArray[np.float32]], num_modules: int
    ) -> None:
        self.genotype = body_genotype
        self.num_modules = num_modules

    def copy(self) -> "RobotBody":
        return type(self)(self.genotype.copy(), self.num_modules)

    def mutation(self) -> Self:
        P = 0.05

        for vec in self.genotype:
            selection = NP_RNG.random(vec.shape) <= P
            vec[selection] += NP_RNG.normal(0, 1, size=vec.shape)[selection]

        return self

    def crossover(self, other: "RobotBody") -> list["RobotBody"]:
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

        P = 0.5

        selection = []
        for vec in self.genotype:
            selection.append(NP_RNG.random(size=vec.shape))
        for i in range(len(self.genotype)):
            left.genotype[i][selection[i] > P] = self.genotype[i][selection[i] > P]
            left.genotype[i][selection[i] <= P] = other.genotype[i][selection[i] <= P]
            right.genotype[i][selection[i] <= P] = self.genotype[i][selection[i] <= P]
            right.genotype[i][selection[i] > P] = other.genotype[i][selection[i] > P]

        return [left, right]

    def export(self) -> dict[str, Any]:
        return {
            "type": str(type(self).__name__),
            "genotype": [vec.tolist() for vec in self.genotype],
            "num_modules": self.num_modules,
        }


class RandomRobotBody(RobotBody):
    def __init__(
        self, body_genotype: list[NDArray[np.float32]], num_modules: int
    ) -> None:
        super().__init__(body_genotype, num_modules)
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        p_matrices = nde.forward(body_genotype)

        hpd = HighProbabilityDecoder(num_modules)
        self.robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        function: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros(shape=(input_size, output_size), dtype=np.float32)
        self.function = function

    def random(self) -> Self:
        self.weights = NP_RNG.standard_normal((self.input_size, self.output_size))
        return self

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.function(np.dot(inputs, self.weights))

    def export(self) -> list[list[float]]:
        return self.weights.tolist()

    def __repr__(self) -> str:
        return f"Layer({self.input_size}, {self.output_size})"


class Brain:
    def __init__(self, input_size: int, output_size: int) -> None:
        pass

    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Any:
        pass

    def mutation(self) -> "Brain":
        return self

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
        return [self, other]

    def copy(self) -> "Brain":
        raise NotImplementedError()

    def export(self) -> dict[str, Any]:
        """Fyi RandomBrain doesnt work with this yet."""
        return {
            "type": str(type(self).__name__),
            "genotype": [layer.export() for layer in self.layers],
        }


class TestBrain(Brain):
    """Class to determine input and output layer sizes for robot controllers."""

    def __init__(self) -> None:
        self.input_size = None
        self.output_size = None

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        self.input_size = len(data.qpos)
        self.output_size = model.nu
        return super().__call__(model, data)


class RandomBrain(Brain):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        hidden_size = 8

        # Initialize the networks weights randomly
        # Normally, you would use the genes of an individual as the weights,
        # Here we set them randomly for simplicity.
        self.w1 = NP_RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
        self.w2 = NP_RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
        self.w3 = NP_RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Any:
        # Get inputs, in this case the positions of the actuator motors (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, self.w1))
        layer2 = np.tanh(np.dot(layer1, self.w2))
        outputs = np.tanh(np.dot(layer2, self.w3))

        # Scale the outputs
        return outputs * np.pi


class TrainingBrain(Brain):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)

        self.layers = [
            Layer(input_size, 50, np.tanh),
            Layer(50, 30, np.tanh),
            Layer(30, output_size, np.tanh),
        ]

    def random(self) -> Self:
        self.layers = [layer.random() for layer in self.layers]
        return self

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        inputs = data.qpos

        results = inputs
        for layer in self.layers:
            results = layer.forward(results)
        return results

    def mutation(self) -> Self:
        P = 0.05

        for layer in self.layers:
            mutation_mask = NP_RNG.random(size=layer.weights.shape) <= P
            layer.weights[mutation_mask] += NP_RNG.normal(
                scale=0.1, size=layer.weights.shape
            )[mutation_mask]

        return self

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

        P = 0.5

        selection = []
        for l in self.layers:
            selection.append(NP_RNG.random(size=l.weights.shape))
        for i in range(len(self.layers)):
            left.layers[i].weights[selection[i] > P] = self.layers[i].weights[
                selection[i] > P
            ]
            left.layers[i].weights[selection[i] <= P] = other.layers[i].weights[
                selection[i] <= P
            ]
            right.layers[i].weights[selection[i] <= P] = self.layers[i].weights[
                selection[i] <= P
            ]
            right.layers[i].weights[selection[i] > P] = other.layers[i].weights[
                selection[i] > P
            ]

        return [left, right]

    def copy(self) -> Brain:
        new = TrainingBrain(self.layers[0].input_size, self.layers[-1].output_size)
        new.layers = [deepcopy(layer) for layer in self.layers]
        return new


class Robot:
    """
    A combination of a RobotBody and a Brain. Robots can only be used once in
    a MuJoCo simulation.
    """

    def __init__(self, body: RobotBody, brain: Brain) -> None:
        self.body = body
        self.brain = brain
        self.core = construct_mjspec_from_graph(body.robot_graph)
        self.tracker = self.get_tracker()
        self.controller = Controller(
            controller_callback_function=brain,
            tracker=self.tracker,
        )

    def get_tracker(self):
        mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
        name_to_bind = "core"
        tracker = Tracker(
            mujoco_obj_to_find=mujoco_type_to_find,
            name_to_bind=name_to_bind,
        )
        return tracker

    def fitness(self) -> float:
        x = self.tracker.history["xpos"][0][-1][0]
        x = abs(x)
        y = self.tracker.history["xpos"][0][-1][1]
        bonus = self.tracker.history["bonus"]
        return y - x + bonus


def random_body_genotype(genotype_size: int) -> list[NDArray[np.float32]]:
    type_p_genes = NP_RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = NP_RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = NP_RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    return genotype


TYPE_MAP = {
    "RandomRobotBody": RandomRobotBody,
}
