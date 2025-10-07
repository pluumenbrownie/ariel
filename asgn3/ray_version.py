from collections.abc import Iterator, Sequence
import json
from pathlib import Path
import time
import math as mt
from typing import Any
import os

import mujoco
import ray
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ariel.simulation.environments.olympic_arena import OlympicArena

# from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner

# from ariel.utils.video_recorder import VideoRecorder
from runners import complicated_runner
from rng import RNG
from robots import (
    # TYPE_MAP,
    Brain,
    TrainingBrain,
    RandomRobotBody,
    Robot,
    RobotBody,
    TestBrain,
    random_body_genotype,
)

from rich.traceback import install

install(width=180, show_locals=True)

ray.init(log_to_driver=False)


class EvolutionaryAlgorithmSettings:
    def __init__(self) -> None:
        self.processes = 8
        self.num_modules = 20
        self.genotype_size = 64
        self.body_generations = 256
        self.body_population_size = 100
        self.brain_generations = 10
        self.brain_population_size = 100
        # self.body_generations = 10
        # self.body_population_size = 8
        # self.brain_generations = 1
        # self.brain_population_size = 8

        self.body_survival_fraction = 0.1
        self.brain_survival_fraction = 0.1

        self.viewer = False
        self.spawn_position = [0, 0, 0.1]

        self.body_children = mt.floor(
            self.body_population_size * ((1 - self.body_survival_fraction) * 0.5)
        )
        self.body_keep = self.body_population_size - 2 * self.body_children

        self.brain_children = mt.floor(
            self.brain_population_size * ((1 - self.brain_survival_fraction) * 0.5)
        )
        self.brain_keep = self.brain_population_size - 2 * self.brain_children

        now = time.localtime()
        self.dir_name = Path(
            f"__data__/ea_run_"
            + f"{now.tm_year}_{now.tm_mon:02}_{now.tm_mday:02}_"
            + f"{now.tm_hour:02}:{now.tm_min:02}:{now.tm_sec:02}"
        )

        assert self.brain_population_size % 4 == 0, "Populations must be div. by 4."
        assert self.body_population_size % 4 == 0, "Populations must be div. by 4."
        assert self.body_keep + 2 * self.body_children == self.body_population_size


def run_random(
    settings: Any, parallel: bool = True
) -> tuple[tuple[RobotBody, Brain], float]:
    print(f"Started EA run ({parallel = })")
    # Create body population
    robot_bodies = generate_bodies(settings)

    fitness = np.zeros((settings.body_generations, settings.body_population_size))

    os.mkdir(settings.dir_name)

    best_bot = run_generations(
        settings,
        parallel,
        robot_bodies,
        fitness,
        range(settings.body_generations),
    )
    print(fitness)
    return best_bot


def run_generations(
    settings: Any,
    parallel: bool,
    robot_bodies: list[RobotBody],
    fitness: NDArray[np.float32],
    generations: Iterator[int],
) -> tuple[tuple[RobotBody, Brain], float]:
    for generation in generations:
        print(f"Gen {generation}")
        # Use multiprocessing to speed up computations

        bodies_fitness_refs = [
            evolve_brains.remote(settings, body) for body in robot_bodies
        ]
        bodies_fitness: list[tuple[tuple[RobotBody, Brain], float]] = []
        progress_bar = tqdm(total=settings.body_population_size)
        while len(bodies_fitness_refs) > 0:
            finished, bodies_fitness_refs = ray.wait(
                bodies_fitness_refs, timeout=60 * 60
            )
            data = ray.get(finished)
            progress_bar.update(len(data))
            bodies_fitness.extend(data)

        bodies_fitness.sort(key=fitness_key, reverse=True)
        best_robot = bodies_fitness[0]
        print(f"Best robot fitness: {best_robot[1]}")

        save_state(settings, generation, bodies_fitness)
        fitness[generation, :] = [r[1] for r in bodies_fitness]

        if generation == settings.body_generations - 1:
            return best_robot

        weights = linear_windowed_weights(bodies_fitness)
        next_gen = children_bodies(settings, bodies_fitness, weights)
        robot_bodies = next_gen

    raise ValueError("self.brain_generations must be at least 1.")


@ray.remote
def evolve_brains(
    settings: Any, robot_body: RobotBody
) -> tuple[tuple[RobotBody, Brain], float]:
    # The bodies get fresh new brains at the start of learning

    brains = generate_brains(settings, robot_body)

    best_brain: tuple[Brain, float]
    fitness = np.zeros((settings.brain_generations, settings.body_population_size))

    for generation in range(settings.brain_generations):
        brains_fitness: list[tuple[Brain, float]] = []

        itis = isinstance(robot_body, RandomRobotBody)
        assert itis, f"{type(robot_body) = }"

        for brain in brains:
            assert isinstance(brain, TrainingBrain), f"{type(brain) = }"
            robot = Robot(robot_body, brain)
            experiment(
                settings,
                robot=robot,
                mode="launcher" if settings.viewer else "complicated",
            )
            brains_fitness.append((brain, robot.fitness()))

        brains_fitness.sort(key=fitness_key, reverse=True)
        best_brain = brains_fitness[0]
        fitness[generation, :] = [pair[1] for pair in brains_fitness]

        # solves a type hinting problem
        if generation == settings.brain_generations - 1:
            return ((robot_body, best_brain[0]), best_brain[1])
        # Stop early if brain fitness is not changing
        # I think this is a good idea, well see
        if generation > 0:
            mean_fitness_change = np.mean(
                fitness[generation, :] - fitness[generation - 1, :]
            )
            if abs(mean_fitness_change) < 0.001:
                return ((robot_body, best_brain[0]), best_brain[1])

        weights = linear_windowed_weights(brains_fitness)

        next_gen = children_brains(settings, brains_fitness, weights)
        brains = next_gen

    raise ValueError("self.brain_generations must be at least 1.")


def experiment(
    settings: Any,
    robot: Robot,
    duration: int = 15,
    mode: str = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    world, model, data = compile_world(settings, robot)

    # Pass the model and data to the tracker
    if robot.controller.tracker is not None:
        robot.controller.tracker.setup(world.spec, data)

    mujoco.set_mjcb_control(
        lambda m, d: robot.controller.set_control(m, d),  # type: ignore
    )

    # ------------------------------------------------------------------ #
    match mode:  # type: ignore
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(model, data, duration)
        case "complicated":
            # No visualisation, with termination function
            complicated_runner(model, data, robot, termination_function, duration)


def generate_brains(settings: Any, robot_body: RobotBody) -> Sequence[Brain]:
    input_size, output_size = get_input_output_sizes(settings, robot_body)
    brains = [
        TrainingBrain(input_size, output_size).random()
        for _ in range(settings.brain_population_size)
    ]

    return brains


def generate_bodies(settings: Any) -> Sequence[RobotBody]:
    body_genotypes = [
        random_body_genotype(settings.genotype_size)
        for _ in range(settings.body_population_size)
    ]
    robot_bodies = [
        RandomRobotBody(body_genotype, settings.num_modules)
        for body_genotype in body_genotypes
    ]

    return robot_bodies


def children_brains(
    settings: Any,
    brains_fitness: list[tuple[Brain, float]],
    weights: NDArray[np.float32],
) -> list[Brain]:
    next_gen: list[Brain] = []
    for _ in range(settings.brain_children):
        choice = RNG.choices(brains_fitness, weights=weights, k=2)
        p1 = choice[0][0]
        p2 = choice[1][0]
        c1, c2 = p1.crossover(p2)
        c1.mutation()
        c2.mutation()
        next_gen.append(c1)
        next_gen.append(c2)

    next_gen.extend([c[0].copy() for c in brains_fitness[: settings.brain_keep]])
    return next_gen


def children_bodies(
    settings: Any,
    bodies_fitness: list[tuple[tuple[RobotBody, Brain], float]],
    weights: NDArray[np.float32],
) -> list[RobotBody]:
    next_gen: list[RobotBody] = []
    for _ in range(settings.body_children):
        choice = RNG.choices(bodies_fitness, weights=weights, k=2)

        p1: RobotBody = choice[0][0][0]
        p2: RobotBody = choice[1][0][0]
        c1, c2 = p1.crossover(p2)
        c1.mutation()
        c2.mutation()
        next_gen.append(c1)
        next_gen.append(c2)

    next_gen.extend([c[0][0].copy() for c in bodies_fitness[: settings.body_keep]])
    return next_gen


def compile_world(
    settings: Any, robot: Robot
) -> tuple[OlympicArena, Any, mujoco.MjData]:
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.core.spec, spawn_position=settings.spawn_position)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)
    return world, model, data


def get_input_output_sizes(settings: Any, robot_body: RobotBody) -> tuple[int, int]:
    """
    Create a MuJoCo world to determine the needed sizes of the input and
    output layers for a given robot body. Try to only run this once per body.

    :param self:
    :param robot_body: The body to determine the layer sizes for.
    :type robot_body: RobotBody
    :return: The input and output sizes
    :rtype: tuple[int, int]
    """
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    robot = Robot(robot_body, TestBrain())
    world, model, data = compile_world(settings, robot)

    if robot.controller.tracker is not None:
        robot.controller.tracker.setup(world.spec, data)

    input_size = len(data.qpos)
    output_size = model.nu
    return input_size, output_size


def linear_windowed_weights(fitness: list[tuple[Any, float]]) -> NDArray[np.float32]:
    weights = np.array([pair[1] - fitness[-1][1] for pair in fitness], dtype=np.float32)
    weights_total = np.sum(weights)
    if weights_total > 0:
        weights /= weights_total
    else:
        # fallback if all weights are equal
        weights = np.ones_like(weights) / np.float32(len(weights))
    return weights


def fitness_key(fitness_tuple: tuple[Any, float]) -> float:
    return fitness_tuple[1]


def termination_function(time: float, robot: Robot) -> bool:
    if robot.controller.tracker is not None:
        x = robot.controller.tracker.history["xpos"][0][-1][0]
        robot.controller.tracker.history["bonus"] = 0.0
        # Early culling of bad bots
        if x < 0.03 * time - 1 / (time + 1) + 0.2:
            return True
        # Early termination of fast bots, with fitness bonus
        if x > 5.0:
            robot.controller.tracker.history["bonus"] = max(time - 120.0, 0.0)
        return False
    else:
        raise ValueError("Robot controller not set.")


def save_state(
    settings: Any,
    generation: int,
    bodies_fitness: list[tuple[tuple[RobotBody, Brain], float]],
) -> None:
    generation_state = []
    for bot in bodies_fitness:
        bot_data = {}
        bot_data["body"] = bot[0][0].export()
        bot_data["brain"] = bot[0][1].export()
        bot_data["fitness"] = bot[1]
        generation_state.append(bot_data)
    with open(
        settings.dir_name.joinpath(Path(f"gen_{generation:04}.json")), "w"
    ) as file:
        file.writelines(json.dumps(generation_state, indent=2))


def main():
    settings = EvolutionaryAlgorithmSettings()
    run_random(settings)


if __name__ == "__main__":
    main()
