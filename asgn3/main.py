import json
import os
from pathlib import Path
import time
from typing import Any
from collections.abc import Iterator, Sequence
from multiprocessing import Pool
import math as mt
import re

from tqdm import tqdm
import mujoco
from mujoco import MjData, viewer
import numpy as np
from numpy.typing import NDArray
import torch

from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    save_graph_as_json,
)

from plotters import LivePlotter
from runners import complicated_runner
from rng import RNG
from robots import (
    TYPE_MAP,
    Brain,
    RandomBrain,
    SelfAdaptiveBrain,
    TrainingBrain,
    RandomRobotBody,
    Robot,
    RobotBody,
    TestBrain,
    random_body_genotype,
)

from rich.traceback import install

install(width=180, show_locals=False)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


class EvolutionaryAlgorithm:
    def __init__(self) -> None:
        self.processes = 12
        self.num_modules = 20
        self.genotype_size = 64
        self.body_generations = 2
        self.body_population_size = 100
        self.brain_generations = 5
        self.brain_population_size = 100
        # self.body_generations = 1
        # self.body_population_size = 8
        # self.brain_generations = 1
        # self.brain_population_size = 8

        self.body_survival_fraction = 0.0
        self.brain_survival_fraction = 0.01

        self.viewer = False
        self.spawn_position = [-0.8, 0, 0.1]

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

        self.nde = NeuralDevelopmentalEncoding(self.num_modules)

        assert self.brain_population_size % 4 == 0, "Populations must be div. by 4."
        assert self.body_population_size % 4 == 0, "Populations must be div. by 4."
        assert self.body_keep + 2 * self.body_children == self.body_population_size

    def run_random(
        self, parallel: bool = True
    ) -> tuple[tuple[RobotBody, Brain], float]:
        print(f"Started EA run ({parallel = })")
        # Create body population
        robot_bodies = self.generate_bodies_preselect()

        fitness = np.zeros((self.body_generations, self.body_population_size))
        plotter = LivePlotter(fitness, self.dir_name)

        os.mkdir(self.dir_name)
        with open(self.dir_name.joinpath("nde.json"), "w") as file:
            output = export_nde(self.nde)
            json.dump(output, file)

        best_bot = self.run_generations(
            parallel, robot_bodies, fitness, range(self.body_generations), plotter
        )
        print(fitness)

        return best_bot

    def resume(
        self, path: Path, override: bool = True, parallel: bool = True
    ) -> tuple[tuple[RobotBody, Brain], float]:
        if override:
            self.dir_name = path

        files = sorted(os.listdir(path))
        assert "nde.json" in files, "No nde.json file found in folder."
        gen_files = [f for f in files if re.match(r"^gen_\d{4}.json$", f)]
        print(f"Detected {len(gen_files)} generations.")

        with open(path.joinpath("nde.json"), "r") as file:
            self.nde = import_nde(json.load(file))
        fitness = self.load_fitness(path, gen_files)
        plotter = LivePlotter(fitness, self.dir_name)
        bodies_fitness = self.load_bodies(path.joinpath(gen_files[-1]))

        weights = self.linear_windowed_weights(bodies_fitness)
        robot_bodies = self.children_bodies(
            [((body, ()), fit) for body, fit in bodies_fitness], weights
        )

        best_bot = self.run_generations(
            parallel,
            robot_bodies,
            fitness,
            range(len(gen_files), self.body_generations),
            plotter,
        )
        print(fitness)

        return best_bot

    def run_single_brain(
        self, path: Path, parallel: bool = True
    ) -> tuple[tuple[RobotBody, Brain], float]:
        self.dir_name = path

        files = sorted(os.listdir(path))
        assert "nde.json" in files, "No nde.json file found in folder."
        gen_files = [f for f in files if re.match(r"^gen_\d{4}.json$", f)]

        with open(path.joinpath("nde.json"), "r") as file:
            self.nde = import_nde(json.load(file))
        bodies_fitness = self.load_bodies(path.joinpath(gen_files[-1]))
        best_body = bodies_fitness[1][0]
        fitness = np.zeros((self.brain_generations, self.brain_population_size))
        best_bot = self.evolve_brains(best_body, fitness=fitness)

        plotter = LivePlotter(fitness, self.dir_name)
        plotter.plot()

        print(fitness)

        return best_bot

    def run_generations(
        self,
        parallel: bool,
        robot_bodies: list[RobotBody],
        fitness: NDArray[np.float32],
        generations: Iterator[int],
        plotter: LivePlotter,
    ) -> tuple[tuple[RobotBody, Brain], float]:
        for generation in generations:
            print(f"Gen {generation}")
            # Use multiprocessing to speed up computations
            if parallel:
                with Pool(processes=self.processes) as pool:
                    bodies_fitness = list(
                        tqdm(
                            pool.imap_unordered(self.evolve_brains, robot_bodies),
                            total=self.body_population_size,
                        )
                    )
            else:
                bodies_fitness = list(
                    tqdm(
                        map(self.evolve_brains, robot_bodies),
                        total=self.body_population_size,
                    )
                )

            bodies_fitness.sort(key=fitness_key, reverse=True)
            best_robot = bodies_fitness[0]
            print(f"Best robot fitness: {best_robot[1]}")

            self.save_state(generation, bodies_fitness)
            fitness[generation, :] = [r[1] for r in bodies_fitness]
            plotter.plot()

            if generation == self.body_generations - 1:
                return best_robot

            weights = self.exponential_ranking_weights(bodies_fitness)
            next_gen = self.children_bodies(bodies_fitness, weights)
            robot_bodies = next_gen

        raise ValueError("self.brain_generations must be at least 1.")

    def evolve_brains(
        self,
        robot_body: RobotBody,
        fitness: NDArray[np.float32] | None = None,
    ) -> tuple[tuple[RobotBody, Brain], float]:
        # The bodies get fresh new brains at the start of learning

        brains = self.generate_brains(robot_body)

        best_brain: tuple[Brain, float]
        if fitness is None:
            fitness = np.zeros(
                (self.brain_generations, self.brain_population_size), dtype=np.float32
            )

        for generation in range(self.brain_generations):
            brains_fitness: list[tuple[Brain, float]] = []

            for brain in brains:
                robot = Robot(robot_body, brain)
                self.experiment(robot=robot)
                brains_fitness.append((brain, robot.fitness()))

            brains_fitness.sort(key=fitness_key, reverse=True)
            best_brain = brains_fitness[0]
            fitness[generation, :] = [pair[1] for pair in brains_fitness]

            # solves a type hinting problem
            if generation == self.brain_generations - 1:
                return ((robot_body, best_brain[0]), best_brain[1])
            # Stop early if brain fitness is not changing
            # I think this is a good idea, well see
            if generation > 4:
                last_five_fitness = np.average(
                    fitness[generation - 4 : generation, :], axis=1
                )
                largest_fitness_change = np.max(np.abs(np.diff(last_five_fitness)))
                if largest_fitness_change < 0.0005:
                    return ((robot_body, best_brain[0]), best_brain[1])

            weights = self.exponential_ranking_weights(brains_fitness)

            next_gen = self.children_brains(brains_fitness, weights)
            brains = next_gen

        raise ValueError("self.brain_generations must be at least 1.")

    def save_state(
        self,
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
            self.dir_name.joinpath(Path(f"gen_{generation:04}.json")), "w"
        ) as file:
            file.writelines(json.dumps(generation_state, indent=2))

    def children_brains(
        self,
        brains_fitness: list[tuple[Brain, float]],
        weights: NDArray[np.float32],
    ) -> list[Brain]:
        next_gen: list[Brain] = []
        for _ in range(self.brain_children):
            choice = RNG.choices(brains_fitness, weights=weights, k=2)
            p1 = choice[0][0]
            p2 = choice[1][0]
            # c1, c2 = p1.crossover(p2)
            c1 = p1.copy()
            c2 = p2.copy()
            c1.mutation()
            c2.mutation()
            next_gen.append(c1)
            next_gen.append(c2)

        next_gen.extend([c[0].copy() for c in brains_fitness[: self.brain_keep]])
        return next_gen

    def children_bodies(
        self,
        bodies_fitness: list[tuple[tuple[RobotBody, Brain], float]],
        weights: NDArray[np.float32],
    ) -> list[RobotBody]:
        next_gen: list[RobotBody] = []
        for _ in range(self.body_children):
            choice = RNG.choices(bodies_fitness, weights=weights, k=2)

            p1: RobotBody = choice[0][0][0]
            p2: RobotBody = choice[1][0][0]
            c1, c2 = p1.crossover(p2)
            c1.mutation()
            c2.mutation()
            next_gen.append(c1)
            next_gen.append(c2)

        next_gen.extend([c[0][0].copy() for c in bodies_fitness[: self.body_keep]])
        return next_gen

    def generate_brains(self, robot_body: RobotBody) -> Sequence[Brain]:
        input_size, output_size = self.get_input_output_sizes(robot_body)
        brains = [
            SelfAdaptiveBrain(input_size, output_size).random()
            for _ in range(self.brain_population_size)
        ]

        return brains

    def generate_bodies(self) -> Sequence[RobotBody]:
        body_genotypes = [
            random_body_genotype(self.genotype_size)
            for _ in range(self.body_population_size)
        ]
        robot_bodies = [
            RandomRobotBody(body_genotype, self.num_modules, self.nde)
            for body_genotype in body_genotypes
        ]

        return robot_bodies

    def generate_bodies_preselect(self) -> Sequence[RobotBody]:
        print("Robot preselection")
        progress_bar = tqdm(total=self.body_population_size)
        body_genotypes = []
        while len(body_genotypes) < self.body_population_size:
            genotype = random_body_genotype(self.genotype_size)
            body = RandomRobotBody(genotype, self.num_modules, self.nde)
            input_size, output_size = self.get_input_output_sizes(body)
            robot = Robot(body, RandomBrain(input_size, output_size))
            self.experiment(robot, duration=3, mode="complicated")
            if robot.fitness() >= 0.1:
                body_genotypes.append(genotype)
                progress_bar.update()

        robot_bodies = [
            RandomRobotBody(body_genotype, self.num_modules, self.nde)
            for body_genotype in body_genotypes
        ]

        return robot_bodies

    def experiment(
        self,
        robot: Robot,
        duration: int = 15,
        mode: str = "complicated",
    ) -> None:
        """Run the simulation with random movements."""
        # ==================================================================== #
        # Initialise controller to controller to None, always in the beginning.
        mujoco.set_mjcb_control(None)  # DO NOT REMOVE

        world, model, data = self.compile_world(robot)

        # Pass the model and data to the tracker
        if robot.controller.tracker is not None:
            robot.controller.tracker.setup(world.spec, data)

        # Set the control callback function
        # This is called every time step to get the next action.
        args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
        kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

        mujoco.set_mjcb_control(
            lambda m, d: robot.controller.set_control(m, d, *args, **kwargs),  # type: ignore
        )

        # ------------------------------------------------------------------ #
        match mode:  # type: ignore
            case "simple":
                # This disables visualisation (fastest option)
                simple_runner(model, data, duration)
            case "complicated":
                # No visualisation, with termination function
                complicated_runner(model, data, robot, termination_function, duration)
            case "frame":
                # Render a single frame (for debugging)
                save_path = str(DATA / "robot.png")
                single_frame_renderer(model, data, save=True, save_path=save_path)
            case "video":
                # This records a video of the simulation
                path_to_video_folder = str(DATA / "videos")
                video_recorder = VideoRecorder(output_folder=path_to_video_folder)

                # Render with video recorder
                video_renderer(
                    model,
                    data,
                    duration=duration,
                    video_recorder=video_recorder,
                )
            case "launcher":
                # This opens a liver viewer of the simulation
                viewer.launch(
                    model=model,
                    data=data,
                )
            case "no_control":
                # If mujoco.set_mjcb_control(None), you can control the limbs manually.
                mujoco.set_mjcb_control(None)
                viewer.launch(
                    model=model,
                    data=data,
                )

    def compile_world(self, robot: Robot) -> tuple[OlympicArena, Any, MjData]:
        world = OlympicArena()

        # Spawn robot in the world
        # Check docstring for spawn conditions
        world.spawn(robot.core.spec, spawn_position=self.spawn_position)

        # Generate the model and data
        # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)
        return world, model, data

    def get_input_output_sizes(self, robot_body: RobotBody) -> tuple[int, int]:
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
        world, model, data = self.compile_world(robot)

        if robot.controller.tracker is not None:
            robot.controller.tracker.setup(world.spec, data)

        input_size = len(data.qpos)
        output_size = model.nu
        return input_size, output_size

    def linear_windowed_weights(
        self, fitness: list[tuple[Any, float]]
    ) -> NDArray[np.float32]:
        weights = np.array(
            [pair[1] - fitness[-1][1] for pair in fitness], dtype=np.float32
        )
        weights_total = np.sum(weights)
        if weights_total > 0:
            weights /= weights_total
        else:
            # fallback if all weights are equal
            weights = np.ones_like(weights) / np.float32(len(weights))
        return weights

    def exponential_ranking_weights(
        self, fitness: list[tuple[Any, float]]
    ) -> NDArray[np.float32]:
        weights = np.array(
            [1 - np.e ** (-rank) for rank in np.arange(len(fitness) - 1, -1, -1)],
            dtype=np.float32,
        )

        weights_total = np.sum(weights)
        weights /= weights_total
        return weights

    def load_fitness(self, directory: Path, files: list[str]) -> NDArray[np.float32]:
        fitness = np.zeros(
            (self.body_generations, self.body_population_size), dtype=np.float32
        )
        for nr, file in enumerate(files):
            with open(directory.joinpath(file), "r") as file:
                data: list[dict[str, Any]] = json.load(file)
                fitness[nr] = [indiv["fitness"] for indiv in data]

        return fitness

    def load_bodies(self, path: Path) -> list[tuple[RobotBody, float]]:
        robot_bodies = []
        with open(path, "r") as file:
            data = json.load(file)
        for individual in data:
            robot_type = TYPE_MAP[individual["body"]["type"]]
            genotype_data = individual["body"]["genotype"]
            genotype = [np.array(lst) for lst in genotype_data]
            num_modules = individual["body"]["num_modules"]

            fitness = individual["fitness"]
            body = robot_type(genotype, num_modules, self.nde)
            robot_bodies.append((body, fitness))

        return robot_bodies
    
    '''
      BASELINE experiment: random search for non-evolution of brain and body
    '''
    
    def run_baseline(
        self, parallel: bool = True
    ) -> tuple[tuple[RobotBody, Brain], float]:
        """
        Baseline experiment: Pure random search with no evolution.
        Each generation gets completely new random bodies and random brains.
        """
        print(f"Started Baseline run (random search, {parallel = })")
    
        fitness = np.zeros((self.body_generations, self.body_population_size))

        '''to create separate directory for baseline results we do:
        now = time.localtime()
        baseline_dir = Path(
            f"__data__/baseline_run_"
            + f"{now.tm_year}_{now.tm_mon:02}_{now.tm_mday:02}_"
            + f"{now.tm_hour:02}:{now.tm_min:02}:{now.tm_sec:02}"
        )'''
        baseline_dir = self.dir_name / "baseline"
        os.mkdir(baseline_dir)
    
        plotter = LivePlotter(fitness, baseline_dir)

        # Save NDE for reproducibility
        with open(baseline_dir.joinpath("nde.json"), "w") as file:
            output = export_nde(self.nde)
            json.dump(output, file)
    
        best_overall: tuple[tuple[RobotBody, Brain], float] | None = None

        for generation in range(self.body_generations):
            print(f"Baseline Gen {generation}")
        
            # Generate completely NEW random bodies each generation (no evolution)
            robot_bodies = self.generate_bodies_preselect()
        
            # Evaluate each body with a random brain (no learning)
            if parallel:
                with Pool(processes=self.processes) as pool:
                    bodies_fitness = list(
                        tqdm(
                            pool.imap_unordered(self.evaluate_random_brain, robot_bodies),
                            total=self.body_population_size,
                        )
                    )
            else:
                bodies_fitness = list(
                    tqdm(
                        map(self.evaluate_random_brain, robot_bodies),
                        total=self.body_population_size,
                    )
                )

            # Sort to find best of this generation
            bodies_fitness.sort(key=fitness_key, reverse=True)
            best_robot = bodies_fitness[0]
            print(f"Best baseline robot fitness: {best_robot[1]}")
        
            # Track best overall across all generations
            if best_overall is None or best_robot[1] > best_overall[1]:
                best_overall = best_robot
        
            # Save state
            self.save_state_baseline(generation, bodies_fitness, baseline_dir)
            fitness[generation, :] = [r[1] for r in bodies_fitness]
            plotter.plot()   

        assert best_overall is not None, "No generations were run"   #typecheck!
        print(f"Best overall baseline fitness: {best_overall[1]}")
        print(fitness)
    
        return best_overall


    def evaluate_random_brain(
        self,
        robot_body: RobotBody,
    ) -> tuple[tuple[RobotBody, Brain], float]:
        input_size, output_size = self.get_input_output_sizes(robot_body)
        # create a random 'brain' for the robot using Class RandomBrain
        brain = RandomBrain(input_size, output_size)
        robot = Robot(robot_body, brain)
        self.experiment(robot=robot)
        return ((robot_body, brain), robot.fitness())
    
    def save_state_baseline(
        self,
        generation: int,
        bodies_fitness: list[tuple[tuple[RobotBody, Brain], float]],
        directory: Path,
    ) -> None:
        generation_state = []
        for bot in bodies_fitness:
            bot_data = {}
            bot_data["body"] = bot[0][0].export()
            bot_data["brain"] = bot[0][1].export()
            bot_data["fitness"] = bot[1]
            generation_state.append(bot_data)
        with open(
            directory.joinpath(Path(f"gen_{generation:04}.json")), "w"
        ) as file:
            file.writelines(json.dumps(generation_state, indent=2))
    



def termination_function(time: float, robot: Robot) -> bool:
    if robot.controller.tracker is not None:
        x_start = robot.controller.tracker.history["xpos"][0][0][0]
        x = robot.controller.tracker.history["xpos"][0][-1][0]
        dx = x - x_start
        robot.controller.tracker.history["bonus"] = 0.0
        # Early culling of bad bots
        if dx < 0.03 * time - 1 / (time + 1) + 0.2:
            return True
        # Early termination of fast bots, with fitness bonus
        if dx > 5.0:
            robot.controller.tracker.history["bonus"] = max(time - 120.0, 0.0)
        return False
    else:
        raise ValueError("Robot controller not set.")


def fitness_key(fitness_tuple: tuple[Any, float]) -> float:
    return fitness_tuple[1]


def export_nde(nde: NeuralDevelopmentalEncoding) -> dict[str, Any]:
    return {
        "number_of_modules": nde.type_p_shape[0],
        "fc1": {
            "weight": nde.fc1.weight.data.tolist(),
            "bias": nde.fc1.bias.data.tolist(),
        },
        "fc2": {
            "weight": nde.fc2.weight.data.tolist(),
            "bias": nde.fc2.bias.data.tolist(),
        },
        "fc3": {
            "weight": nde.fc3.weight.data.tolist(),
            "bias": nde.fc3.bias.data.tolist(),
        },
        "fc4": {
            "weight": nde.fc4.weight.data.tolist(),
            "bias": nde.fc4.bias.data.tolist(),
        },
        "type_p_out": {
            "weight": nde.type_p_out.weight.data.tolist(),
            "bias": nde.type_p_out.bias.data.tolist(),
        },
        "conn_p_out": {
            "weight": nde.conn_p_out.weight.data.tolist(),
            "bias": nde.conn_p_out.bias.data.tolist(),
        },
        "rot_p_out": {
            "weight": nde.rot_p_out.weight.data.tolist(),
            "bias": nde.rot_p_out.bias.data.tolist(),
        },
    }


def import_nde(data: dict[str, Any]) -> NeuralDevelopmentalEncoding:
    nde = NeuralDevelopmentalEncoding(data["number_of_modules"])
    nde.fc1.weight.data = torch.Tensor(data["fc1"]["weight"])
    nde.fc1.bias.data = torch.Tensor(data["fc1"]["bias"])
    nde.fc2.weight.data = torch.Tensor(data["fc2"]["weight"])
    nde.fc2.bias.data = torch.Tensor(data["fc2"]["bias"])
    nde.fc3.weight.data = torch.Tensor(data["fc3"]["weight"])
    nde.fc3.bias.data = torch.Tensor(data["fc3"]["bias"])
    nde.fc4.weight.data = torch.Tensor(data["fc4"]["weight"])
    nde.fc4.bias.data = torch.Tensor(data["fc4"]["bias"])
    nde.type_p_out.weight.data = torch.Tensor(data["type_p_out"]["weight"])
    nde.type_p_out.bias.data = torch.Tensor(data["type_p_out"]["bias"])
    nde.conn_p_out.weight.data = torch.Tensor(data["conn_p_out"]["weight"])
    nde.conn_p_out.bias.data = torch.Tensor(data["conn_p_out"]["bias"])
    nde.rot_p_out.weight.data = torch.Tensor(data["rot_p_out"]["weight"])
    nde.rot_p_out.bias.data = torch.Tensor(data["rot_p_out"]["bias"])

    return nde


def main():
    ea = EvolutionaryAlgorithm()
    best_robot = ea.run_random(parallel=True)
    robot = best_robot[0]
    save_graph_as_json(robot[0].robot_graph, DATA / "robot_graph.json")
    json_data = json.dumps(robot[1].export(), indent=4)
    with Path(DATA / "brain.json").open("w", encoding="utf-8") as f:
        f.write(json_data)
    # ea.resume(Path("__data__/ea_run_2025_10_08_18:23:14"))
    # ea.run_single_brain(Path("asgn3/example_results"))


if __name__ == "__main__":
    main()
