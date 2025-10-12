import json
import os
from pathlib import Path
from typing import Any
import re
import networkx

import mujoco
from mujoco import MjData, viewer

from ariel.simulation.environments.olympic_arena import OlympicArena

from robots import (
    Robot,
    ViewerBody,
    BRAIN_TYPE_MAP,
)


def compile_world(robot: Robot) -> tuple[OlympicArena, Any, MjData]:
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.core.spec, position=[-0.8, 0, 0.1])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)
    return world, model, data


def experiment(
    robot: Robot,
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    world, model, data = compile_world(robot)

    # Pass the model and data to the tracker
    robot.controller.tracker.setup(world.spec, data)

    mujoco.set_mjcb_control(
        lambda m, d: robot.controller.set_control(m, d),  # type: ignore
    )

    # This opens a liver viewer of the simulation
    viewer.launch(
        model=model,
        data=data,
    )


def view(dir_path: Path, generation: int, individual_number: int) -> None:
    files = sorted(os.listdir(dir_path))
    if "nde.json" not in files:
        raise FileNotFoundError("No nde.json file found in folder.")
    gen_files = [f for f in files if re.match(r"^gen_\d{4}.json$", f)]
    print(f"Detected {len(gen_files)} generations.")
    if generation >= len(gen_files):
        raise ValueError(
            f"Given generation to high for given folder: {generation} > {len(gen_files)-1}"
        )

    # load right file
    with open(dir_path.joinpath(f"gen_{generation:04}.json"), "r") as file:
        gen_data = json.load(file)
    individual = gen_data[individual_number]
    print(
        f"Loaded individual {individual_number} with fitness {individual["fitness"]:.4}."
    )

    phenotype = json.loads(individual["body"]["phenotype"])

    robot_body = ViewerBody(networkx.node_link_graph(phenotype, edges="edges"))
    robot_brain = BRAIN_TYPE_MAP[individual["brain"]["type"]].from_dict(
        individual["brain"]
    )

    experiment(Robot(robot_body, robot_brain))


if __name__ == "__main__":
    view(Path("__data__/ea_run_2025_10_12_01:26:59"), 41, 2)
