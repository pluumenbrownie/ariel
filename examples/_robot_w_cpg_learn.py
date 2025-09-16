"""TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-07-08
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     In progress ⚙️

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ] documentation

"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import mujoco
import numpy as np
from cmaes import CMA
from mujoco import viewer
from rich.console import Console

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (NUM_OF_FACES,
                                                       NUM_OF_ROTATIONS,
                                                       NUM_OF_TYPES_OF_MODULES)
from ariel.body_phenotypes.robogen_lite.constructor import \
    construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder, save_graph_as_json)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.cpg_with_sensory_feedback import \
    CPGSensoryFeedback
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.runners import simple_runner

if TYPE_CHECKING:
    from networkx import Graph

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 41

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def main() -> None:
    """Entry point."""
    # System parameters
    num_modules = 30

    # Generate the probability space for the robot modules
    type_p, conn_p, rot_p = generate_prob_space(num_modules)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    graph: Graph[Any] = hpd.probability_matrices_to_graph(type_p, conn_p, rot_p)

    # Save the graph to a file
    save_graph_as_json(
        graph,
        DATA / "graph.json",
    )

    # Create the robot
    robot = construct_mjspec_from_graph(graph)
    run(robot)


def generate_prob_space(
    num_modules: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the probability space for the robot modules.

    Parameters
    ----------
    num_modules : int
        The number of modules in the robot.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The type, connection, and rotation probability spaces.
    """
    # "Type" probability space
    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = RNG.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    rotation_probability_space = RNG.random(
        size=(num_modules, NUM_OF_ROTATIONS),
        dtype=np.float32,
    )

    return (
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )


def run(
    robot: CoreModule,
) -> None:
    """Entry point."""
    # BugFix -> "Python exception raised"
    mujoco.set_mjcb_control(None)

    # MuJoCo configuration
    viz_options = mujoco.MjvOption()  # visualization of various elements

    # Visualization of the corresponding model or decoration element
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    world = SimpleFlatWorld()

    # Set transparency for the robot geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Save the model to XML
    xml = world.spec.to_xml()
    with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)

    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # What to track (fitness function inputs should be bound here!)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # CPG Optimizer
    num_of_actuators = model.nu
    optimizer = CMA(
        mean=np.zeros((num_of_actuators, num_of_actuators)),
        sigma=2.0,
    )

    # Optimize the robot's behaviour
    for generation in range(50):
        solutions = []
        best_weights = None
        best_value = 1.0
        for member in range(optimizer.population_size):
            # Actuators and CPG
            weight_matrix = optimizer.ask()
            cpg = CPGSensoryFeedback(
                num_neurons=int(num_of_actuators),
                sensory_term=-0.0,
                _lambda=0.01,
                coupling_weights=weight_matrix,
            )
            cpg.reset()
            mujoco.set_mjcb_control(lambda m, d: policy(m, d, cpg))

            simple_runner(
                model=model,
                data=data,
                duration=60,
            )

            # Calculate displacement (fitness function goes here!)
            value = np.sqrt(
                np.sum(np.exp2(to_track[0].xpos - to_track[-1].xpos)),
            )
            print(to_track[0].xpos, to_track[-1].xpos)

            solutions.append((weight_matrix, value))

            # Check if this is the best solution
            if value > best_value:
                best_value = value
                best_weights = weight_matrix

            console.log(
                f"#{generation} {member} {value} {weight_matrix.shape}",
            )
        optimizer.tell(solutions)

    console.log(best_value)

    cpg = CPGSensoryFeedback(
        num_neurons=int(num_of_actuators),
        sensory_term=-0.0,
        _lambda=0.01,
        coupling_weights=best_weights,
    )
    cpg.reset()
    mujoco.set_mjcb_control(lambda m, d: policy(m, d, cpg))

    viewer.launch(
        model=model,
        data=data,
    )


def policy(
    model: mujoco.MjModel,  # noqa: ARG001
    data: mujoco.MjData,
    cpg: CPGSensoryFeedback,
) -> None:
    """Use feedback term to shift the output of the CPGs."""
    x, _ = cpg.step()
    data.ctrl = x * np.pi / 2


if __name__ == "__main__":
    main()
