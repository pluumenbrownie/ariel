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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
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

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)

    instrumentation = ng.p.Instrumentation(
        type_p=ng.p.Array(
            init=RNG.random(
                size=(num_modules, NUM_OF_TYPES_OF_MODULES),
            ),
            lower=0.0,
            upper=1.0,
        ),
        conn_p=ng.p.Array(
            init=RNG.random(
                size=(num_modules, num_modules, NUM_OF_FACES),
            ),
            lower=0.0,
            upper=1.0,
        ),
        rot_p=ng.p.Array(
            init=RNG.random(
                size=(num_modules, NUM_OF_ROTATIONS),
            ),
            lower=0.0,
            upper=1.0,
        ),
    )
    budget = 200
    optimizer = ng.optimizers.CMA(
        parametrization=instrumentation,
        budget=budget,
        num_workers=1,
    )

    loss_history_f1 = []
    for idx in range(budget):
        console.rule(f"idx: {idx}")
        x = optimizer.ask()

        # Create the robot
        type_p, conn_p, rot_p = (
            x.kwargs["type_p"],
            x.kwargs["conn_p"],
            x.kwargs["rot_p"],
        )
        graph: Graph[Any] = hpd.probability_matrices_to_graph(
            type_p,
            conn_p,
            rot_p,
        )
        robot = construct_mjspec_from_graph(graph)
        fitness = run(robot)

        # Save the graph to a file
        save_graph_as_json(
            graph,
            DATA / f"graph_{idx}.json",
        )

        loss_history_f1.append(fitness)
        optimizer.tell(x, fitness)

    recommendation = optimizer.provide_recommendation()
    console.log(recommendation.value)

    # plot
    plt.plot(loss_history_f1)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.tight_layout()
    plt.show()


def run(
    robot: CoreModule,
) -> float:
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
    instrumentation = ng.p.Instrumentation(
        genotype=ng.p.Array(
            init=RNG.uniform(
                size=(num_of_actuators, num_of_actuators),
            ),
        ),
    )

    budget = 1
    optimizer = ng.optimizers.CMA(
        parametrization=instrumentation,
        budget=budget,
        num_workers=1,
    )

    # Optimize the robot's behaviour
    best_value = 1.0
    for generation in range(budget):
        # Actuators and CPG
        weight_matrix = optimizer.ask()
        cpg = CPGSensoryFeedback(
            num_neurons=int(num_of_actuators),
            sensory_term=-0.0,
            _lambda=0.01,
            coupling_weights=weight_matrix.kwargs["genotype"],
        )
        cpg.reset()
        mujoco.set_mjcb_control(lambda m, d: policy(m, d, cpg))

        start = time.time()
        simple_runner(
            model=model,
            data=data,
            duration=60,
        )
        end = time.time()

        # Calculate displacement (fitness function goes here!)
        value = -np.sqrt(
            np.sum(np.exp2(to_track[0].xpos - to_track[-1].xpos)),
        )

        # Check if this is the best solution
        best_value = min(best_value, value)

        console.log(f"#{generation} {value}")
        optimizer.tell(weight_matrix, value)

    console.log(best_value)
    return best_value


def policy(
    model: mujoco.MjModel,  # noqa: ARG001
    data: mujoco.MjData,
    cpg: CPGSensoryFeedback,
) -> None:
    """Use feedback term to shift the output of the CPGs."""
    x, _ = cpg.step()
    data.ctrl = x * np.pi / 2


if __name__ == "__main__":
    # Test several times
    main()
