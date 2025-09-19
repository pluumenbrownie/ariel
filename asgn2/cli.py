import click

from enum import Enum

from A2_template import main, BrainType
from Neural_Net import SelfAdaptiveBrain, UniformBrain


@click.command()
@click.option(
    "--type",
    "-t",
    "brain_type",
    type=click.Choice(BrainType, case_sensitive=False),
    help="The Brain class used.",
)
@click.option(
    "--pop-size",
    "-p",
    "pop_size",
    default=100,
    help="The amount of individuals tested each generation.",
)
@click.option(
    "--max-gens",
    "-m",
    "max_gens",
    default=1,
    help="The maximum number amount of generations to run.",
)
def run(brain_type: BrainType, pop_size: int, max_gens: int):
    main(brain_type, pop_size, max_gens)


if __name__ == "__main__":
    run()
