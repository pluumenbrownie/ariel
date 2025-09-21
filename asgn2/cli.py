import click

from enum import Enum

from A2_template import main, BrainType


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
    help="The maximum number of generations to run.",
)
@click.option(
    "--save-dir",
    "-d",
    "save_dir",
    default="__data__",
    help="Where to save any generated files.",
)
@click.option(
    "--postfix",
    "-f",
    "postfix",
    default="",
    help="Postfix used on any saved files.",
)
def run(
    brain_type: BrainType, pop_size: int, max_gens: int, save_dir: str, postfix: str
):
    main(brain_type, pop_size, max_gens, postfix=postfix, save_dir=save_dir)


if __name__ == "__main__":
    run()
