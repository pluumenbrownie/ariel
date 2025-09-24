import click

from enum import Enum
from A2_template import main, BrainType, load_brain, compile_world
import mujoco
from mujoco import viewer


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # if ctx.invoked_subcommand is None:
    #     click.echo('I was invoked without subcommand')
    # else:
    #     click.echo(f"I am about to invoke {ctx.invoked_subcommand}")
    pass


@cli.command()
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


@cli.command()
@click.argument("brain_file", type=click.Path(exists=True))
def show(brain_file):
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE
    model, data, to_track = compile_world()
    controller = load_brain(brain_file)
    mujoco.set_mjcb_control(lambda m, d: controller.control(model, data, to_track))
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )


if __name__ == "__main__":
    cli()
