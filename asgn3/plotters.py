from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.rcParams["figure.constrained_layout.use"] = True


class LivePlotter:
    line_amount = 9

    def __init__(self, fitness: NDArray[np.float32], directory: Path) -> None:
        self.fitness = fitness
        self.directory = directory

    def plot(self) -> None:
        averages = np.average(self.fitness, axis=1)
        quantiles = np.quantile(
            self.fitness, np.linspace(0, 1, self.line_amount), axis=1
        ).T
        current_gen = np.max(np.nonzero(np.array(averages)))
        gen_range = np.arange(current_gen + 1)

        for q in range(self.line_amount - 1):
            plt.plot(gen_range, quantiles[: current_gen + 1, q], "k-")
        plt.plot(
            gen_range,
            quantiles[: current_gen + 1, self.line_amount - 1],
            "k-",
            label="quantiles",
        )
        plt.plot(gen_range, averages[: current_gen + 1], "r-", label="average")

        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title(f"Progress generation {current_gen}")
        plt.legend()
        plt.savefig(self.directory.joinpath("fitness_plot.pdf"))
        plt.close()
