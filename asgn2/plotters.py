import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import csv

plt.rcParams["figure.constrained_layout.use"] = True

from dataclasses import dataclass, fields, asdict


@dataclass
class Datapoint:
    gen: int
    mean: float
    std: float
    highest: float
    lowest: float


class FitnessPlotter:
    def __init__(self) -> None:
        self.history: list[Datapoint] = []

    def add(self, data: list[float], gen: int):
        self.history.append(
            Datapoint(
                gen, float(np.mean(data)), float(np.std(data)), max(data), min(data)
            ),
        )

    def generations(self) -> NDArray:
        return np.array([p.gen for p in self.history])

    def mean(self) -> NDArray:
        return np.array([p.mean for p in self.history])

    def std(self) -> NDArray:
        return np.array([p.std for p in self.history])

    def max(self) -> NDArray:
        return np.array([p.highest for p in self.history])

    def min(self) -> NDArray:
        return np.array([p.lowest for p in self.history])

    def savefig(self, save_dir: str = "__data__", postfix: None | str = None):
        plt.plot(self.generations(), self.mean(), "b-", label="Average")
        plt.fill_between(
            self.generations(),
            self.mean() - self.std(),
            self.mean() + self.std(),
            facecolor="C0",
            alpha=0.4,
            label="Std. dev.",
        )
        plt.plot(self.generations(), self.max(), "k-", linewidth=1, label="Min/Max")
        plt.plot(self.generations(), self.min(), "k-", linewidth=1)

        plt.ylim(bottom=0)
        plt.title("Average fitness")
        plt.legend(loc="lower right")

        plt.savefig(f"{save_dir}/mean_fitness{"_" if postfix else ""}{postfix}.png")
        plt.close()

    def savedata(self, save_dir: str = "__data__", postfix: str = ""):
        with open(
            f"{save_dir}/fitness_progression{"_" if postfix else ""}{postfix}.csv", "w"
        ) as file:
            field_names = [field.name for field in fields(Datapoint)]
            writer = csv.DictWriter(file, field_names)
            writer.writeheader()
            writer.writerows([asdict(point) for point in self.history])
