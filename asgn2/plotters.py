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


def plot_data():
    with open("asgn2/results/fitness_progression_nobrain_1.csv") as file:
        reader = csv.DictReader(file)
        nobrain_1_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_1_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_1_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_1_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_nobrain_2.csv") as file:
        reader = csv.DictReader(file)
        nobrain_2_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_2_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_2_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_2_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_nobrain_3.csv") as file:
        reader = csv.DictReader(file)
        nobrain_3_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_3_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_3_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        nobrain_3_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_uniform_1.csv") as file:
        reader = csv.DictReader(file)
        uniform_1_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_1_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_1_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_1_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_uniform_2.csv") as file:
        reader = csv.DictReader(file)
        uniform_2_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_2_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_2_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_2_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_uniform_3.csv") as file:
        reader = csv.DictReader(file)
        uniform_3_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_3_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_3_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        uniform_3_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_selfadaptive_1.csv") as file:
        reader = csv.DictReader(file)
        selfadaptive_1_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_1_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_1_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_1_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_selfadaptive_2.csv") as file:
        reader = csv.DictReader(file)
        selfadaptive_2_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_2_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_2_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_2_min = np.array([float(row["lowest"]) for row in reader])

    with open("asgn2/results/fitness_progression_selfadaptive_3.csv") as file:
        reader = csv.DictReader(file)
        selfadaptive_3_mean = np.array([float(row["mean"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_3_std = np.array([float(row["std"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_3_max = np.array([float(row["highest"]) for row in reader])
        file.seek(0)
        next(reader)
        selfadaptive_3_min = np.array([float(row["lowest"]) for row in reader])

    nobrain_mean = (nobrain_1_mean + nobrain_2_mean + nobrain_3_mean) / 3


if __name__ == "__main__":
    plot_data()
