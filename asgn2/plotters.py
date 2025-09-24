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


def walking_avg(array: NDArray, radius: int) -> NDArray:
    return np.convolve(array, np.ones(2 * radius + 1), mode="same") / (
        np.convolve(np.ones_like(array), np.ones(2 * radius + 1), mode="same")
    )


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
    uniform_mean = (uniform_1_mean + uniform_2_mean + uniform_3_mean) / 3
    selfadaptive_mean = (
        selfadaptive_1_mean + selfadaptive_2_mean + selfadaptive_3_mean
    ) / 3

    nobrain_mean_std = np.std([nobrain_1_mean, nobrain_2_mean, nobrain_3_mean], axis=0)
    uniform_mean_std = np.std([uniform_1_mean, uniform_2_mean, uniform_3_mean], axis=0)
    selfadaptive_mean_std = np.std(
        [selfadaptive_1_mean, selfadaptive_2_mean, selfadaptive_3_mean], axis=0
    )

    plt.plot(range(2000), nobrain_mean, label="NoBrain avg")
    plt.fill_between(
        range(2000),
        nobrain_mean - nobrain_mean_std,
        nobrain_mean + nobrain_mean_std,
        facecolor="C0",
        alpha=0.4,
        label="NoBrain avg $\\sigma$",
    )

    plt.plot(range(2000), uniform_mean, label="Uniform avg")
    plt.fill_between(
        range(2000),
        uniform_mean - uniform_mean_std,
        uniform_mean + uniform_mean_std,
        facecolor="C1",
        alpha=0.4,
        label="Uniform avg $\\sigma$",
    )

    plt.plot(range(2000), selfadaptive_mean, label="SelfAdaptive avg")
    plt.fill_between(
        range(2000),
        selfadaptive_mean - selfadaptive_mean_std,
        selfadaptive_mean + selfadaptive_mean_std,
        facecolor="C2",
        alpha=0.4,
        label="SelfAdaptive avg $\\sigma$",
    )
    plt.title("Mean fitness progression, 3 run average")
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0, 1)
    plt.savefig("__data__/mean.png")
    plt.close()

    nobrain_max = (nobrain_1_max + nobrain_2_max + nobrain_3_max) / 3
    uniform_max = (uniform_1_max + uniform_2_max + uniform_3_max) / 3
    selfadaptive_max = (
        selfadaptive_1_max + selfadaptive_2_max + selfadaptive_3_max
    ) / 3

    nobrain_max_std = np.std([nobrain_1_max, nobrain_2_max, nobrain_3_max], axis=0)
    uniform_max_std = np.std([uniform_1_max, uniform_2_max, uniform_3_max], axis=0)
    selfadaptive_max_std = np.std(
        [selfadaptive_1_max, selfadaptive_2_max, selfadaptive_3_max], axis=0
    )

    plt.plot(range(2000), nobrain_max, label="NoBrain")
    plt.fill_between(
        range(2000),
        nobrain_max - nobrain_max_std,
        nobrain_max + nobrain_max_std,
        facecolor="C0",
        alpha=0.4,
        label="NoBrain $\\sigma$",
    )

    plt.plot(range(2000), uniform_max, label="Uniform")
    plt.fill_between(
        range(2000),
        uniform_max - uniform_max_std,
        uniform_max + uniform_max_std,
        facecolor="C1",
        alpha=0.4,
        label="Uniform $\\sigma$",
    )

    plt.plot(range(2000), selfadaptive_max, label="SelfAdaptive")
    plt.fill_between(
        range(2000),
        selfadaptive_max - selfadaptive_max_std,
        selfadaptive_max + selfadaptive_max_std,
        facecolor="C2",
        alpha=0.4,
        label="SelfAdaptive $\\sigma$",
    )
    plt.title("Highest fitness progression, 3 run average")
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0, 1)
    plt.savefig("__data__/max.png")
    plt.close()

    plt.plot(range(2000), walking_avg(uniform_1_mean, 5), "C0", label="Run 1 avg")
    plt.plot(range(2000), uniform_1_max, "C0--", label="Run 1 max")
    plt.fill_between(
        range(2000),
        walking_avg(uniform_1_mean - uniform_1_std, 5),
        walking_avg(uniform_1_mean + uniform_1_std, 5),
        facecolor="C0",
        alpha=0.4,
        label="Run 1 std",
    )

    plt.plot(range(2000), walking_avg(uniform_2_mean, 5), "C1", label="Run 2 avg")
    plt.plot(range(2000), uniform_2_max, "C1--", label="Run 2 max")
    plt.fill_between(
        range(2000),
        walking_avg(uniform_2_mean - uniform_2_std, 5),
        walking_avg(uniform_2_mean + uniform_2_std, 5),
        facecolor="C1",
        alpha=0.4,
        label="Run 2 std",
    )

    plt.plot(range(2000), walking_avg(uniform_3_mean, 5), "C2", label="Run 3 avg")
    plt.plot(range(2000), uniform_3_max, "C2--", label="Run 3 max")
    plt.fill_between(
        range(2000),
        walking_avg(uniform_3_mean - uniform_3_std, 5),
        walking_avg(uniform_3_mean + uniform_3_std, 5),
        facecolor="C2",
        alpha=0.4,
        label="Run 3 std",
    )
    plt.title("Uniform fitnesses")
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0, 0.7)
    plt.savefig("__data__/uniform_fitnesses.png")
    plt.close()

    plt.plot(range(2000), walking_avg(selfadaptive_1_mean, 5), "C0", label="Run 1 avg")
    plt.plot(range(2000), selfadaptive_1_max, "C0--", label="Run 1 max")
    plt.fill_between(
        range(2000),
        walking_avg(selfadaptive_1_mean - selfadaptive_1_std, 5),
        walking_avg(selfadaptive_1_mean + selfadaptive_1_std, 5),
        facecolor="C0",
        alpha=0.4,
        label="Run 1 std",
    )

    plt.plot(range(2000), walking_avg(selfadaptive_2_mean, 5), "C1", label="Run 2 avg")
    plt.plot(range(2000), selfadaptive_2_max, "C1--", label="Run 2 max")
    plt.fill_between(
        range(2000),
        walking_avg(selfadaptive_2_mean - selfadaptive_2_std, 5),
        walking_avg(selfadaptive_2_mean + selfadaptive_2_std, 5),
        facecolor="C1",
        alpha=0.6,
        label="Run 2 std",
    )

    plt.plot(range(2000), walking_avg(selfadaptive_3_mean, 5), "C2", label="Run 3 avg")
    plt.plot(range(2000), selfadaptive_3_max, "C2--", label="Run 3 max")
    plt.fill_between(
        range(2000),
        walking_avg(selfadaptive_3_mean - selfadaptive_3_std, 5),
        walking_avg(selfadaptive_3_mean + selfadaptive_3_std, 5),
        facecolor="C2",
        alpha=0.4,
        label="Run 3 std",
    )
    plt.title("SelfAdaptive fitnesses")
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0, 1)
    plt.savefig("__data__/selfadaptive_fitnesses.png")
    plt.close()


if __name__ == "__main__":
    plot_data()
