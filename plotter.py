import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


df = pd.read_csv("r0123456.csv",skiprows=2,header=None,usecols=[0,1,2,3],names=["Iteration", "Time", "Mean", "Best"])


def convergence():
    plt.title("Convergence")

    plotMean = plt.plot(df["Iteration"], df["Mean"], 'b', label="Mean")
    plotBest = plt.plot(df["Iteration"], df["Best"], 'r', label="Best")

    plt.xlabel("Iteration (#)")
    plt.ylabel("Path length (#)")
    plt.legend()
    plt.show()


def variation():
    plt.title("Variation of result between runs")
    plt.xlabel("Path length (#)")
    plt.ylabel("Solution count (#)")

    runs = np.fromfile("multiple_tries.csv")
    sns.displot(runs)
    plt.show()

variation()


