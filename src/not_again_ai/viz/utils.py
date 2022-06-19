import matplotlib.pyplot as plt
import seaborn as sns


def reset_plot_libs() -> None:
    """Resets the plot libraries so that subsequent method calls are not impacted."""
    plt.clf()
    plt.cla()
    plt.close()
    sns.reset_orig()
