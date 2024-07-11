import numpy as np
import matplotlib.pyplot as plt

COLOR_ZPDES = "#114B5B"
COLOR_BASELINE = "#9CC19F"


def plot_scatter_per_activity_on_ax(ax, idx, data, size, condition):
    c = COLOR_ZPDES if condition == "zpdes" else COLOR_BASELINE
    delta = 1 if condition == "zpdes" else -1
    shift = 0.1
    x = np.array([0, 1, 2, 3]) + delta * shift
    for participant in range(size):
        ax.scatter(x, data[idx][participant], c=c, s=0.3)


def plot_regression_line_per_activity_on_ax(ax, idx, data, condition):
    c = COLOR_ZPDES if condition == "zpdes" else COLOR_BASELINE
    # We just need 2 points to draw a line:
    x = np.array([0, 1, 2, 3])
    # Compute y based on and slope and intercept
    y = data[idx][0] + x * data[idx][1]
    ax.plot(x, y, c=c)


def plot_mean_intra(df, path, metric_type="binary", study="v3_prolific"):
    figsize, dpi = (8, 6), 300
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    zpdes = df[df['condition'] == 'zpdes']
    baseline = df[df['condition'] == 'baseline']
    for index, row in df.iterrows():
        shift = 0.08 if row['condition'] == "zpdes" else -0.08
        color = COLOR_ZPDES if row['condition'] == "zpdes" else COLOR_BASELINE
        x = [i + shift for i in range(4)]
        ax.scatter(x, row[['s_0', 's_1', 's_2', 's_3']], c=color, s=1.5)
        # ax.plot([i for i in range(4)], row[['s_0', 's_1', 's_2', 's_3']], c='red', linewidth=2, marker="s", markersize=4)
    ax.plot([i for i in range(4)], zpdes[['s_0', 's_1', 's_2', 's_3']].mean(), c=COLOR_ZPDES, marker="s",
            markersize=4, label="ZPDES")
    ax.plot([i for i in range(4)], baseline[['s_0', 's_1', 's_2', 's_3']].mean(), c=COLOR_BASELINE, marker="s",
            markersize=4, label="Baseline")
    ax.set_xticks([i for i in range(4)], ['s_1', 's_4', 's_5', 's_8'])
    ax.set_title(f"Average {metric_type} score \n {study}")
    fig.savefig(f"{path}intra_mean_{metric_type}_{study}.png")