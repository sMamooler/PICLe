import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns


def bar_plot(results_df: pd.DataFrame, model: str, metric: str):

    fontsize = 70
    plt.rcParams["hatch.linewidth"] = 1
    _, ax = plt.subplots(figsize=(30, 7))

    palette = [
        "#A7C7E7",  # Pastel Blue
        "#F8E5A5",  # Pastel Yellow
        "#B5EAD7",  # Pastel Green
        "#A0D8B3",  #  Muted Pastel Green
        "#F6C7A3",  # Pastel Orange
        "#E8B494",  # Muted Pastel Orange
        "#D6B3E7",  # Pastel Purple
        "#C9A2D8",  # Muted Pastel Purple
        "#F2B2C2",  # Pastel Pink
        "#E3A3B5",  # Muted Pastel Pink
    ]
    palette = [
        "lightgray",
        "gray",
        "darkcyan",
        "#40B9B9",
        "cornflowerblue",
        "#A2B9F2",
        "orangered",
        "#FF8566",
        "darkorange",
        "#FFBE66",
    ]

    ax = sns.barplot(
        x="dataset",
        y=metric,
        hue="experiment_name",
        data=results_df,
        ax=ax,
        edgecolor=".2",
        linewidth=2.5,
        palette=palette,
        alpha=1,
    )

    hatches = itertools.cycle(["", "o", "x", "*", ".", "\\", "+", "/", "-", "/"])
    hatch = "/"
    num_locations = len(results_df["experiment_name"].unique())
    location_w_hatch = []
    for i in [1, 3, 5, 7, 9]:
        location_w_hatch.extend(
            [
                num_locations * i,
                num_locations * i + 1,
                num_locations * i + 2,
                num_locations * i + 3,
                num_locations * i + 4,
            ]
        )

    for i, patch in enumerate(ax.patches):
        if i % (num_locations) == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        horizontalalignment="center",
        fontsize=fontsize - 5,
    )
    ax.set_xlabel("")

    ax.set_ylabel(metric, fontsize=fontsize - 5)
    if model == "gpt-3.5-turbo":
        ax.set_ylim(0.0, 0.9)
    elif model == "mistral":
        ax.set_ylim(0.0, 0.62)

    ax.tick_params(axis="y", labelsize=fontsize - 5)

    for y in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ax.axhline(y=y, color="gray", linestyle="--")

    plt.legend(
        bbox_to_anchor=(-0.02, 1.3),
        loc=2,
        borderaxespad=0.0,
        fontsize=fontsize - 10,
        ncol=5,
    )
    plt.tight_layout()
    plt.xlabel("")
    plt.show()
