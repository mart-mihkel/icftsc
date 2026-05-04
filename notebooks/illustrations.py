import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt

    from instruct.constants import logdir

    return logdir, os, plt


@app.cell
def _(logdir, os):
    figpath = logdir / "fig" / "illustrations"
    os.makedirs(figpath, exist_ok=True)
    return (figpath,)


@app.cell
def _(figpath, plt):
    events = [
        (0, "1948", "Shannon / n-grammid", "#888780"),
        (1, "1951", "N-gramm keele\nmodelleerimine", "#888780"),
        (2, "1986", "Tehisnärvivõrgud /\nvektoresitused", "#378ADD"),
        (3, "1997", "LSTM", "#378ADD"),
        (4, "2003", "Süvaõppega\nkeelemudeldamine", "#1D9E75"),
        (5, "2014", "Tähelepanumehhanism", "#1D9E75"),
        (6, "2017", "Transformer", "#7F77DD"),
        (7, "2018", "Isejuhendatud\neeltreenimine", "#7F77DD"),
        (8, "2021", "Instruktsioonipõhine\npeenhäälestus", "#D85A30"),
        (9, "2023", "Inimtagasisega\nstiimulõpe", "#D85A30"),
    ]

    _fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.6, len(events) - 0.2)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")

    ax.plot([-0.4, len(events) - 0.3], [0, 0], color="#aaaaaa", lw=1.5, zorder=1)
    ax.annotate(
        "",
        xy=(len(events) - 0.25, 0),
        xytext=(len(events) - 0.55, 0),
        arrowprops={"arrowstyle": "-|>", "color": "#aaaaaa", "lw": 1.5},
    )

    for i, (x, year, label, color) in enumerate(events):
        above = i % 2 == 0
        y_box = 1.0 if above else -1.0
        y_start = 0.14 if above else -0.14
        y_end = 0.7 if above else -0.7

        ax.plot([x, x], [y_start, y_end], color=color, lw=1.2, zorder=2)

        ax.plot(
            x,
            0,
            "o",
            color=color,
            markersize=9,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        ax.text(
            x,
            y_box,
            f"{year}\n{label}",
            ha="center",
            va="center",
            fontsize=16,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.2,
            },
        )

    plt.tight_layout()
    plt.savefig(figpath / "timeline.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
