from not_again_ai.viz.barplots import simple_barplot


def test_simple_barplot() -> None:
    save_pathname = ".nox/temp/barplot_test1.png"
    x = ["fence", "wall", "gate", "door", "window", "counter", "stair", "curtain", "ceiling", "floor"]
    y = [0.5, 0.01, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.3]
    simple_barplot(
        x,
        y,
        save_pathname,
        order="asc",
        title="Token logits",
        x_label="token",
        y_label="logit",
    )

    save_pathname = ".nox/temp/barplot_test2.png"
    simple_barplot(
        y,
        x,
        save_pathname,
        order="desc",
        orient_bars_vertically=False,
        title="Token logits",
        x_label="logit",
        y_label="token",
    )

    save_pathname = ".nox/temp/barplot_test3.png"
    simple_barplot(
        y,
        x,
        save_pathname,
        order="asc",
        orient_bars_vertically=False,
        title="Token logits",
        x_label="logit",
        y_label="token",
    )

    save_pathname = ".nox/temp/barplot_test4.png"
    simple_barplot(
        x,
        y,
        save_pathname,
        order="desc",
        orient_bars_vertically=True,
        title="Token logits",
        x_label="logit",
        y_label="token",
    )
