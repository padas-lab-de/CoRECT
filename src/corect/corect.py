import click

from corect.cli.evaluate import evaluate
from corect.cli.statistics import statistics
from corect.cli.visualize.heatmap import heatmap
from corect.cli.visualize.line_chart import line_chart
from corect.cli.visualize.pareto_plot import pareto_plot


@click.group()
def main():
    """
    CoRECT

    Studying the correlation between dataset complexity and retrieval performance.
    """
    pass


@click.group()
def jl_lemma():
    """
    Run experiments around the JL Lemma.
    """
    pass


@click.group()
def visualize():
    """
    Visualize the results of the experiments.
    """
    pass


main.add_command(evaluate)
main.add_command(statistics)

main.add_command(visualize)
visualize.add_command(heatmap)
visualize.add_command(line_chart)
visualize.add_command(pareto_plot)


if __name__ == "__main__":
    main()
