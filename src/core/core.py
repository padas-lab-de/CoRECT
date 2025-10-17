import click

from core.cli.create import create
from core.cli.huggingface.hf_push import hf_push
from core.cli.prepare import prepare


@click.group()
def main():
    """
    CoRE

    Scripts for creating the CoRE dataset collection.
    """
    pass


main.add_command(prepare)
main.add_command(create)
main.add_command(hf_push)


if __name__ == "__main__":
    main()
