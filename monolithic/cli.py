"""Console script for monolithic."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("monolithic")
    click.echo("=" * len("monolithic"))
    click.echo("A python library for ultra-precision optical metrology and fabrication.")


if __name__ == "__main__":
    main()  # pragma: no cover
