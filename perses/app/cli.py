# New cli for testing

import click

@click.command()
def cli():
    """test"""
    click.echo('Hello World!')
