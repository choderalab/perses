"""
Utility classes and functions for perses protocols using gufe objects.
"""

from gufe.settings import Settings
from openff.units import unit


def _serialize_pydantic(settings: Settings) -> str:
    def serialize_unit(thing):
        # this gets called when a thing can't get jsonified by pydantic
        # for now only unit.Quantity fall foul of this requirement
        if not isinstance(thing, unit.Quantity):
            raise TypeError
        return '__Quantity__' + str(thing)
    return settings.json(encoder=serialize_unit)
