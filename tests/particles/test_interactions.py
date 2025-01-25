"""Tests for classes representing particle interactions."""

from collections import namedtuple

import pytest

from plasmapy.particles.interactions import NuclearReaction

ReactionTestCase = namedtuple("ReactionTestCase", ["reactants", "products"])


@pytest.mark.parametrize(("reactants", "products"), [(["D", "T"], ["He-4", "n"])])
def test_reaction(reactants, products):
    nuclear_reaction = NuclearReaction(reactants=reactants, products=products)
    assert nuclear_reaction.reactants == reactants
    assert nuclear_reaction.products == products
