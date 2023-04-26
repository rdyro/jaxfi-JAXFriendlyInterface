import unittest

from jfi import jaxm
import jfi


def test_randint():
    low, high, shape = -2, 5, (10, 3)
    assert jaxm.randint(low, high, shape).dtype == jaxm.int64
    assert jaxm.randint(low, high, shape, dtype=jaxm.int32).dtype == jaxm.int32