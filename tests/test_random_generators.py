import sys
from pathlib import Path

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(0, str(path))


from jaxfi import jaxm


def test_randint():
    low, high, shape = -2, 5, (10, 3)
    assert jaxm.randint(low, high, shape).dtype == jaxm.int64
    assert jaxm.randint(low, high, shape, dtype=jaxm.int32).dtype == jaxm.int32


if __name__ == "__main__":
    test_randint()
