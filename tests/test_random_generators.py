import sys
from pathlib import Path

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(path, 0)


from jfi import jaxm


def test_randint():
    low, high, shape = -2, 5, (10, 3)
    assert jaxm.randint(low, high, shape).dtype == jaxm.int64
    assert jaxm.randint(low, high, shape, dtype=jaxm.int32).dtype == jaxm.int32


if __name__ == "__main__":
    test_randint()