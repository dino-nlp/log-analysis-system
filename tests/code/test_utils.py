import tempfile
from pathlib import Path

from logparser import utils


def test_save_and_load_dict():
    """Test saving and loading a dictionary."""
    d = {"a": 1, "b": 2, "c": 3}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        filepath = tmpdir / "test.json"
        utils.save_dict(d, filepath)
        d2 = utils.load_dict(filepath)
        assert d == d2
