import os
import tempfile
from logparser.utils import load_dict, save_dict

def test_load_dict():
    # create a temporary file and save a dictionary to it
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        filepath = fp.name
        d = {"key1": "value1", "key2": "value2"}
        save_dict(d, filepath)

    # load the dictionary from the temporary file
    loaded_dict = load_dict(filepath)

    # check that the loaded dictionary matches the original dictionary
    assert loaded_dict == d

    # delete the temporary file
    os.remove(filepath)