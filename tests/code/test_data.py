import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from logparser import data

TEST_LOG = "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log"


@patch("requests.get")
def test_download_logs_success(mock_get):
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "log line 1\nlog line 2\nlog line 3"
    mock_get.return_value = mock_response

    # Call the function
    lines = data.download_logs(TEST_LOG)

    # Check that the mock objects were called correctly
    mock_get.assert_called_once_with(TEST_LOG)

    # Check that the function returns the expected output
    expected_lines = ["log line 1", "log line 2", "log line 3"]
    assert lines == expected_lines


@patch("requests.get")
def test_download_logs_failure(mock_get):
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_get.return_value = mock_response

    # Call the function
    url = "https://example.com/logs/bgl.log"
    lines = data.download_logs(url)

    # Check that the mock objects were called correctly
    mock_get.assert_called_once_with(url)

    # Check that the function returns an empty list
    expected_lines = []
    assert lines == expected_lines
