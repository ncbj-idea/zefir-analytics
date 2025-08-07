# ZefirAnalytics
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pyzefir.model.network import Network

from zefir_analytics._engine.data_loader import (
    DataLoader,
    MissingParametersFileException,
)


# Define a fixture for common mocks
@pytest.fixture
def mock_data_loader() -> Generator[dict[str, Any], Any, None]:
    with patch("zefir_analytics._engine.data_loader.CsvParser") as MockCsvParser, patch(
        "zefir_analytics._engine.data_loader.NetworkCreator"
    ) as MockNetworkCreator, patch(
        "zefir_analytics._engine.data_loader.pd.read_csv"
    ) as mock_read_csv, patch(
        "zefir_analytics._engine.data_loader.pd.read_feather"
    ) as mock_read_feather, patch(
        "zefir_analytics._engine.data_loader.GeneralResultDirectory"
    ) as MockGeneralResultDirectory:

        # Configure mocks
        mock_csv_parser_instance = MockCsvParser.return_value
        mock_csv_parser_instance.load_dfs.return_value = {
            "source_key": {"df_key": pd.DataFrame({"A": [1, 2], "B": [3, 4]})}
        }

        MockNetworkCreator.create.return_value = MagicMock(Network)

        mock_read_csv.return_value = pd.Series([42.0], index=[0]).squeeze()
        mock_read_feather.return_value = pd.DataFrame({"index": [0], "value": [42.0]})

        MockGeneralResultDirectory.__iter__.return_value = [
            MagicMock(value="group1"),
            MagicMock(value="group2"),
        ]

        yield {
            "mock_csv_parser": MockCsvParser,
            "mock_network_creator": MockNetworkCreator,
            "mock_read_csv": mock_read_csv,
            "mock_read_feather": mock_read_feather,
            "mock_general_result_directory": MockGeneralResultDirectory,
        }


@pytest.mark.parametrize(
    "result_path_name, expected_function_value", [("csv", 42.0), ("feather", 42.0)]
)
def test_load_data(
    mock_data_loader: dict[str, Any],
    result_path_name: str,
    expected_function_value: float,
) -> None:
    # Mock file paths
    source_path = Path("/fake/source")
    result_path = Path(f"/fake/{result_path_name}")
    scenario_name = "test_scenario"

    # Mock methods
    data_loader = mock_data_loader["mock_csv_parser"]
    data_loader.load_dfs.return_value = {
        "source_key": {"df_key": pd.DataFrame({"A": [1, 2], "B": [3, 4]})}
    }
    network_creator = mock_data_loader["mock_network_creator"]
    network_creator.create.return_value = MagicMock(Network)
    read_csv = mock_data_loader["mock_read_csv"]
    read_feather = mock_data_loader["mock_read_feather"]
    read_csv.return_value = pd.Series([42.0], index=[0]).squeeze()
    read_feather.return_value = pd.DataFrame({"index": [0], "value": [42.0]})

    result = DataLoader.load_data(source_path, result_path, scenario_name)

    assert isinstance(result, tuple)
    assert len(result) == 4
    assert isinstance(result[0], dict)
    assert isinstance(result[1], Network)
    assert isinstance(result[2], dict)
    assert isinstance(result[3], float)
    assert result[3] == expected_function_value


def test_load_source_data(mock_data_loader: dict[str, Any]) -> None:
    # Test the _load_source_data method
    result = DataLoader._load_source_data(Path("/fake/source"), "test_scenario")
    assert isinstance(result, dict)
    assert "source_key" in result
    assert "df_key" in result["source_key"]
    assert isinstance(result["source_key"]["df_key"], pd.DataFrame)


def test_create_network(mock_data_loader: dict[str, Any]) -> None:
    # Test the _create_network method
    df_dict = {"source_key": {"df_key": pd.DataFrame()}}
    result = DataLoader._create_network(df_dict)
    assert isinstance(result, Network)


def test_load_objective_function_from_csv(mock_data_loader: dict[str, Any]) -> None:
    # Test the _load_objective_function_from_csv method
    result = DataLoader._load_objective_function_from_csv(Path("/fake/path"))
    assert result == 42.0


def test_load_objective_function_from_feather(mock_data_loader: dict[str, Any]) -> None:
    # Test the _load_objective_function_from_feather method
    result = DataLoader._load_objective_function_from_feather(Path("/fake/feather"))
    assert result == 42.0


@pytest.mark.parametrize(
    "group_str, category, file_name, expected_result_key",
    [
        ("group1", "gens", "file", "file"),
        ("line", "line", "file-replace", "file->replace"),
    ],
)
def test_load_csv_result_data(
    mock_data_loader: dict[str, Any],
    group_str: str,
    file_name: str,
    category: str,
    expected_result_key: str,
) -> None:

    mock_data_loader["mock_general_result_directory"].__iter__.return_value = [
        MagicMock(value=group_str)
    ]
    mock_data_loader["mock_general_result_directory"].LINES_RESULTS = "line"

    data_category = MagicMock(spec=Path, name=f"{file_name}", stem=category)
    data_category.glob.return_value = [MagicMock(stem=file_name)]

    mock_group_path = MagicMock(spec=Path)
    mock_group_path.glob.side_effect = [[data_category]]

    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__.return_value = mock_group_path
    mock_path.glob.return_value = [mock_group_path]

    mock_read_csv = mock_data_loader["mock_read_csv"]
    mock_read_csv.return_value = pd.DataFrame({"data": [1, 2]})

    result = DataLoader._load_csv_result_data(mock_path)
    assert group_str in result
    assert expected_result_key in result[group_str][category]
    assert isinstance(result[group_str][category][expected_result_key], pd.DataFrame)


@pytest.mark.parametrize(
    "group_str, category, file_name, expected_result_key",
    [
        ("group1", "gens", "file", "file"),
        ("line", "line", "file-replace", "file->replace"),
    ],
)
def test_load_feather_result_data(
    mock_data_loader: dict[str, Any],
    group_str: str,
    file_name: str,
    category: str,
    expected_result_key: str,
) -> None:
    mock_data_loader["mock_general_result_directory"].__iter__.return_value = [
        MagicMock(value=group_str)
    ]
    mock_data_loader["mock_general_result_directory"].LINES_RESULTS = "line"

    data_category = MagicMock(spec=Path, name=f"{file_name}", stem=category)
    data_category.glob.return_value = [MagicMock(stem=file_name)]

    mock_group_path = MagicMock(spec=Path)
    mock_group_path.glob.side_effect = [[data_category]]

    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__.return_value = mock_group_path
    mock_path.glob.return_value = [mock_group_path]

    mock_read_feather = mock_data_loader["mock_read_feather"]
    mock_read_feather.return_value = pd.DataFrame({"index": [0], "data": [1]})

    result = DataLoader._load_feather_result_data(mock_path)
    assert group_str in result
    assert expected_result_key in result[group_str][category]
    assert isinstance(result[group_str][category][expected_result_key], pd.DataFrame)


def test_missing_parameters_file_exception() -> None:
    with pytest.raises(MissingParametersFileException):
        raise MissingParametersFileException("Test exception")
