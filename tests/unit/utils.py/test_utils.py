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

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.utils import (
    argument_condition,
    assign_multiindex,
    dict_filter,
    get_generators_emission_types,
    handle_n_sample_results,
    reindex_multiindex_df_binding_years,
    reindex_single_df_binding_years,
)


@pytest.mark.parametrize(
    "name, expected_output",
    [
        pytest.param(
            "single_name",
            pd.DataFrame({"single_name": [6]}, index=["sum"]),
            id="single_name",
        ),
        pytest.param(
            ["name1", "name2"],
            {
                "name1": pd.DataFrame({"name1": [6]}, index=["sum"]),
                "name2": pd.DataFrame({"name2": [6]}, index=["sum"]),
            },
            id="list_of_two_names",
        ),
        pytest.param(
            ["type1", "type2", "type3"],
            {
                "type1": pd.DataFrame({"type1": [6]}, index=["sum"]),
                "type2": pd.DataFrame({"type2": [6]}, index=["sum"]),
                "type3": pd.DataFrame({"type3": [6]}, index=["sum"]),
            },
            id="list_of_three_names",
        ),
    ],
)
def test_argument_condition(
    name: str | list[str],
    expected_output: pd.DataFrame | dict[str, pd.DataFrame],
) -> None:
    def sample_function(name: str | list[str]) -> pd.DataFrame:
        if isinstance(name, str):
            df = pd.DataFrame({name: [1, 2, 3]}, index=[0, 1, 2])
            return pd.DataFrame({name: [df[name].sum()]}, index=["sum"])
        elif isinstance(name, list):
            return {el_name: sample_function(el_name) for el_name in name}

    result = argument_condition(name, sample_function)

    if isinstance(name, str):
        assert_frame_equal(result, expected_output)
    else:
        assert isinstance(result, dict)
        for key in name:
            assert_frame_equal(result[key], expected_output[key])


@pytest.mark.parametrize(
    "dictionary, keys, expected_output",
    [
        pytest.param(
            {"a": 1, "b": 2, "c": 3}, None, {"a": 1, "b": 2, "c": 3}, id="None_case"
        ),
        pytest.param({"a": 1, "b": 2, "c": 3}, "b", 2, id="Single_key_case"),
        pytest.param(
            {"a": 1, "b": 2, "c": 3},
            ["a", "c"],
            {"a": 1, "c": 3},
            id="Multiple_keys_case",
        ),
        pytest.param({"a": 1, "b": 2, "c": 3}, [], {}, id="Empty_list_case"),
        pytest.param({}, None, {}, id="Empty_dict_case"),
        pytest.param({"a": 1, "b": 2, "c": 3}, "d", {}, id="key_not_in_dict"),
        pytest.param(
            {"a": 1, "b": 2, "c": 3}, ["d", "e", "f"], {}, id="keys_not_in_dict"
        ),
        pytest.param(
            {"a": 1, "b": 2, "c": 3},
            ["a", "e", "b", "h"],
            {"a": 1, "b": 2},
            id="keys_not_or_in_dict",
        ),
    ],
)
def test_dict_filter(
    dictionary: dict[str, Any],
    keys: list[str] | str | None,
    expected_output: dict[str, Any] | Any,
) -> None:
    result = dict_filter(dictionary, keys)
    assert result == expected_output


@pytest.mark.parametrize(
    "df, label, expected_columns",
    [
        pytest.param(
            pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"]),
            "group1",
            pd.MultiIndex.from_tuples(
                [("A", "group1"), ("B", "group1"), ("C", "group1")]
            ),
            id="basic_case",
        ),
        pytest.param(
            pd.DataFrame([[1, 2], [3, 4]], columns=["X", "Y"]),
            "category",
            pd.MultiIndex.from_tuples([("X", "category"), ("Y", "category")]),
            id="different_label_case",
        ),
        pytest.param(
            pd.DataFrame(columns=["M", "N"]),
            "label",
            pd.Index(["M", "N"], dtype="object"),
            id="empty_df_case",
        ),
        pytest.param(
            pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["X", "Y", "Z"]),
            "year",
            pd.MultiIndex.from_tuples([("X", "year"), ("Y", "year"), ("Z", "year")]),
            id="another_label_case",
        ),
    ],
)
def test_assign_multiindex(
    df: pd.DataFrame, label: str, expected_columns: pd.MultiIndex
) -> None:
    result_df = assign_multiindex(df, label)
    assert result_df.columns.equals(expected_columns)


@pytest.mark.parametrize(
    "df, binding_years, level, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                np.array([[10, 20], [30, 40], [50, 60]]),
                index=pd.MultiIndex.from_tuples(
                    [("group1", 0), ("group1", 1), ("group1", 2)],
                    names=["Element Type", "Year"],
                ),
                columns=["A", "B"],
            ),
            pd.Series([100, 200, 300], index=[0, 1, 2]),
            "Year",
            pd.DataFrame(
                np.array([[10, 20], [30, 40], [50, 60]]),
                index=pd.MultiIndex.from_tuples(
                    [("group1", 100), ("group1", 200), ("group1", 300)],
                    names=["Element Type", "Year"],
                ),
                columns=["A", "B"],
            ),
            id="standard_case",
        ),
        pytest.param(
            pd.DataFrame(
                np.array([[5, 6], [7, 8]]),
                index=pd.MultiIndex.from_tuples(
                    [("group2", 1), ("group2", 2)], names=["Element Type", "Year"]
                ),
                columns=["M", "N"],
            ),
            pd.Series([1000, 2000], index=[1, 2]),
            "Year",
            pd.DataFrame(
                np.array([[5, 6], [7, 8]]),
                index=pd.MultiIndex.from_tuples(
                    [("group2", 1000), ("group2", 2000)], names=["Element Type", "Year"]
                ),
                columns=["M", "N"],
            ),
            id="different_index_types",
        ),
    ],
)
def test_reindex_multiindex_df_binding_years(
    df: pd.DataFrame, binding_years: pd.Series, level: str, expected_df: pd.DataFrame
) -> None:
    result = reindex_multiindex_df_binding_years(df, binding_years, level)
    assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "df, binding_years, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                np.array([[1, 2], [3, 4], [5, 6]]), index=[0, 1, 2], columns=["A", "B"]
            ),
            pd.Series([10, 20, 30], index=[0, 1, 2]),
            pd.DataFrame(
                np.array([[1, 2], [3, 4], [5, 6]]),
                index=[10, 20, 30],
                columns=["A", "B"],
            ),
            id="standard_case",
        ),
        pytest.param(
            pd.DataFrame(
                np.array([[11, 12], [13, 14]]), index=["a", "b"], columns=["M", "N"]
            ),
            pd.Series([1000, 2000], index=["a", "b"]),
            pd.DataFrame(
                np.array([[11, 12], [13, 14]]), index=[1000, 2000], columns=["M", "N"]
            ),
            id="different_index_types",
        ),
        pytest.param(
            pd.DataFrame(columns=["C", "D"]),
            pd.Series(dtype=int),
            pd.DataFrame(columns=["C", "D"]),
            id="empty_df_series",
        ),
    ],
)
def test_reindex_single_df_binding_years(
    df: pd.DataFrame, binding_years: pd.Series, expected_df: pd.DataFrame
) -> None:
    result = reindex_single_df_binding_years(df, binding_years)
    assert_frame_equal(result, expected_df)


def test_reindex_single_df_binding_years_key_error() -> None:
    df = pd.DataFrame([1, 2], index=["A", "B"])
    binding_years = pd.Series([100], index=["C"])

    with pytest.raises(KeyError):
        reindex_single_df_binding_years(df, binding_years)


def test_reindex_multiindex_df_binding_years_key_error() -> None:
    df = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=pd.MultiIndex.from_tuples(
            [("group1", 1), ("group1", 2)], names=["Element Type", "Year"]
        ),
        columns=["A", "B"],
    )
    binding_years = pd.Series([100], index=[3])

    with pytest.raises(KeyError):
        reindex_multiindex_df_binding_years(df, binding_years, "Year")


@pytest.mark.parametrize(
    "mock_generators, expected_output",
    [
        pytest.param(
            {
                "gen1": MagicMock(emission_fee={"CO2"}),
                "gen2": MagicMock(emission_fee={"CH4"}),
            },
            {"gen1": {"CO2"}, "gen2": {"CH4"}},
            id="valid_data",
        ),
        pytest.param(
            {
                "gen1": MagicMock(emission_fee=set()),
                "gen2": MagicMock(emission_fee={"CH4"}),
            },
            {"gen2": {"CH4"}},
            id="gen1_empty",
        ),
        pytest.param(
            {
                "gen1": MagicMock(emission_fee={"CO2"}),
                "gen2": MagicMock(emission_fee=None),
            },
            {"gen1": {"CO2"}},
            id="gen2_none",
        ),
        pytest.param(
            {
                "gen1": MagicMock(emission_fee=set()),
                "gen2": MagicMock(emission_fee=set()),
            },
            {},
            id="both_empty",
        ),
        pytest.param({}, {}, id="empty_case"),
    ],
)
def test_get_generators_emission_types(
    mock_generators: dict[str, MagicMock], expected_output: dict[str, set[str]]
) -> None:
    mock_network = MagicMock()
    mock_network.generators = mock_generators

    result = get_generators_emission_types(mock_network)

    assert result == expected_output


@pytest.mark.parametrize(
    "results, is_multiindex, year_binding",
    [
        pytest.param(
            {"df1": pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])},
            False,
            pd.Series([0, 1, 2]),
            id="result_dict_single_key_value",
        ),
        pytest.param(
            {
                "df1": pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2]),
                "df2": pd.DataFrame({"A": [4, 5, 36]}, index=[0, 1, 2]),
                "df3": pd.DataFrame({"A": [21, 21, 32]}, index=[0, 1, 2]),
                "df4": pd.DataFrame({"A": [21, 24, 35]}, index=[0, 1, 2]),
                "df5": pd.DataFrame({"A": [61, 72, 53]}, index=[0, 1, 2]),
            },
            False,
            pd.Series([0, 1, 2]),
            id="result_dict_single_key_value",
        ),
        pytest.param(
            {"df1": pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])},
            False,
            None,
            id="year_binding_none_result_dict",
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2]),
            False,
            None,
            id="year_binding_none_result_df",
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2]),
            True,
            pd.Series([0, 1, 2]),
            id="result_df_multiindex",
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2]),
            False,
            pd.Series([0, 1, 2]),
            id="result_df_single_key_value",
        ),
    ],
)
def test_handle_n_sample_results(
    results: dict[str, pd.DataFrame] | pd.DataFrame,
    is_multiindex: bool,
    year_binding: pd.Series | None,
) -> None:

    single_response = pd.DataFrame({"single": [True]})
    multi_response = pd.DataFrame({"multi": [True]})

    with patch(
        "zefir_analytics._engine.data_queries.utils.reindex_single_df_binding_years",
        return_value=single_response,
    ), patch(
        "zefir_analytics._engine.data_queries.utils.reindex_multiindex_df_binding_years",
        return_value=multi_response,
    ):
        result = handle_n_sample_results(results, year_binding, is_multiindex)
        if year_binding is None:
            if isinstance(result, dict):
                for key, df in result.items():
                    assert key in results
                    assert_frame_equal(df, results[key])
            else:
                assert_frame_equal(result, results)
        elif isinstance(result, dict):
            for df in result.values():
                assert_frame_equal(df, single_response)
        else:
            if is_multiindex:
                assert_frame_equal(result, multi_response)
            else:
                assert_frame_equal(result, single_response)
