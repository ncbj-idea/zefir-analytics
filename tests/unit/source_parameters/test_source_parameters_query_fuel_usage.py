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

from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.fixture
def fuel_mocked_source_parameters_over_years_query(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
) -> SourceParametersOverYearsQuery:
    gen1 = MagicMock(energy_source_type="GEN_ET_1")
    gen1.name = "gen1"

    gen2 = MagicMock(energy_source_type="GEN_ET_2")
    gen2.name = "gen2"

    gen3 = MagicMock(energy_source_type="GEN_ET_1")
    gen3.name = "gen3"

    mocked_source_parameters_over_years_query._network.generators = {
        gen.name: gen for gen in (gen1, gen2, gen3)
    }

    gen_et_1 = MagicMock(fuel="fuel1")
    gen_et_1.name = "GEN_ET_1"

    gen_et_2 = MagicMock(fuel=None)
    gen_et_2.name = "GEN_ET_2"

    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2)
    }

    fuel1 = MagicMock(energy_per_unit=100)
    fuel1.name = "fuel1"

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen1", "gen2"},
        "group2": {"gen3"},
    }

    mocked_source_parameters_over_years_query._network.fuels = {fuel1.name: fuel1}

    return mocked_source_parameters_over_years_query


@pytest.mark.parametrize(
    "df, generator, fuel_name, expected_output",
    [
        pytest.param(
            pd.Series([0.1, 0.2, 0.3]),
            "gen1",
            "fuel1",
            pd.DataFrame(
                [[10.0, 20.0, 30.0]],
                columns=pd.MultiIndex.from_product([["gen1"], [0, 1, 2]]),
                index=["fuel1"],
            ),
            id="valid_data",
        ),
        pytest.param(
            pd.Series(),
            "gen1",
            "fuel1",
            pd.DataFrame(
                index=pd.Index([], dtype="object"),
                columns=pd.MultiIndex(levels=[[], []], codes=[[], []]),
            ),
            id="empty_series",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            "gen1",
            "fuel1",
            pd.DataFrame(
                [[100.0, 200.0, 300.0]],
                columns=pd.MultiIndex.from_product([["gen1"], [0, 1, 2]]),
                index=["fuel1"],
            ),
            id="valid_data_int_values",
        ),
        pytest.param(
            pd.Series([0.1, np.nan, 0.2]),
            "gen1",
            "fuel1",
            pd.DataFrame(
                [[10.0, 20.0]],
                columns=pd.MultiIndex.from_product([["gen1"], [0, 2]]),
                index=["fuel1"],
            ),
            id="with_np.nan",
        ),
        pytest.param(
            pd.Series([np.nan, np.nan, np.nan]),
            "gen1",
            "fuel1",
            pd.DataFrame(
                index=pd.Index([], dtype="object"),
                columns=pd.MultiIndex(levels=[[], []], codes=[[], []]),
            ),
            id="all_np.nan",
        ),
        pytest.param(
            pd.Series([0.1, 0.2, 0.3]),
            None,
            None,
            pd.DataFrame(
                index=pd.Index([], dtype="object"),
                columns=pd.MultiIndex(levels=[[], []], codes=[[], []]),
            ),
            id="generator_name_fuel_name_none",
        ),
    ],
)
def test_format_total_resolution(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    df: pd.Series,
    generator: str,
    fuel_name: str,
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._hourly_scale = 100.0
    result = mocked_source_parameters_over_years_query._format_total_resolution(
        df, generator, fuel_name
    )
    assert_frame_equal(result, expected_output, check_column_type=False)


@pytest.mark.parametrize(
    "df, generator, fuel_name, expected_output",
    [
        pytest.param(
            pd.DataFrame(
                {"0": [1, 2, 3], "1": [0.1, 0.2, 0.3]},
                index=pd.Index([0, 1, 2], name="Hour"),
            ),
            "gen1",
            "fuel1",
            pd.DataFrame(
                [[1, 0.1], [2, 0.2], [3, 0.3]],
                index=pd.MultiIndex.from_tuples(
                    [(0, "fuel1"), (1, "fuel1"), (2, "fuel1")]
                ),
                columns=pd.MultiIndex.from_tuples([("gen1", 0), ("gen1", 1)]),
            ),
            id="valid_data",
        ),
        pytest.param(
            pd.DataFrame(
                {"0": [1, 2, np.nan], "1": [np.nan, 0.2, 0.3]},
                index=pd.Index([0, 1, 2], name="Hour"),
            ),
            "gen1",
            "fuel1",
            pd.DataFrame(
                [[1, np.nan], [2, 0.2], [np.nan, 0.3]],
                index=pd.MultiIndex.from_tuples(
                    [(0, "fuel1"), (1, "fuel1"), (2, "fuel1")]
                ),
                columns=pd.MultiIndex.from_tuples([("gen1", 0), ("gen1", 1)]),
            ),
            id="with_np.nan",
        ),
        pytest.param(
            pd.DataFrame(
                {"0": [1, 2, 3], "1": [0.1, 0.2, 0.3]},
                index=pd.Index([0, 1, 2], name="Hour"),
            ),
            None,
            None,
            pd.DataFrame(
                [[1, 0.1], [2, 0.2], [3, 0.3]],
                index=pd.MultiIndex.from_tuples(
                    [(0, np.nan), (1, np.nan), (2, np.nan)]
                ),
                columns=pd.MultiIndex.from_tuples([(np.nan, 0), (np.nan, 1)]),
            ),
            id="generator_name_fuel_name_none",
        ),
    ],
)
def test_format_hourly_resolution(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    df: pd.DataFrame,
    generator: str,
    fuel_name: str,
    expected_output: pd.DataFrame,
) -> None:
    result = mocked_source_parameters_over_years_query._format_hourly_resolution(
        df, generator, fuel_name
    )
    assert_frame_equal(result, expected_output, check_column_type=False)


@pytest.mark.parametrize(
    "generation_results, generators, is_hours_resolution, expected_output",
    [
        pytest.param(
            {
                "gen1": pd.DataFrame(
                    {
                        0: [10.0, 10.0],
                        1: [10.0, 10.0],
                        2: [10.0, 10.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen2": pd.DataFrame(
                    {
                        0: [5.0, 5.0],
                        1: [5.0, 5.0],
                        2: [5.0, 5.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen3": pd.DataFrame(
                    {
                        0: [0.0, 0.0],
                        1: [0.0, 0.0],
                        2: [0.0, 0.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
            },
            ["gen1", "gen2", "gen3"],
            False,
            pd.DataFrame(
                [
                    [0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
                ],
                index=pd.Index(["fuel1"]),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                        ("gen3", 0),
                        ("gen3", 1),
                        ("gen3", 2),
                    ]
                ),
            ),
            id="base_case",
        ),
        pytest.param(
            {
                "gen1": pd.DataFrame(
                    {
                        0: [10.0, 10.0],
                        1: [10.0, 10.0],
                        2: [10.0, 10.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen2": pd.DataFrame(
                    {
                        0: [5.0, 5.0],
                        1: [5.0, 5.0],
                        2: [5.0, 5.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen3": pd.DataFrame(
                    {
                        0: [0.0, 0.0],
                        1: [0.0, 0.0],
                        2: [0.0, 0.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
            },
            ["gen1"],
            False,
            pd.DataFrame(
                [
                    [0.2, 0.2, 0.2],
                ],
                index=pd.Index(["fuel1"]),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ]
                ),
            ),
            id="only_gen_1",
        ),
        pytest.param(
            {
                "gen1": pd.DataFrame(
                    {
                        0: [10.0, 10.0],
                        1: [10.0, 10.0],
                        2: [10.0, 10.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen2": pd.DataFrame(
                    {
                        0: [5.0, 5.0],
                        1: [5.0, 5.0],
                        2: [5.0, 5.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen3": pd.DataFrame(
                    {
                        0: [0.0, 0.0],
                        1: [0.0, 0.0],
                        2: [0.0, 0.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
            },
            [],
            False,
            pd.DataFrame(),
            id="no_generators",
        ),
        pytest.param(
            {},
            ["gen1", "gen2", "gen3"],
            False,
            pd.DataFrame(),
            id="empty_results_df",
        ),
        pytest.param(
            {
                "gen1": pd.DataFrame(
                    {
                        0: [10.0, 10.0],
                        1: [10.0, 10.0],
                        2: [10.0, 10.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen2": pd.DataFrame(
                    {
                        0: [5.0, 5.0],
                        1: [5.0, 5.0],
                        2: [5.0, 5.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
                "gen3": pd.DataFrame(
                    {
                        0: [0.0, 0.0],
                        1: [0.0, 0.0],
                        2: [0.0, 0.0],
                    },
                    index=pd.Index([0, 1], name="Hour"),
                ),
            },
            ["gen1", "gen2", "gen3"],
            True,
            pd.DataFrame(
                [
                    [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                ],
                index=pd.MultiIndex.from_tuples([(0, "fuel1"), (1, "fuel1")]),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                        ("gen3", 0),
                        ("gen3", 1),
                        ("gen3", 2),
                    ],
                ),
            ),
            id="all_gens_hours_resolution",
        ),
    ],
)
def test__get_fuel_usage(
    fuel_mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generation_results: dict[str, pd.DataFrame],
    generators: list[str],
    is_hours_resolution: bool,
    expected_output: pd.DataFrame,
) -> None:
    fuel_mocked_source_parameters_over_years_query._generator_results = {
        "generation": generation_results
    }
    result = fuel_mocked_source_parameters_over_years_query._get_fuel_usage(
        generators, [], is_hours_resolution
    )
    assert_frame_equal(result, expected_output, check_column_type=False)


@pytest.mark.parametrize(
    "generators, level, is_hours_resolution, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "type",
            False,
            pd.DataFrame(
                data=[[0.2], [0.2], [0.2], [0.0], [0.0], [0.0]],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 0),
                        ("group1", 1),
                        ("group1", 2),
                        ("group2", 0),
                        ("group2", 1),
                        ("group2", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["fuel1"], name="Fuel"),
            ),
            id="all_gens_type",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "element",
            False,
            pd.DataFrame(
                data=[[0.2], [0.2], [0.2], [0.0], [0.0], [0.0]],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                        ("gen3", 0),
                        ("gen3", 1),
                        ("gen3", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["fuel1"], name="Fuel"),
            ),
            id="all_gens_elements",
        ),
        pytest.param(
            ["gen1"],
            "element",
            False,
            pd.DataFrame(
                data=[[0.2], [0.2], [0.2]],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["fuel1"], name="Fuel"),
            ),
            id="only_gen1",
        ),
        pytest.param(
            [],
            "element",
            False,
            pd.DataFrame(),
            id="empty_case",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "element",
            True,
            pd.DataFrame(
                {
                    "fuel1": [
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0, 0),
                        ("gen1", 0, 1),
                        ("gen1", 1, 0),
                        ("gen1", 1, 1),
                        ("gen1", 2, 0),
                        ("gen1", 2, 1),
                        ("gen3", 0, 0),
                        ("gen3", 0, 1),
                        ("gen3", 1, 0),
                        ("gen3", 1, 1),
                        ("gen3", 2, 0),
                        ("gen3", 2, 1),
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["fuel1"], name="Fuel"),
            ),
            id="all_gens_type_hour_resolution",
        ),
    ],
)
def test_get_fuel_usage(
    fuel_mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generators: list[str],
    is_hours_resolution: bool,
    level: Literal["type", "element"],
    expected_output: pd.DataFrame,
    generator_results_per_year_per_hour: dict[str, pd.DataFrame],
) -> None:
    fuel_mocked_source_parameters_over_years_query._generator_results = {
        "generation": generator_results_per_year_per_hour
    }
    with patch.object(
        fuel_mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=(generators, []),
    ):
        result = fuel_mocked_source_parameters_over_years_query.get_fuel_usage(
            level, None, None, is_hours_resolution
        )
        assert_frame_equal(result, expected_output)
