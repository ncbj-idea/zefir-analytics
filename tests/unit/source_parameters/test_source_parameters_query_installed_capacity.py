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
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "generators, storages, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "gen2": [10.0, 20.0, 30.0],
                    "gen3": [0.0, 0.0, 0.0],
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [5.0, 5.0, 5.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="all_gens_and_stors",
        ),
        pytest.param(
            ["gen1"],
            ["stor1"],
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "stor1": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="gen1_stor_1_filtered",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            [],
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "gen2": [10.0, 20.0, 30.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="only_gens",
        ),
        pytest.param(
            [],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [5.0, 5.0, 5.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="only_stors",
        ),
        pytest.param(
            [],
            [],
            pd.DataFrame(index=pd.Index([0, 1, 2], name="Year"), columns=[]),
            id="empty_case",
        ),
        pytest.param(
            ["gen4", "gen5", "gen6"],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [5.0, 5.0, 5.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="gens_not_in_results",
        ),
        pytest.param(
            ["gen2", "gen5", "gen3"],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    "gen2": [10.0, 20.0, 30.0],
                    "gen3": [0.0, 0.0, 0.0],
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [5.0, 5.0, 5.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="gen5_not_in_results",
        ),
    ],
)
def test__get_installed_capacity(
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generators: list[str],
    storages: list[str],
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year}
    }
    mocked_source_parameters_over_years_query._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year}
    }
    result = mocked_source_parameters_over_years_query._get_installed_capacity(
        generators, storages
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filtered_generators, filtered_storages,  level, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "type",
            pd.DataFrame(
                [11.0, 22.0, 33.0, 5.0, 5.0, 5.0],
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
                columns=pd.Index([0], name="Energy Type"),
            ),
            id="all_type",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "element",
            pd.DataFrame(
                [
                    1.0,
                    2.0,
                    3.0,
                    10.0,
                    20.0,
                    30.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0,
                    5.0,
                    5.0,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                        ("gen2", 0),
                        ("gen2", 1),
                        ("gen2", 2),
                        ("gen3", 0),
                        ("gen3", 1),
                        ("gen3", 2),
                        ("stor1", 0),
                        ("stor1", 1),
                        ("stor1", 2),
                        ("stor2", 0),
                        ("stor2", 1),
                        ("stor2", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index([0], name="Energy Type"),
            ),
            id="all_elements",
        ),
        pytest.param(
            ["gen1", "gen2"],
            ["stor1"],
            "element",
            pd.DataFrame(
                [
                    1.0,
                    2.0,
                    3.0,
                    10.0,
                    20.0,
                    30.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                        ("gen2", 0),
                        ("gen2", 1),
                        ("gen2", 2),
                        ("stor1", 0),
                        ("stor1", 1),
                        ("stor1", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index([0], name="Energy Type"),
            ),
            id="gen1_gen2_stor1_filtered_elements",
        ),
        pytest.param(
            [],
            ["stor1", "stor2"],
            "type",
            pd.DataFrame(
                [5.0, 5.0, 5.0],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group2", 0),
                        ("group2", 1),
                        ("group2", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index([0], name="Energy Type"),
            ),
            id="only_storages_type",
        ),
        pytest.param(
            [],
            [],
            "element",
            pd.DataFrame(
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.MultiIndex.from_tuples([], names=[None, "Year"]),
            ),
            id="empty_case",
        ),
    ],
)
def test_get_installed_capacity(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
    filtered_generators: list[str],
    filtered_storages: list[str],
    level: Literal["type", "element"],
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year}
    }
    mocked_source_parameters_over_years_query._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year}
    }

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen1", "gen2", "gen3"},
        "group2": {"stor1", "stor2"},
    }

    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=(filtered_generators, filtered_storages),
    ):
        result = mocked_source_parameters_over_years_query.get_installed_capacity(
            level, None, None
        )
        assert_frame_equal(result, expected_output, check_column_type=False)
