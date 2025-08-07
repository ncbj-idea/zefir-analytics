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

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "generators, storages, is_hours_resolution, generator_results, storage_results, expected_output",
    [
        pytest.param(
            ["gen1"],
            ["stor1"],
            True,
            pd.DataFrame(
                {
                    ("gen1", 0): [0.0, 0.0],
                    ("gen1", 1): [1.0, 0.0],
                    ("gen1", 2): [2.0, 0.0],
                },
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            pd.DataFrame(
                {
                    ("stor1", 0): [0.0, 10.0],
                    ("stor1", 1): [0.0, 20.0],
                    ("stor1", 2): [0.0, 30.0],
                },
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            pd.DataFrame(
                {
                    ("gen1", 0): [0.0, 0.0],
                    ("gen1", 1): [1.0, 0.0],
                    ("gen1", 2): [2.0, 0.0],
                    ("stor1", 0): [0.0, 10.0],
                    ("stor1", 1): [0.0, 20.0],
                    ("stor1", 2): [0.0, 30.0],
                },
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="yearly_shape_hour_resolution_on",
        ),
        pytest.param(
            ["gen1"],
            ["stor1"],
            False,
            pd.DataFrame(
                {
                    ("gen1", 0): [0.0, 0.0],
                    ("gen1", 1): [1.0, 0.0],
                    ("gen1", 2): [2.0, 0.0],
                },
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            pd.DataFrame(
                {
                    ("stor1", 0): [0.0, 10.0],
                    ("stor1", 1): [0.0, 20.0],
                    ("stor1", 2): [0.0, 30.0],
                },
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            pd.DataFrame(
                {
                    ("gen1", 0): [0.0, 0.0],
                    ("gen1", 1): [1.0, 0.0],
                    ("gen1", 2): [2.0, 0.0],
                    ("stor1", 0): [0.0, 10.0],
                    ("stor1", 1): [0.0, 20.0],
                    ("stor1", 2): [0.0, 30.0],
                },
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
            )
            * 2.0,
            id="yearly_shape_hour_resolution_off",
        ),
        pytest.param(
            ["gen1"],
            ["stor1"],
            True,
            pd.DataFrame(
                [
                    [0, 1, 10],
                    [0, 10, 100],
                    [0, 2, 20],
                    [0, 20, 200],
                    [0, 3, 30],
                    [0, 30, 300],
                ],
                index=pd.MultiIndex.from_product(
                    [[0, 1, 2], ["ET1", "ET2"]], names=["Hour", "Energy Type"]
                ),
                columns=pd.MultiIndex.from_product([["gen1"], ["0", "1", "2"]]),
            ),
            pd.DataFrame(
                {
                    ("stor1", "0"): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ("stor1", "1"): [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
                    ("stor1", "2"): [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "ET1"),
                        (1, "ET1"),
                        (2, "ET1"),
                        (0, "ET2"),
                        (1, "ET2"),
                        (2, "ET2"),
                    ],
                    names=["Hour", "Energy Type"],
                ),
            ),
            pd.DataFrame(
                [
                    [0, 1, 10, 0.0, 10.0, 100.0],
                    [0, 10, 100, 0.0, 0.0, 0.0],
                    [0, 2, 20, 0.0, 20.0, 200.0],
                    [0, 20, 200, 0.0, 0.0, 0.0],
                    [0, 3, 30, 0.0, 30.0, 300.0],
                    [0, 30, 300, 0.0, 0.0, 0.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "ET1"),
                        (0, "ET2"),
                        (1, "ET1"),
                        (1, "ET2"),
                        (2, "ET1"),
                        (2, "ET2"),
                    ],
                    names=["Hour", "Energy Type"],
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", "0"),
                        ("gen1", "1"),
                        ("gen1", "2"),
                        ("stor1", "0"),
                        ("stor1", "1"),
                        ("stor1", "2"),
                    ]
                ),
            ),
            id="hourly_shape_hour_resolution_on",
        ),
        pytest.param(
            ["gen1"],
            ["stor1"],
            False,
            pd.DataFrame(
                [
                    [0, 1, 10],
                    [0, 10, 100],
                    [0, 2, 20],
                    [0, 20, 200],
                    [0, 3, 30],
                    [0, 30, 300],
                ],
                index=pd.MultiIndex.from_product(
                    [[0, 1, 2], ["ET1", "ET2"]], names=["Hour", "Energy Type"]
                ),
                columns=pd.MultiIndex.from_product([["gen1"], ["0", "1", "2"]]),
            ),
            pd.DataFrame(
                {
                    ("stor1", "0"): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ("stor1", "1"): [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
                    ("stor1", "2"): [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "ET1"),
                        (1, "ET1"),
                        (2, "ET1"),
                        (0, "ET2"),
                        (1, "ET2"),
                        (2, "ET2"),
                    ],
                    names=["Hour", "Energy Type"],
                ),
            ),
            pd.DataFrame(
                [
                    [0, 1, 10, 0.0, 10.0, 100.0],
                    [0, 10, 100, 0.0, 0.0, 0.0],
                    [0, 2, 20, 0.0, 20.0, 200.0],
                    [0, 20, 200, 0.0, 0.0, 0.0],
                    [0, 3, 30, 0.0, 30.0, 300.0],
                    [0, 30, 300, 0.0, 0.0, 0.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "ET1"),
                        (0, "ET2"),
                        (1, "ET1"),
                        (1, "ET2"),
                        (2, "ET1"),
                        (2, "ET2"),
                    ],
                    names=["Hour", "Energy Type"],
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", "0"),
                        ("gen1", "1"),
                        ("gen1", "2"),
                        ("stor1", "0"),
                        ("stor1", "1"),
                        ("stor1", "2"),
                    ]
                ),
            )
            * 2.0,
            id="hourly_shape_hour_resolution_off",
        ),
        pytest.param(
            [],
            [],
            True,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            id="empty_case",
        ),
    ],
)
def test__get_generation_sum(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generators: list[str],
    storages: list[str],
    is_hours_resolution: bool,
    generator_results: pd.DataFrame,
    storage_results: pd.DataFrame,
    expected_output: pd.DataFrame,
) -> None:
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_get_generator_et_results",
        return_value=generator_results,
    ), patch.object(
        mocked_source_parameters_over_years_query,
        "_get_storage_results",
        return_value=storage_results,
    ), patch.object(
        mocked_source_parameters_over_years_query, "_hourly_scale", 2.0
    ):
        result = mocked_source_parameters_over_years_query._get_generation_sum(
            generators=generators,
            storages=storages,
            is_hours_resolution=is_hours_resolution,
        )
        assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filtered_generators, filtered_storages, level, is_hours_resolution, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "type",
            False,
            pd.DataFrame(
                {
                    "ET1": [20.0, 0.0, 30.0, 20.0, 22.0, 24.0, 0.0, 0.0, 4.0],
                    "ET2": [20.0, 0.0, 30.0, 20.0, 22.0, 24.0, 0.0, 6.0, 14.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 0),
                        ("group1", 1),
                        ("group1", 2),
                        ("group2", 0),
                        ("group2", 1),
                        ("group2", 2),
                        ("group3", 0),
                        ("group3", 1),
                        ("group3", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_elements",
        ),
        pytest.param(
            ["gen3"],
            ["stor2"],
            "type",
            False,
            pd.DataFrame(
                {
                    "ET1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "ET2": [0.0, 0.0, 0.0, 0.0, 6.0, 14.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 0),
                        ("group1", 1),
                        ("group1", 2),
                        ("group3", 0),
                        ("group3", 1),
                        ("group3", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="filtered_gen3_and_stor2",
        ),
        pytest.param(
            ["gen3", "gen1"],
            ["stor1"],
            "element",
            True,
            pd.DataFrame(
                {
                    "ET1": [
                        10.0,
                        10.0,
                        0.0,
                        0,
                        15.0,
                        15.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2.0,
                        2.0,
                    ],
                    "ET2": [
                        10.0,
                        10.0,
                        0.0,
                        0.0,
                        15.0,
                        15.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
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
                        ("stor1", 0, 0),
                        ("stor1", 0, 1),
                        ("stor1", 1, 0),
                        ("stor1", 1, 1),
                        ("stor1", 2, 0),
                        ("stor1", 2, 1),
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="filtered_gen3_gen1_stor_1_hour_resolution",
        ),
        pytest.param(
            ["gen1"],
            [],
            "element",
            True,
            pd.DataFrame(
                {
                    "ET1": [
                        10.0,
                        10.0,
                        0.0,
                        0,
                        15.0,
                        15.0,
                    ],
                    "ET2": [
                        10.0,
                        10.0,
                        0.0,
                        0.0,
                        15.0,
                        15.0,
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
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="no_storages_element_hour_resolution",
        ),
        pytest.param(
            [],
            ["stor1", "stor2"],
            "type",
            False,
            pd.DataFrame(
                {
                    "ET1": [0.0, 0.0, 4.0],
                    "ET2": [0.0, 6.0, 14.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group3", 0),
                        ("group3", 1),
                        ("group3", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="no_generators_type_year_resolution",
        ),
        pytest.param(
            [],
            [],
            "type",
            False,
            pd.DataFrame(),
            id="empty_case",
        ),
    ],
)
def test_get_generation_sum(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generator_results_per_et_per_year_per_hour: dict[str, pd.DataFrame],
    storage_results_per_year_per_hour: dict[str, pd.DataFrame],
    filtered_generators: list[str],
    filtered_storages: list[str],
    is_hours_resolution: bool,
    level: Literal["type", "element"],
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._generator_results = {
        "generation_per_energy_type": generator_results_per_et_per_year_per_hour
    }
    mocked_source_parameters_over_years_query._storage_results = {
        "generation": storage_results_per_year_per_hour
    }
    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen1", "gen3"},
        "group2": {"gen2"},
        "group3": {"stor1", "stor2"},
    }

    stor1 = MagicMock(energy_source_type="Storage_ET1")
    stor2 = MagicMock(energy_source_type="Storage_ET2")
    stor_type_1 = MagicMock(energy_type="ET1")
    stor_type_2 = MagicMock(energy_type="ET2")

    mocked_source_parameters_over_years_query._network.storages = {
        "stor1": stor1,
        "stor2": stor2,
    }

    mocked_source_parameters_over_years_query._network.storage_types = {
        "Storage_ET1": stor_type_1,
        "Storage_ET2": stor_type_2,
    }
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=(filtered_generators, filtered_storages),
    ):
        result = mocked_source_parameters_over_years_query.get_generation_sum(
            level, None, None, is_hours_resolution
        )
        assert_frame_equal(result, expected_output)
