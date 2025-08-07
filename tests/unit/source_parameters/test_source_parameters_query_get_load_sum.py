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
    "filtered_storages, level, is_hours_resolution, expected_output",
    [
        pytest.param(
            ["stor1", "stor2"],
            "type",
            False,
            pd.DataFrame(
                {
                    "ET1": [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                    "ET2": [0.0, 0.0, 0.0, 0.0, 12.0, 28.0],
                },
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
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_type",
        ),
        pytest.param(
            ["stor1", "stor2"],
            "element",
            False,
            pd.DataFrame(
                {
                    "ET1": [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                    "ET2": [0.0, 0.0, 0.0, 0.0, 12.0, 28.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("stor1", 0),
                        ("stor1", 1),
                        ("stor1", 2),
                        ("stor2", 0),
                        ("stor2", 1),
                        ("stor2", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_elements",
        ),
        pytest.param(
            ["stor1", "stor2"],
            "type",
            True,
            pd.DataFrame(
                {
                    "ET1": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "ET2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 7.0, 7.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 0, 0),
                        ("group1", 0, 1),
                        ("group1", 1, 0),
                        ("group1", 1, 1),
                        ("group1", 2, 0),
                        ("group1", 2, 1),
                        ("group2", 0, 0),
                        ("group2", 0, 1),
                        ("group2", 1, 0),
                        ("group2", 1, 1),
                        ("group2", 2, 0),
                        ("group2", 2, 1),
                    ],
                    names=["Network element type", "Year", "Hour"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_elements_hourly_scale",
        ),
        pytest.param(
            ["stor1"],
            "element",
            True,
            pd.DataFrame(
                {"ET1": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("stor1", 0, 0),
                        ("stor1", 0, 1),
                        ("stor1", 1, 0),
                        ("stor1", 1, 1),
                        ("stor1", 2, 0),
                        ("stor1", 2, 1),
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["ET1"], name="Energy Type"),
            ),
            id="stor1_filtered_hourly_scale",
        ),
        pytest.param(
            [],
            "element",
            True,
            pd.DataFrame(),
            id="empty_filter_list",
        ),
    ],
)
def test_get_load_sum(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    storage_results_per_year_per_hour: dict[str, pd.DataFrame],
    filtered_storages: list[str],
    is_hours_resolution: bool,
    level: Literal["type", "element"],
    expected_output: pd.DataFrame,
) -> None:
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
    mocked_source_parameters_over_years_query._storage_results = {
        "load": storage_results_per_year_per_hour
    }
    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"stor1"},
        "group2": {"stor2"},
    }
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=([], filtered_storages),
    ), patch.object(mocked_source_parameters_over_years_query, "_hourly_scale", 2.0):
        result = mocked_source_parameters_over_years_query.get_load_sum(
            level, None, None, is_hours_resolution
        )
        assert_frame_equal(result, expected_output)
