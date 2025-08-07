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
    "filtered_generators, level, is_hours_resolution, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "type",
            False,
            pd.DataFrame(
                {
                    "ET1": [40.0, 0.0, 60.0, 40.0, 44.0, 48.0],
                    "ET2": [40.0, 0.0, 60.0, 40.0, 44.0, 48.0],
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
            ["gen1", "gen2", "gen3"],
            "element",
            False,
            pd.DataFrame(
                {
                    "ET1": [40.0, 0.0, 60.0, 40.0, 44.0, 48.0, 0.0, 0.0, 0.0],
                    "ET2": [40.0, 0.0, 60.0, 40.0, 44.0, 48.0, 0.0, 0.0, 0.0],
                },
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
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_elements",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "type",
            True,
            pd.DataFrame(
                {
                    "ET1": [
                        10.0,
                        10.0,
                        0.0,
                        0.0,
                        15.0,
                        15.0,
                        10.0,
                        10.0,
                        11.0,
                        11.0,
                        12.0,
                        12.0,
                    ],
                    "ET2": [
                        10.0,
                        10.0,
                        0.0,
                        0.0,
                        15.0,
                        15.0,
                        10.0,
                        10.0,
                        11.0,
                        11.0,
                        12.0,
                        12.0,
                    ],
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
            ["gen1"],
            "element",
            True,
            pd.DataFrame(
                {
                    "ET1": [
                        10.0,
                        10.0,
                        0.0,
                        0.0,
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
            id="gen1_filtered_hourly_scale",
        ),
        pytest.param(
            [],
            "element",
            True,
            pd.DataFrame(),
            id="empty_filter_list",
        ),
        pytest.param(
            ["gen2", "gen3"],
            "type",
            False,
            pd.DataFrame(
                {
                    "ET1": [0.0, 0.0, 0.0, 40.0, 44.0, 48.0],
                    "ET2": [0.0, 0.0, 0.0, 40.0, 44.0, 48.0],
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
            id="gen2_gen3_filtered_type",
        ),
    ],
)
def test_get_generation_sum(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generator_results_per_et_per_year_per_hour: dict[str, pd.DataFrame],
    filtered_generators: list[str],
    is_hours_resolution: bool,
    level: Literal["type", "element"],
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._generator_results = {
        "dump_energy_per_energy_type": generator_results_per_et_per_year_per_hour
    }
    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen1", "gen3"},
        "group2": {"gen2"},
    }
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=(filtered_generators, []),
    ), patch.object(mocked_source_parameters_over_years_query, "_hourly_scale", 2.0):
        result = mocked_source_parameters_over_years_query.get_dump_energy_sum(
            level, None, None, is_hours_resolution
        )
        assert_frame_equal(result, expected_output)
