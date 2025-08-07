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

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.line_parameters_over_years import (
    LineParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "line_name, expected_output",
    [
        pytest.param(
            "line_1",
            pd.DataFrame(
                {"Total energy volume": [30.0, 30.0, 30.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_1_valid_data",
        ),
        pytest.param(
            "line_2",
            pd.DataFrame(
                {"Total energy volume": [10.0, 10.0, 10.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_2_valid_data",
        ),
        pytest.param(
            "line_6",
            pd.DataFrame(),
            id="line_not_in_results",
        ),
    ],
)
def test__get_flow(
    mocked_line_parameters_over_year_query: LineParametersOverYearsQuery,
    line_name: str,
    expected_output: pd.DataFrame,
) -> None:
    result = mocked_line_parameters_over_year_query._get_flow(line_name)
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "line_name, expected_output",
    [
        pytest.param(
            "line_1",
            pd.DataFrame(
                {"Total energy volume": [10.0] * 9},
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (1, 0),
                        (1, 1),
                        (1, 2),
                        (2, 0),
                        (2, 1),
                        (2, 2),
                    ],
                    names=["Year", "Hour"],
                ),
            ),
            id="line_1_valid_data",
        ),
        pytest.param(
            "line_2",
            pd.DataFrame(
                {"Total energy volume": [0.0, 5.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 5.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (1, 0),
                        (1, 1),
                        (1, 2),
                        (2, 0),
                        (2, 1),
                        (2, 2),
                    ],
                    names=["Year", "Hour"],
                ),
            ),
            id="line_2_valid_data",
        ),
        pytest.param(
            "line_6",
            pd.DataFrame(),
            id="line_not_in_results",
        ),
    ],
)
def test_get_flow_hourly(
    mocked_line_parameters_over_year_query: LineParametersOverYearsQuery,
    line_name: str,
    expected_output: pd.DataFrame,
) -> None:
    result = mocked_line_parameters_over_year_query._get_flow_hourly(line_name)
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "line_name, is_hours_resolution, expected_output",
    [
        pytest.param(
            "line_1",
            False,
            pd.DataFrame(
                {"Total energy volume": [30.0, 30.0, 30.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_1_only",
        ),
        pytest.param(
            ["line_1", "line_3"],
            False,
            {
                "line_1": pd.DataFrame(
                    {"Total energy volume": [30.0, 30.0, 30.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "line_3": pd.DataFrame(
                    {"Total energy volume": [2.0, 2.0, 2.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
            },
            id="list_of_lines",
        ),
        pytest.param(
            None,
            False,
            {
                "line_2": pd.DataFrame(
                    {"Total energy volume": [10.0, 10.0, 10.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "line_1": pd.DataFrame(
                    {"Total energy volume": [30.0, 30.0, 30.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "line_3": pd.DataFrame(
                    {"Total energy volume": [2.0, 2.0, 2.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
            },
            id="all_lines",
        ),
        pytest.param("line_6", False, pd.DataFrame(), id="line_not_in_results"),
        pytest.param(
            ["line_6", "line_5"],
            False,
            {"line_6": pd.DataFrame(), "line_5": pd.DataFrame()},
            id="lines_not_in_results",
        ),
        pytest.param(
            "line_1",
            True,
            pd.DataFrame(
                {"Total energy volume": [10.0] * 9},
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (1, 0),
                        (1, 1),
                        (1, 2),
                        (2, 0),
                        (2, 1),
                        (2, 2),
                    ],
                    names=["Year", "Hour"],
                ),
            ),
            id="line_1_only_hour_resolution",
        ),
        pytest.param(
            ["line_1", "line_3"],
            True,
            {
                "line_1": pd.DataFrame(
                    {"Total energy volume": [10.0] * 9},
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                            (2, 0),
                            (2, 1),
                            (2, 2),
                        ],
                        names=["Year", "Hour"],
                    ),
                ),
                "line_3": pd.DataFrame(
                    {
                        "Total energy volume": [
                            2.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                        ]
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                            (2, 0),
                            (2, 1),
                            (2, 2),
                        ],
                        names=["Year", "Hour"],
                    ),
                ),
            },
            id="list_of_lines_hour_resolution",
        ),
        pytest.param(
            None,
            True,
            {
                "line_2": pd.DataFrame(
                    {
                        "Total energy volume": [
                            0.0,
                            5.0,
                            5.0,
                            0.0,
                            5.0,
                            5.0,
                            0.0,
                            5.0,
                            5.0,
                        ]
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                            (2, 0),
                            (2, 1),
                            (2, 2),
                        ],
                        names=["Year", "Hour"],
                    ),
                ),
                "line_1": pd.DataFrame(
                    {"Total energy volume": [10.0] * 9},
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                            (2, 0),
                            (2, 1),
                            (2, 2),
                        ],
                        names=["Year", "Hour"],
                    ),
                ),
                "line_3": pd.DataFrame(
                    {
                        "Total energy volume": [
                            2.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                        ]
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                            (2, 0),
                            (2, 1),
                            (2, 2),
                        ],
                        names=["Year", "Hour"],
                    ),
                ),
            },
            id="all_lines_hour_resolution",
        ),
        pytest.param(
            ["line_6", "line_5"],
            True,
            {"line_6": pd.DataFrame(), "line_5": pd.DataFrame()},
            id="lines_not_in_results_hour_resolution",
        ),
        pytest.param("line_6", True, pd.DataFrame(), id="line_not_in_results"),
    ],
)
def test_get_flow(
    mocked_line_parameters_over_year_query: LineParametersOverYearsQuery,
    line_name: str | list[str] | None,
    expected_output: pd.DataFrame | dict[str, pd.DataFrame],
    is_hours_resolution: bool,
) -> None:
    result = mocked_line_parameters_over_year_query.get_flow(
        line_name, is_hours_resolution
    )
    if isinstance(result, dict):
        for line, df in result.items():
            assert_frame_equal(df, expected_output[line])
    else:
        assert_frame_equal(result, expected_output)
