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

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.line_parameters_over_years import (
    LineParametersOverYearsQuery,
)


@pytest.fixture
def mocked_transmission_line_parameters(
    mocked_line_parameters_over_year_query: LineParametersOverYearsQuery,
) -> LineParametersOverYearsQuery:
    line_1 = MagicMock(transmission_fee="TF_1")
    line_1.name = "line_1"

    line_2 = MagicMock(transmission_fee="TF_2")
    line_2.name = "line_2"

    line_3 = MagicMock(transmission_fee=None)
    line_3.name = "line_3"

    tf1 = MagicMock(fee=pd.Series([10, 20, 30]))
    tf1.name = "TF_1"

    tf2 = MagicMock(fee=pd.Series([100, 200, 300]))
    tf2.name = "TF_2"

    mocked_line_parameters_over_year_query._network.lines = {
        line.name: line for line in (line_1, line_2, line_3)
    }

    mocked_line_parameters_over_year_query._network.transmission_fees = {
        tf.name: tf for tf in (tf1, tf2)
    }
    mocked_line_parameters_over_year_query._hour_sample = np.array([0, 1, 2])
    mocked_line_parameters_over_year_query._years_binding = None
    return mocked_line_parameters_over_year_query


@pytest.mark.parametrize(
    "df, column_name, operation, expected_output",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "0": [1, 2, 3, 5, 6],
                    "1": [4, 5, 6, 7, 8],
                    "2": [4, 5, 6, 7, 8],
                    "3": [2, 2, 2, 2, 2],
                }
            ),
            "Sum",
            "sum",
            pd.DataFrame(
                {"Sum": [17, 30, 30, 10]}, index=pd.RangeIndex(0, 4, name="Year")
            ),
            id="sum_operation",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "0": [1, 1, 1, 1, 1],
                    "1": [2, 2, 2, 2, 2],
                    "2": [3, 3, 3, 3, 3],
                    "3": [4, 4, 4, 4, 4],
                }
            ),
            "Mean",
            "mean",
            pd.DataFrame(
                {"Mean": [1.0, 2.0, 3.0, 4.0]}, index=pd.RangeIndex(0, 4, name="Year")
            ),
            id="mean_operation",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "0": [1, 41, 2, 4, 11],
                    "1": [2, 5, 25, 5, 2],
                    "2": [32, 12, 3, 31, 12],
                    "3": [12, 14, 5, 42, 14],
                }
            ),
            "Column_name_mean_or_not_mean",
            "mean",
            pd.DataFrame(
                {"Column_name_mean_or_not_mean": [11.8, 7.8, 18.0, 17.4]},
                index=pd.RangeIndex(0, 4, name="Year"),
            ),
            id="mean_operation_custom_name",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "0": [1, 1, 1, 1, 10],
                    "1": [2, 2, 20, 2, 2],
                    "2": [3, 30, 3, 3, 3],
                    "3": [40, 4, 4, 4, 4],
                }
            ),
            "Max",
            "max",
            pd.DataFrame(
                {"Max": [10, 20, 30, 40]}, index=pd.RangeIndex(0, 4, name="Year")
            ),
            id="max_operation",
        ),
    ],
)
def test_get_yearly_summary(
    mocked_transmission_line_parameters: LineParametersOverYearsQuery,
    df: pd.DataFrame,
    column_name: str,
    operation: str,
    expected_output: pd.DataFrame,
) -> None:
    result = mocked_transmission_line_parameters._get_yearly_summary(
        df, column_name, operation
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "line_name, expected_output",
    [
        pytest.param(
            "line_1",
            pd.DataFrame(
                {"Transmission fee total cost": [600.0, 600.0, 600.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_1_valid_data",
        ),
        pytest.param(
            "line_2",
            pd.DataFrame(
                {"Transmission fee total cost": [2500.0, 2500.0, 2500.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_2_valid_data",
        ),
        pytest.param(
            "line_6",
            pd.DataFrame(),
            id="line_not_in_network",
        ),
        pytest.param(
            "line_3",
            pd.DataFrame(
                {"Transmission fee total cost": [0.0, 0.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="transmission_fee_None",
        ),
    ],
)
def test__get_transmission_fee(
    mocked_transmission_line_parameters: LineParametersOverYearsQuery,
    line_name: str,
    expected_output: pd.DataFrame,
) -> None:
    result = mocked_transmission_line_parameters._get_transmission_fee(line_name)
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "line_name, expected_output",
    [
        pytest.param(
            "line_1",
            pd.DataFrame(
                {"Transmission fee total cost": [600.0, 600.0, 600.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_1_only",
        ),
        pytest.param(
            "line_3",
            pd.DataFrame(
                {"Transmission fee total cost": [0.0, 0.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="line_3_only_transmission_fee_None",
        ),
        pytest.param(
            ["line_1", "line_2"],
            {
                "line_2": pd.DataFrame(
                    {"Transmission fee total cost": [2500.0, 2500.0, 2500.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "line_1": pd.DataFrame(
                    {"Transmission fee total cost": [600.0, 600.0, 600.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
            },
            id="list_of_lines",
        ),
        pytest.param(
            None,
            {
                "line_2": pd.DataFrame(
                    {"Transmission fee total cost": [2500.0, 2500.0, 2500.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "line_1": pd.DataFrame(
                    {"Transmission fee total cost": [600.0, 600.0, 600.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "line_3": pd.DataFrame(
                    {"Transmission fee total cost": [0.0, 0.0, 0.0]},
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
            },
            id="all_lines",
        ),
    ],
)
def test_get_transmission_fee(
    mocked_transmission_line_parameters: LineParametersOverYearsQuery,
    line_name: str | list[str] | None,
    expected_output: pd.DataFrame | dict[str, pd.DataFrame],
) -> None:
    result = mocked_transmission_line_parameters.get_transmission_fee(line_name)
    if isinstance(result, dict):
        for line, df in result.items():
            assert_frame_equal(df, expected_output[line])
    else:
        assert_frame_equal(result, expected_output)
