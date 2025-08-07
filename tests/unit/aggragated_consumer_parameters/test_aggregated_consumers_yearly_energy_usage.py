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

from zefir_analytics._engine.data_queries.aggregated_consumer_parameters_over_years import (
    AggregatedConsumerParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "aggregate_name, expected_result",
    [
        pytest.param(
            "aggr1",
            pd.DataFrame(
                {"ET1": [2, 2, 2], "ET2": [10, 10, 10]},
            ),
            id="aggr_1_str",
        ),
        pytest.param(
            ["aggr1", "aggr2"],
            {
                "aggr1": pd.DataFrame(
                    {"ET1": [2, 2, 2], "ET2": [10, 10, 10]},
                ),
                "aggr2": pd.DataFrame(
                    {"ET1": [2, 2, 2], "ET3": [100, 100, 100]},
                ),
            },
            id="list_of_aggr",
        ),
        pytest.param(
            None,
            {
                "aggr1": pd.DataFrame(
                    {"ET1": [2, 2, 2], "ET2": [10, 10, 10]},
                ),
                "aggr2": pd.DataFrame(
                    {"ET1": [2, 2, 2], "ET3": [100, 100, 100]},
                ),
            },
            id="None",
        ),
        pytest.param(
            ["aggr1", "aggr1232"],
            {
                "aggr1": pd.DataFrame(
                    {"ET1": [2, 2, 2], "ET2": [10, 10, 10]},
                ),
            },
            id="list_of_aggr_with_one_incorrect",
        ),
        pytest.param(
            ["aggr11234", "aggr_not_in_network"],
            pd.DataFrame(),
            id="list_of_aggr_with_all_incorrect",
        ),
        pytest.param(
            "qw31",
            pd.DataFrame(),
            id="str_aggr_not_in_network",
        ),
    ],
)
def test_get_yearly_energy_usage(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    aggregate_name: str | list[str] | None,
    expected_result: pd.DataFrame | dict[str, pd.DataFrame],
) -> None:
    result = mocked_aggr_parameters_over_year_query.get_yearly_energy_usage(
        aggregate_name
    )
    if isinstance(expected_result, pd.DataFrame):
        assert_frame_equal(result, expected_result)
    else:
        for aggr_name, expected_df in expected_result.items():
            assert_frame_equal(result[aggr_name], expected_df)


@pytest.mark.parametrize(
    "aggregate_name, expected_result",
    [
        pytest.param(
            "aggr1",
            pd.DataFrame(
                {"ET1": [20, 20, 20], "ET2": [100, 100, 100]},
            ),
            id="aggr_1_str",
        ),
        pytest.param(
            ["aggr1", "aggr2"],
            {
                "aggr1": pd.DataFrame(
                    {"ET1": [20, 20, 20], "ET2": [100, 100, 100]},
                ),
                "aggr2": pd.DataFrame(
                    {"ET1": [200, 200, 200], "ET3": [10000, 10000, 10000]},
                ),
            },
            id="list_of_aggr",
        ),
        pytest.param(
            None,
            {
                "aggr1": pd.DataFrame(
                    {"ET1": [20, 20, 20], "ET2": [100, 100, 100]},
                ),
                "aggr2": pd.DataFrame(
                    {"ET1": [200, 200, 200], "ET3": [10000, 10000, 10000]},
                ),
            },
            id="None",
        ),
        pytest.param(
            ["aggr1", "aggr1232"],
            {
                "aggr1": pd.DataFrame(
                    {"ET1": [20, 20, 20], "ET2": [100, 100, 100]},
                ),
            },
            id="list_of_aggr_with_one_incorrect",
        ),
        pytest.param(
            ["aggr11234", "aggr_not_in_network"],
            pd.DataFrame(),
            id="list_of_aggr_with_all_incorrect",
        ),
        pytest.param(
            "3aggr",
            pd.DataFrame(),
            id="str_aggr_not_in_network",
        ),
    ],
)
def test_get_total_yearly_energy_usage(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    aggregate_name: str | list[str] | None,
    expected_result: pd.DataFrame | dict[str, pd.DataFrame],
) -> None:
    result = mocked_aggr_parameters_over_year_query.get_total_yearly_energy_usage(
        aggregate_name
    )
    if isinstance(expected_result, pd.DataFrame):
        assert_frame_equal(result, expected_result)
    else:
        for aggr_name, expected_df in expected_result.items():
            assert_frame_equal(result[aggr_name], expected_df)
