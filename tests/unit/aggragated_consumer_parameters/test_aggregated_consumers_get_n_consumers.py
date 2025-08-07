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
from pandas.testing import assert_series_equal

from zefir_analytics._engine.data_queries.aggregated_consumer_parameters_over_years import (
    AggregatedConsumerParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "aggregate_name, expected_result",
    [
        pytest.param(
            "aggr1",
            pd.Series([10, 10, 10], name="N_consumers"),
            id="aggr_1_str",
        ),
        pytest.param(
            ["aggr1", "aggr2"],
            {
                "aggr1": pd.Series([10, 10, 10], name="N_consumers"),
                "aggr2": pd.Series([100, 100, 100], name="N_consumers"),
            },
            id="list_of_aggr",
        ),
        pytest.param(
            None,
            {
                "aggr1": pd.Series([10, 10, 10], name="N_consumers"),
                "aggr2": pd.Series([100, 100, 100], name="N_consumers"),
            },
            id="None",
        ),
        pytest.param(
            ["aggr1", "aggr56"],
            {
                "aggr1": pd.Series([10, 10, 10], name="N_consumers"),
            },
            id="list_of_aggr_with_one_incorrect",
        ),
        pytest.param(
            ["re121", "aggr_not_in_network"],
            pd.DataFrame(),
            id="list_of_aggr_with_all_incorrect",
        ),
        pytest.param(
            "not_aggr",
            pd.DataFrame(),
            id="str_aggr_not_in_network",
        ),
    ],
)
def test_get_n_consumers(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    aggregate_name: str | list[str] | None,
    expected_result: pd.Series | dict[str, pd.Series],
) -> None:
    result = mocked_aggr_parameters_over_year_query.get_n_consumers(aggregate_name)
    if isinstance(expected_result, pd.Series):
        assert_series_equal(result, expected_result)
    else:
        for aggr_name, expected_df in expected_result.items():
            assert_series_equal(result[aggr_name], expected_df)
