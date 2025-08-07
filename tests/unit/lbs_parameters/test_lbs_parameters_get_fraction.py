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

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.lbs_parameters_over_years import (
    LbsParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "lbs_name, aggregated_consumers, is_year_binding, years_binding, expected_result",
    [
        pytest.param(
            "lbs1",
            {
                "aggr1": MagicMock(available_stacks=["lbs1"]),
                "aggr2": MagicMock(available_stacks=["lbs2"]),
            },
            True,
            None,
            pd.DataFrame(
                {
                    "aggr1": {0: 10.0, 1: 0.0, 2: 0.0},
                    "aggr2": {0: 10.0, 1: 0.0, 2: 0.0},
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="Basic case with non year aggregation: lbs1",
        ),
        pytest.param(
            "lbs2",
            {
                "aggr1": MagicMock(available_stacks=["lbs2"]),
                "aggr2": MagicMock(available_stacks=[]),
            },
            True,
            None,
            pd.DataFrame(
                {
                    "aggr1": {0: 0.0, 1: 0.0, 2: 0.0},
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="Basic case with non year aggregation: lbs2",
        ),
        pytest.param(
            "lbs1",
            {
                "aggr1": MagicMock(available_stacks=["lbs1"]),
                "aggr2": MagicMock(available_stacks=["lbs2"]),
            },
            True,
            pd.Series([2, 4, 6]),
            pd.DataFrame(
                {
                    "aggr1": {2: 10.0, 4: 0.0, 6: 0.0},
                    "aggr2": {2: 10.0, 4: 0.0, 6: 0.0},
                },
                index=pd.Index([2, 4, 6], name="Year"),
            ),
            id="Aggr case: lbs1",
        ),
        pytest.param(
            "lbs2",
            {
                "aggr1": MagicMock(available_stacks=[]),
                "aggr2": MagicMock(available_stacks=["lbs1"]),
            },
            True,
            None,
            pd.DataFrame(index=pd.Index([0, 1, 2], name="Year")),
            id="lbs not in available stacks",
        ),
    ],
)
def test__get_lbs_fraction(
    lbs_name: str,
    aggregated_consumers: dict[str, MagicMock],
    is_year_binding: bool,
    years_binding: pd.Series | None,
    expected_result: pd.DataFrame,
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
) -> None:
    mocked_lbs_parameters_over_year_query._network.aggregated_consumers = (
        aggregated_consumers
    )
    mocked_lbs_parameters_over_year_query._years_binding = years_binding

    result = mocked_lbs_parameters_over_year_query._get_lbs_fraction(
        lbs_name, is_year_binding
    )

    assert_frame_equal(result, expected_result, check_column_type=False)


@pytest.mark.parametrize(
    "lbs_name,  expected_result",
    [
        pytest.param(
            "lbs1",
            pd.DataFrame(
                {
                    "aggr1": {0: 10.0, 1: 0.0, 2: 0.0},
                    "aggr2": {0: 10.0, 1: 0.0, 2: 0.0},
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="lbs1",
        ),
        pytest.param(
            "lbs2",
            pd.DataFrame(index=pd.Index([0, 1, 2], name="Year")),
            id="lbs2",
        ),
        pytest.param(
            ["lbs1", "lbs2"],
            {
                "lbs1": pd.DataFrame(
                    {
                        "aggr1": {0: 10.0, 1: 0.0, 2: 0.0},
                        "aggr2": {0: 10.0, 1: 0.0, 2: 0.0},
                    },
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "lbs2": pd.DataFrame(index=pd.Index([0, 1, 2], name="Year")),
            },
            id="many_lbs_names",
        ),
    ],
)
def test_get_lbs_fraction(
    lbs_name: str | list[str] | None,
    expected_result: pd.DataFrame | dict[str, pd.DataFrame],
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
) -> None:
    result = mocked_lbs_parameters_over_year_query.get_lbs_fraction(lbs_name)
    if isinstance(result, dict):
        for aggr, df in result.items():
            assert_frame_equal(df, expected_result[aggr], check_column_type=False)
    else:
        assert_frame_equal(result, expected_result, check_column_type=False)
