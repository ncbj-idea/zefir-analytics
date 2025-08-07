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

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.lbs_parameters_over_years import (
    LbsParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "lbs_name, gen_attach, storage_attach, fraction_factor, expected_result",
    [
        pytest.param(
            "lbs1",
            ["gen1", "gen2"],
            ["stor1"],
            np.array([1.0, 1.0, 1.0]),
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "gen2": [10.0, 20.0, 30.0],
                    "stor1": [10.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="lbs_1_gen1_gen2_stor1_attached_no_fraction_factor",
        ),
        pytest.param(
            "lbs1",
            ["gen1", "gen2"],
            ["stor1"],
            np.array([2.0, 2.0, 2.0]),
            pd.DataFrame(
                {
                    "gen1": [0.5, 1.0, 1.5],
                    "gen2": [5.0, 10.0, 15.0],
                    "stor1": [5.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="lbs_1_gen1_gen2_stor1_attached_fraction_const_half_values",
        ),
        pytest.param(
            "lbs1",
            ["gen1", "gen2"],
            ["stor1"],
            np.array([0.5, 0.0, 5.0]),
            pd.DataFrame(
                {
                    "gen1": [2.0, 0.0, 0.6],
                    "gen2": [20.0, 0.0, 6.0],
                    "stor1": [20.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="lbs_1_gen1_gen2_stor1_attached_fraction_mixed_values",
        ),
        pytest.param(
            "lbs1",
            ["gen1"],
            [],
            np.array([0.5, 0.0, 5.0]),
            pd.DataFrame(
                {
                    "gen1": [2.0, 0.0, 0.6],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="lbs_1_gen1_attached_fraction_mixed_values",
        ),
        pytest.param(
            "lbs2",
            [],
            [],
            np.array(
                [
                    0.5,
                    0.0,
                    5.0,
                ]
            ),
            pd.DataFrame(index=pd.Index([0, 1, 2], name="Year")),
            id="lbs_2_no_gens_and_stors",
        ),
    ],
)
def test__get_lbs_capacity(
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
    lbs_name: str,
    gen_attach: list[str],
    storage_attach: list[str],
    fraction_factor: np.ndarray[float],
    expected_result: pd.DataFrame,
) -> None:
    with patch.object(
        mocked_lbs_parameters_over_year_query,
        "_get_attached_sources",
        return_value=(gen_attach, storage_attach),
    ), patch.object(
        mocked_lbs_parameters_over_year_query,
        "_get_fraction_factor",
        return_value=fraction_factor,
    ):
        result = mocked_lbs_parameters_over_year_query._get_lbs_capacity(lbs_name)
        assert_frame_equal(result, expected_result, check_column_type=False)


@pytest.mark.parametrize(
    "lbs_name, expected_result",
    [
        pytest.param(
            "lbs1",
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "gen2": [10.0, 20.0, 30.0],
                    "stor1": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="lbs_1_df",
        ),
        pytest.param(
            ["lbs1", "lbs2"],
            {
                "lbs1": pd.DataFrame(
                    {
                        "gen1": [1.0, 2.0, 3.0],
                        "gen2": [10.0, 20.0, 30.0],
                        "stor1": [0.0, 0.0, 0.0],
                    },
                    index=pd.Index([0, 1, 2], name="Year"),
                ),
                "lbs2": pd.DataFrame(),
            },
            id="list_of_lbs",
        ),
        pytest.param(
            None,
            {
                "lbs1": pd.DataFrame(),
                "lbs2": pd.DataFrame(),
            },
            id="None",
        ),
    ],
)
@patch("zefir_analytics._engine.data_queries.utils.argument_condition")
def test_get_lbs_capacity(
    mocked_argument_condition: MagicMock,
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
    lbs_name: str | list[str],
    expected_result: pd.DataFrame | dict[str, pd.DataFrame],
) -> None:
    mocked_argument_condition.return_value = expected_result
    result = mocked_lbs_parameters_over_year_query.get_lbs_capacity(lbs_name)
    if isinstance(result, dict):
        for aggr, df in result.items():
            assert_frame_equal(df, expected_result[aggr])
    else:
        assert_frame_equal(result, expected_result)
    mocked_argument_condition.assert_called_once_with(
        (
            lbs_name
            if lbs_name is not None
            else mocked_lbs_parameters_over_year_query.lbs_names
        ),
        mocked_lbs_parameters_over_year_query._get_lbs_capacity,
    )
