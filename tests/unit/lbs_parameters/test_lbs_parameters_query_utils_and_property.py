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

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from zefir_analytics._engine.data_queries.lbs_parameters_over_years import (
    LbsParametersOverYearsQuery,
)


def test_property_line_names_mocked_object(
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
) -> None:
    expected_lines_names = ["lbs1", "lbs2"]
    assert sorted(mocked_lbs_parameters_over_year_query.lbs_names) == sorted(
        expected_lines_names
    )


@pytest.mark.parametrize(
    "results, expected_output",
    [
        pytest.param(
            {
                "fraction": {
                    "aggr1": pd.DataFrame(columns=["lbs1", "lbs2"]),
                }
            },
            ["lbs1", "lbs2"],
            id="one_aggr_2_lbs",
        ),
        pytest.param(
            {
                "fraction": {
                    "aggr1": pd.DataFrame(columns=["lbs1", "lbs2"]),
                    "aggr2": pd.DataFrame(columns=["lbs3"]),
                    "aggr3": pd.DataFrame(columns=["lbs5", "lbs4"]),
                }
            },
            ["lbs1", "lbs2", "lbs3", "lbs4", "lbs5"],
            id="3_aggr_5_lbs",
        ),
        pytest.param(
            {
                "fraction": {
                    "aggr1": pd.DataFrame(columns=["lbs1", "lbs2"]),
                    "aggr2": pd.DataFrame(),
                    "aggr3": pd.DataFrame(columns=["lbs5", "lbs4"]),
                }
            },
            ["lbs1", "lbs2", "lbs4", "lbs5"],
            id="3_aggr_4_lbs_one_empty_df",
        ),
        pytest.param(
            {
                "fraction": {
                    "aggr1": pd.DataFrame(),
                    "aggr2": pd.DataFrame(),
                }
            },
            [],
            id="2_aggr_all_empty_df",
        ),
        pytest.param(
            {"fraction": {}},
            [],
            id="no_results",
        ),
    ],
)
def test_property_lines_names(
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
    results: dict[str, dict[str, pd.DataFrame]],
    expected_output: list[str],
) -> None:
    mocked_lbs_parameters_over_year_query._fractions_results = results
    assert sorted(mocked_lbs_parameters_over_year_query.lbs_names) == sorted(
        expected_output
    )


@pytest.mark.parametrize(
    "lbs_name, lbs_buses, buses_data, bus_generators, bus_storages, expected_generators, expected_storages",
    [
        pytest.param(
            "LBS1",
            ["bus1", "bus2"],
            {"bus1": ["gen1", "gen2"], "bus2": ["gen3"]},
            {"bus1": ["gen1", "gen2"], "bus2": ["gen3"]},
            {"bus1": ["stor1"], "bus2": ["stor2"]},
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            id="Multiple buses and sources",
        ),
        pytest.param(
            "LBS2",
            [],
            {"bus1": []},
            {"bus1": []},
            {"bus1": []},
            [],
            [],
            id="Empty buses",
        ),
        pytest.param(
            "LBS3",
            ["bus1", "bus2"],
            {"bus1": ["gen1"], "bus2": []},
            {"bus1": ["gen1"], "bus2": []},
            {"bus1": ["stor1"], "bus2": []},
            ["gen1"],
            ["stor1"],
            id="Some buses empty",
        ),
    ],
)
def test_get_attached_sources(
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
    lbs_name: str,
    lbs_buses: list[str],
    buses_data: dict[str, list[str]] | dict[str, list[Any]],
    bus_generators: dict[str, list[str]] | dict[str, list[Any]],
    bus_storages: dict[str, list[str]] | dict[str, list[Any]],
    expected_generators: list[str],
    expected_storages: list[str],
) -> None:
    mock_network = MagicMock()
    mock_network.local_balancing_stacks = {
        lbs_name: MagicMock(buses={"et1": lbs_buses})
    }

    mock_network.buses = {
        bus_name: MagicMock(
            generators=bus_generators.get(bus_name, []),
            storages=bus_storages.get(bus_name, []),
        )
        for bus_name in buses_data.keys()
    }

    mocked_lbs_parameters_over_year_query._network = mock_network

    generators, storages = mocked_lbs_parameters_over_year_query._get_attached_sources(
        lbs_name
    )

    assert generators == expected_generators
    assert storages == expected_storages


@pytest.mark.parametrize(
    "lbs_name, aggregated_consumers, df_fraction, expected_result",
    [
        pytest.param(
            "lbs1",
            {
                "aggr1": MagicMock(n_consumers=pd.Series([10, 10, 1])),
            },
            pd.DataFrame(
                {
                    "aggr1": {0: 10.0, 1: 0.0, 2: 0.0},
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            np.array([100, 0, 0]),
            id="Basic case for lbs1",
        ),
        pytest.param(
            "lbs2",
            {
                "aggr2": MagicMock(n_consumers=pd.Series([2, 2, 2])),
            },
            pd.DataFrame(
                {
                    "aggr2": {0: 10.0, 1: 10.0, 2: 5.0},
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            np.array([20, 20, 10]),
            id="Basic case for lbs2",
        ),
        pytest.param(
            "lbs2",
            {
                "aggr2": MagicMock(n_consumers=pd.Series([2, 2, 2])),
            },
            pd.DataFrame(index=pd.Index([0, 1, 2], name="Year")),
            np.array([0, 0, 0]),
            id="empty_case",
        ),
    ],
)
def test_get_fraction_factor(
    mocked_lbs_parameters_over_year_query: LbsParametersOverYearsQuery,
    lbs_name: str,
    aggregated_consumers: dict[str, MagicMock],
    df_fraction: pd.DataFrame,
    expected_result: np.ndarray,
) -> None:
    mocked_lbs_parameters_over_year_query._network.aggregated_consumers = (
        aggregated_consumers
    )
    with patch.object(
        mocked_lbs_parameters_over_year_query,
        "_get_lbs_fraction",
        return_value=df_fraction,
    ):
        result = mocked_lbs_parameters_over_year_query._get_fraction_factor(lbs_name)
        np.testing.assert_array_equal(result, expected_result)
