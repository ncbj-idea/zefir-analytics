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

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
    source_parameter_filter_types,
)


@pytest.mark.parametrize(
    "is_hours_resolution, buses, mock_data, expected_output",
    [
        pytest.param(
            True,
            ["bus1", "bus2"],
            {
                "generation_ens": {
                    "bus1": pd.DataFrame(
                        {
                            0: [1, 1, 1],
                            1: [1, 1, 1],
                            2: [1, 1, 1],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                    "bus2": pd.DataFrame(
                        {
                            0: [2, 2, 2],
                            1: [2, 2, 2],
                            2: [2, 2, 2],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                }
            },
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                ],
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
                columns=pd.MultiIndex.from_product(
                    [
                        [
                            "bus1",
                            "bus2",
                        ],
                        [0, 1, 2],
                    ]
                ),
            ),
            id="hourly_resolution_two_buses",
        ),
        pytest.param(
            False,
            ["bus1"],
            {
                "generation_ens": {
                    "bus1": pd.DataFrame(
                        {
                            0: [1, 1, 1],
                            1: [1, 1, 1],
                            2: [1, 1, 1],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                    "bus2": pd.DataFrame(
                        {
                            0: [2, 2, 2],
                            1: [2, 2, 2],
                            2: [2, 2, 2],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                }
            },
            pd.DataFrame(
                [[3, 3, 3]],
                index=["ET1"],
                columns=pd.MultiIndex.from_product([["bus1"], [0, 1, 2]]),
            ),
            id="non_hourly_resolution_single_bus",
        ),
        pytest.param(
            False,
            ["bus4"],
            {
                "generation_ens": {
                    "bus1": pd.DataFrame(
                        {
                            0: [1, 1, 1],
                            1: [1, 1, 1],
                            2: [1, 1, 1],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                    "bus2": pd.DataFrame(
                        {
                            0: [2, 2, 2],
                            1: [2, 2, 2],
                            2: [2, 2, 2],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                }
            },
            pd.DataFrame(),
            id="bus_not_in_generation_ens",
        ),
        pytest.param(
            False,
            ["bus4", "bus5"],
            {
                "generation_ens": {
                    "bus4": pd.DataFrame(
                        {
                            0: [2, 2, 2],
                            1: [2, 2, 2],
                            2: [2, 2, 2],
                        },
                        index=pd.Index([0, 1, 2], name="Hour"),
                    ),
                }
            },
            pd.DataFrame(),
            id="buses_not_in_network",
        ),
        pytest.param(
            True,
            ["bus1", "bus2"],
            {"generation_ens": {}},
            pd.DataFrame(),
            id="empty_generation_ens",
        ),
    ],
)
def test__get_ens(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    is_hours_resolution: bool,
    buses: list[str],
    mock_data: dict[str, dict[str, pd.DataFrame]],
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._bus_results = mock_data
    mocked_source_parameters_over_years_query._network.buses = {
        "bus1": MagicMock(energy_type="ET1"),
        "bus2": MagicMock(energy_type="ET2"),
        "bus3": MagicMock(energy_type="ET3"),
    }

    result = mocked_source_parameters_over_years_query._get_ens(
        is_hours_resolution, buses
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filter_type, filter_names, is_hours_resolution, filtered_bus, ens_df, expected_df",
    [
        pytest.param(
            None,
            None,
            False,
            ["bus1", "bus2", "bus3"],
            pd.DataFrame(
                [
                    [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0],
                ],
                index=["ET1", "ET2", "ET3"],
                columns=pd.MultiIndex.from_product(
                    [["bus1", "bus2", "bus3"], [0, 1, 2]]
                ),
            ),
            pd.DataFrame(
                [
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 0.0, 9.0],
                    [0.0, 0.0, 9.0],
                    [0.0, 0.0, 9.0],
                ],
                index=pd.MultiIndex.from_product(
                    [["bus1", "bus2", "bus3"], range(3)],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2", "ET3"], name="Energy Type"),
            ),
            id="non_hours_resolution_no_filters",
        ),
        pytest.param(
            None,
            ["bus1"],
            True,
            ["bus1"],
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "ET1"),
                        (1, "ET1"),
                        (2, "ET1"),
                    ],
                    names=["Hour", "Energy Type"],
                ),
                columns=pd.MultiIndex.from_product(
                    [
                        [
                            "bus1",
                        ],
                        [0, 1, 2],
                    ]
                ),
            ),
            pd.DataFrame(
                {
                    "ET1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("bus1", 0, 0),
                        ("bus1", 0, 1),
                        ("bus1", 0, 2),
                        ("bus1", 1, 0),
                        ("bus1", 1, 1),
                        ("bus1", 1, 2),
                        ("bus1", 2, 0),
                        ("bus1", 2, 1),
                        ("bus1", 2, 2),
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["ET1"], name="Energy Type"),
            ),
            id="hours_resolution_filter_bus1",
        ),
        pytest.param(
            "aggr",
            None,
            False,
            ["bus1", "bus3"],
            pd.DataFrame(
                [
                    [3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 9.0, 9.0, 9.0],
                ],
                index=["ET1", "ET3"],
                columns=pd.MultiIndex.from_product([["bus1", "bus3"], [0, 1, 2]]),
            ),
            pd.DataFrame(
                [
                    [3.0, 0.0],
                    [3.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 9.0],
                    [0.0, 9.0],
                    [0.0, 9.0],
                ],
                index=pd.MultiIndex.from_product(
                    [["bus1", "bus3"], range(3)],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["ET1", "ET3"], name="Energy Type"),
            ),
            id="non_hours_resolution_filter_aggr",
        ),
        pytest.param(
            "aggr",
            ["bus4"],
            False,
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            id="non_hours_resolution_filter_aggr_filter_non_existing_bus",
        ),
    ],
)
def test_get_ens(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    filter_type: source_parameter_filter_types | None,
    filter_names: None | list[str],
    is_hours_resolution: bool,
    filtered_bus: list[str] | None,
    ens_df: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._network.buses = {
        "bus1": MagicMock(energy_type="ET1"),
        "bus2": MagicMock(energy_type="ET2"),
        "bus3": MagicMock(energy_type="ET3"),
    }
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=([], filtered_bus),
    ), patch.object(
        mocked_source_parameters_over_years_query, "_get_ens", return_value=ens_df
    ):
        result = mocked_source_parameters_over_years_query.get_ens(
            filter_type, filter_names, is_hours_resolution
        )
        assert_frame_equal(result, expected_df)
