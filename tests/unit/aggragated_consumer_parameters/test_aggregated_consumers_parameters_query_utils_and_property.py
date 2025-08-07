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

from zefir_analytics._engine.data_queries.aggregated_consumer_parameters_over_years import (
    AggregatedConsumerParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "buses, expected_result",
    [
        pytest.param(
            set(),
            set(),
            id="empty_set",
        ),
        pytest.param(
            {"bus1"},
            {
                "generator_type_1",
                "storage_type_1",
                "generator_type_2",
                "storage_type_2",
            },
            id="only_bus1",
        ),
        pytest.param(
            {"bus2"},
            {"generator_type_3", "storage_type_1"},
            id="only_bus2",
        ),
        pytest.param(
            {"bus1", "bus2"},
            {
                "storage_type_1",
                "generator_type_3",
                "generator_type_1",
                "generator_type_2",
                "storage_type_2",
            },
            id="all_buses",
        ),
    ],
)
def test_get_unique_element_type_from_buses(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    buses: set[str],
    expected_result: set[str],
) -> None:
    result = mocked_aggr_parameters_over_year_query._get_unique_element_type_from_buses(
        buses
    )
    assert result == expected_result


@pytest.mark.parametrize(
    "aggregate_name, expected_result",
    [
        pytest.param(
            "aggr1",
            pd.DataFrame(
                [
                    [1, 1, 1, 1],
                ],
                index=pd.MultiIndex.from_product(
                    [["aggr1"], ["lkt1"]], names=["agg_name", "lbs_name"]
                ),
                columns=pd.Index(
                    [
                        "generator_type_1",
                        "generator_type_2",
                        "storage_type_1",
                        "storage_type_2",
                    ],
                    name="attached_tech",
                ),
            ),
            id="aggr_1",
        ),
        pytest.param(
            "aggr2",
            pd.DataFrame(
                [
                    [1, 1],
                ],
                index=pd.MultiIndex.from_product(
                    [["aggr2"], ["lkt2"]], names=["agg_name", "lbs_name"]
                ),
                columns=pd.Index(
                    [
                        "generator_type_3",
                        "storage_type_1",
                    ],
                    name="attached_tech",
                ),
            ),
            id="aggr_2",
        ),
        pytest.param(
            "aggr3",
            pd.DataFrame(),
            id="aggr_not_in_network",
        ),
    ],
)
def test_get_single_aggregate_elements_type_attachments_dataframe(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    aggregate_name: str,
    expected_result: pd.DataFrame,
) -> None:
    result = mocked_aggr_parameters_over_year_query._get_single_aggregate_elements_type_attachments_dataframe(
        aggregate_name
    )
    assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    "aggregate_name, expected_result",
    [
        pytest.param(
            "aggr1",
            pd.DataFrame(
                [
                    [1, 1, 1, 1],
                ],
                index=pd.MultiIndex.from_product(
                    [["aggr1"], ["lkt1"]], names=["agg_name", "lbs_name"]
                ),
                columns=pd.Index(
                    [
                        "generator_type_1",
                        "generator_type_2",
                        "storage_type_1",
                        "storage_type_2",
                    ],
                    name="attached_tech",
                ),
            ),
            id="aggr_1_str",
        ),
        pytest.param(
            ["aggr1", "aggr2"],
            pd.DataFrame(
                [
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 1],
                ],
                index=pd.MultiIndex.from_tuples(
                    [("aggr1", "lkt1"), ("aggr2", "lkt2")],
                    names=["agg_name", "lbs_name"],
                ),
                columns=pd.Index(
                    [
                        "generator_type_1",
                        "generator_type_2",
                        "storage_type_1",
                        "storage_type_2",
                        "generator_type_3",
                    ],
                    name="attached_tech",
                ),
            ),
            id="list_of_aggr",
        ),
        pytest.param(
            None,
            pd.DataFrame(
                [
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 1],
                ],
                index=pd.MultiIndex.from_tuples(
                    [("aggr1", "lkt1"), ("aggr2", "lkt2")],
                    names=["agg_name", "lbs_name"],
                ),
                columns=pd.Index(
                    [
                        "generator_type_1",
                        "generator_type_2",
                        "storage_type_1",
                        "storage_type_2",
                        "generator_type_3",
                    ],
                    name="attached_tech",
                ),
            ),
            id="None",
        ),
        pytest.param(
            ["aggr3", "no_element"],
            pd.DataFrame(),
            id="list_of_not_network_aggr",
        ),
        pytest.param(
            "no_element",
            pd.DataFrame(),
            id="aggr_not_in_network",
        ),
    ],
)
def test_get_aggregate_elements_type_attachments(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    aggregate_name: str | list[str] | None,
    expected_result: pd.DataFrame,
) -> None:
    result = (
        mocked_aggr_parameters_over_year_query.get_aggregate_elements_type_attachments(
            aggregate_name
        )
    )
    assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    "amount_of_consumers, area, name",
    [
        pytest.param(
            [10, 10, 10],
            10,
            "aggr_1",
            id="Simple_aggregate_consumer",
        ),
        pytest.param(
            [100],
            2,
            "aggr_1",
            id="just_one_year",
        ),
        pytest.param(
            [100, 200, 300, 400],
            None,
            "aggr_1",
            id="None of area",
        ),
        pytest.param(
            [100, 200, 300, 400],
            164.23,
            "aggr_1",
            id="area float value",
        ),
        pytest.param(
            [],
            2,
            "aggr_1",
            id="Empty years",
        ),
    ],
)
def test_create_aggregate_parameters_dataframe(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    amount_of_consumers: list[int],
    area: int | float | None,
    name: str,
) -> None:
    aggregate = MagicMock(n_consumers=pd.Series(amount_of_consumers), average_area=area)
    aggregate.name = name
    result = (
        mocked_aggr_parameters_over_year_query._create_aggregate_parameters_dataframe(
            aggregate
        )
    )
    index = pd.MultiIndex.from_tuples(
        [(name, year) for year in range(len(amount_of_consumers))],
        names=["Aggregate Name", "Year"],
    )
    index = index.set_levels(
        [index.levels[0].astype(str), index.levels[1].astype(int)],
        level=["Aggregate Name", "Year"],
    )
    if area is not None:
        expected_df = pd.DataFrame(
            {
                "n_consumers": amount_of_consumers,
                "total_usable_area": [amount * area for amount in amount_of_consumers],
            },
            index=index,
        )
    else:
        expected_df = pd.DataFrame(
            {
                "n_consumers": amount_of_consumers,
                "total_usable_area": amount_of_consumers,
            },
            index=index,
        )

    assert_frame_equal(result, expected_df, check_dtype=False)


@pytest.mark.parametrize(
    "aggregate_name, expected_result",
    [
        pytest.param(
            "aggr1",
            pd.DataFrame(
                {"n_consumers": [10, 10, 10], "total_usable_area": [20, 20, 20]},
                index=pd.MultiIndex.from_product(
                    [["aggr1"], [0, 1, 2]], names=["Aggregate Name", "Year"]
                ),
            ),
            id="aggr_1_str",
        ),
        pytest.param(
            ["aggr1", "aggr2"],
            pd.DataFrame(
                {
                    "n_consumers": [10, 10, 10, 100, 100, 100],
                    "total_usable_area": [20, 20, 20, 200, 200, 200],
                },
                index=pd.MultiIndex.from_product(
                    [["aggr1", "aggr2"], [0, 1, 2]], names=["Aggregate Name", "Year"]
                ),
            ),
            id="list_of_aggr",
        ),
        pytest.param(
            None,
            pd.DataFrame(
                {
                    "n_consumers": [10, 10, 10, 100, 100, 100],
                    "total_usable_area": [20, 20, 20, 200, 200, 200],
                },
                index=pd.MultiIndex.from_product(
                    [["aggr1", "aggr2"], [0, 1, 2]], names=["Aggregate Name", "Year"]
                ),
            ),
            id="None",
        ),
        pytest.param(
            ["aggr1", "aggr2213"],
            pd.DataFrame(
                {
                    "n_consumers": [
                        10,
                        10,
                        10,
                    ],
                    "total_usable_area": [20, 20, 20],
                },
                index=pd.MultiIndex.from_product(
                    [["aggr1"], [0, 1, 2]], names=["Aggregate Name", "Year"]
                ),
            ),
            id="list_of_aggr_with_one_incorrect",
        ),
        pytest.param(
            ["rwe", "aggr_not_in_network"],
            pd.DataFrame(),
            id="list_of_aggr_with_all_incorrect",
        ),
        pytest.param(
            "aggr3",
            pd.DataFrame(),
            id="str_aggr_not_in_network",
        ),
    ],
)
def test_get_aggregate_parameters(
    mocked_aggr_parameters_over_year_query: AggregatedConsumerParametersOverYearsQuery,
    aggregate_name: str | list[str] | None,
    expected_result: pd.DataFrame,
) -> None:
    result = mocked_aggr_parameters_over_year_query.get_aggregate_parameters(
        aggregate_name
    )
    assert_frame_equal(result, expected_result)
