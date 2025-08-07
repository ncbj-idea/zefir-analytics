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


from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.constants import ENERGY_TYPE_LABEL
from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "bus_set, expected_generators, expected_storages",
    [
        pytest.param({"bus1"}, ["gen1"], ["stor1"], id="bus1"),
        pytest.param({"bus2"}, ["gen2"], [], id="bus2"),
        pytest.param({"bus3"}, ["gen2", "gen3"], ["stor2"], id="bus3"),
        pytest.param(
            {"bus1", "bus3"},
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            id="bus1_bus3",
        ),
        pytest.param({"bus4"}, [], [], id="bus4_not_included_in_network"),
    ],
)
def test_get_generator_and_storage_from_set_of_buses(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    bus_set: set[str],
    expected_generators: list[str],
    expected_storages: list[str],
) -> None:
    gen1 = MagicMock(buses={"bus1"})
    gen1.name = "gen1"

    gen2 = MagicMock(buses={"bus2", "bus3"})
    gen2.name = "gen2"

    gen3 = MagicMock(buses={"bus3"})
    gen3.name = "gen3"

    stor1 = MagicMock(bus="bus1")
    stor1.name = "stor1"

    stor2 = MagicMock(bus="bus3")
    stor2.name = "stor2"

    with patch.object(
        mocked_source_parameters_over_years_query._network,
        "generators",
        {
            "gen1": gen1,
            "gen2": gen2,
            "gen3": gen3,
        },
    ), patch.object(
        mocked_source_parameters_over_years_query._network,
        "storages",
        {
            "stor1": stor1,
            "stor2": stor2,
        },
    ):
        result_generators, result_storages = (
            mocked_source_parameters_over_years_query._get_generator_and_storage_from_set_of_buses(
                bus_set
            )
        )

    assert sorted(result_generators) == sorted(expected_generators)
    assert sorted(result_storages) == sorted(expected_storages)


def test_get_global_generators_and_storage(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
) -> None:
    gen1 = MagicMock(buses={"bus1"})
    gen1.name = "gen1"

    gen2 = MagicMock(buses={"bus2", "bus3"})
    gen2.name = "gen2"

    gen3 = MagicMock(buses={"bus3"})
    gen3.name = "gen3"

    stor1 = MagicMock(bus="bus1")
    stor1.name = "stor1"

    stor2 = MagicMock(bus="bus3")
    stor2.name = "stor2"

    mocked_source_parameters_over_years_query._network.generators.values.return_value = [
        gen1,
        gen2,
        gen3,
    ]
    mocked_source_parameters_over_years_query._network.storages.values.return_value = [
        stor1,
        stor2,
    ]
    mocked_source_parameters_over_years_query._network.buses = {"bus1", "bus2", "bus3"}

    stack1 = MagicMock()
    stack1.buses_out.values.return_value = {"bus1"}

    stack2 = MagicMock()
    stack2.buses_out.values.return_value = {"bus2"}

    mocked_source_parameters_over_years_query._network.local_balancing_stacks.values.return_value = [
        stack1,
        stack2,
    ]

    result_generators, result_storages = (
        mocked_source_parameters_over_years_query._get_global_generators_and_storage()
    )

    assert sorted(result_generators) == ["gen2", "gen3"]
    assert sorted(result_storages) == ["stor2"]


@pytest.mark.parametrize(
    "generators, results_group, is_hours_resolution, expected_df",
    [
        pytest.param(
            ["gen1"],
            "group1",
            False,
            pd.DataFrame(
                [[0, 6, 60], [0, 60, 600]],
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
                columns=pd.MultiIndex.from_product([["gen1"], ["0", "1", "2"]]),
            ),
            id="group1_gen1_not_hours",
        ),
        pytest.param(
            ["gen1"],
            "group1",
            True,
            pd.DataFrame(
                [
                    [0, 1, 10],
                    [0, 10, 100],
                    [0, 2, 20],
                    [0, 20, 200],
                    [0, 3, 30],
                    [0, 30, 300],
                ],
                index=pd.MultiIndex.from_product(
                    [[0, 1, 2], ["ET1", "ET2"]], names=["Hour", "Energy Type"]
                ),
                columns=pd.MultiIndex.from_product([["gen1"], ["0", "1", "2"]]),
            ),
            id="group1_gen1_hours",
        ),
        pytest.param(
            ["gen1", "gen2"],
            "group1",
            True,
            pd.DataFrame(
                [
                    [0, 1, 10, 0, 0, 0],
                    [0, 10, 100, 0, 0, 1],
                    [0, 2, 20, 0, 0, 0],
                    [0, 20, 200, 0, 0, 2],
                    [0, 3, 30, 0, 0, 0],
                    [0, 30, 300, 0, 0, 3],
                ],
                index=pd.MultiIndex.from_product(
                    [[0, 1, 2], ["ET1", "ET2"]], names=["Hour", "Energy Type"]
                ),
                columns=pd.MultiIndex.from_product([["gen1", "gen2"], ["0", "1", "2"]]),
            ),
            id="group1_gen1_gen2_hours",
        ),
        pytest.param(
            ["gen3"],
            "group2",
            False,
            pd.DataFrame(
                [[0, 6, 60], [0, 60, 600]],
                index=pd.Index(["ET1", "ET2"], name="Energy Type"),
                columns=pd.MultiIndex.from_product([["gen3"], ["0", "1", "2"]]),
            ),
            id="group2_gen3_not_hours",
        ),
        pytest.param(
            [],
            "group2",
            False,
            pd.DataFrame(),
            id="group2_no_generators",
        ),
    ],
)
def test_get_generator_results(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generators: list[str],
    results_group: str,
    is_hours_resolution: bool,
    expected_df: pd.DataFrame,
) -> None:
    generator_results = {
        "group1": {
            "gen1": pd.DataFrame(
                {
                    ENERGY_TYPE_LABEL: ["ET1", "ET1", "ET1", "ET2", "ET2", "ET2"],
                    "0": [0, 0, 0, 0, 0, 0],
                    "1": [1, 2, 3, 10, 20, 30],
                    "2": [10, 20, 30, 100, 200, 300],
                },
                index=pd.Index([0, 1, 2, 0, 1, 2], name="Hour"),
            ),
            "gen2": pd.DataFrame(
                {
                    ENERGY_TYPE_LABEL: ["ET1", "ET1", "ET1", "ET2", "ET2", "ET2"],
                    "0": [0, 0, 0, 0, 0, 0],
                    "1": [0, 0, 0, 0, 0, 0],
                    "2": [0, 0, 0, 1, 2, 3],
                },
                index=pd.Index([0, 1, 2, 0, 1, 2], name="Hour"),
            ),
        },
        "group2": {
            "gen3": pd.DataFrame(
                {
                    ENERGY_TYPE_LABEL: ["ET1", "ET1", "ET1", "ET2", "ET2", "ET2"],
                    "0": [0, 0, 0, 0, 0, 0],
                    "1": [1, 2, 3, 10, 20, 30],
                    "2": [10, 20, 30, 100, 200, 300],
                },
                index=pd.Index([0, 1, 2, 0, 1, 2], name="Hour"),
            ),
        },
    }

    with patch.object(
        mocked_source_parameters_over_years_query,
        "_generator_results",
        generator_results,
    ):
        result = mocked_source_parameters_over_years_query._get_generator_et_results(
            results_group, generators, is_hours_resolution
        )
        assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "storages, results_group, is_hours_resolution, expected_df",
    [
        pytest.param(
            ["stor1"],
            "group1",
            False,
            pd.DataFrame(
                [[0, 6, 60]],
                index=pd.Index(["ET1"]),
                columns=pd.MultiIndex.from_product([["stor1"], ["0", "1", "2"]]),
            ),
            id="group1_stor1_not_hours",
        ),
        pytest.param(
            ["stor1", "stor2"],
            "group1",
            False,
            pd.DataFrame(
                {
                    ("stor1", "0"): [0.0, 0.0],
                    ("stor1", "1"): [6.0, 0.0],
                    ("stor1", "2"): [60.0, 0.0],
                    ("stor2", "0"): [0.0, 0.0],
                    ("stor2", "1"): [0.0, 0.0],
                    ("stor2", "2"): [0.0, 6.0],
                },
                index=pd.Index(["ET1", "ET2"]),
                columns=pd.MultiIndex.from_product(
                    [["stor1", "stor2"], ["0", "1", "2"]]
                ),
            ),
            id="group1_stor1_stor2_not_hours",
        ),
        pytest.param(
            ["stor3", "stor4"],
            "group2",
            True,
            pd.DataFrame(
                {
                    ("stor3", "0"): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ("stor3", "1"): [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
                    ("stor3", "2"): [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
                    ("stor4", "0"): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ("stor4", "1"): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ("stor4", "2"): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
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
            ),
            id="group2_stor3_stor4_hours",
        ),
        pytest.param(
            [],
            "group2",
            True,
            pd.DataFrame(),
            id="group2_no_storages",
        ),
    ],
)
def test_get_storages_results(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    storages: list[str],
    results_group: str,
    is_hours_resolution: bool,
    expected_df: pd.DataFrame,
) -> None:
    stor1 = MagicMock(energy_source_type="Storage_ET1")
    stor2 = MagicMock(energy_source_type="Storage_ET2")
    stor3 = MagicMock(energy_source_type="Storage_ET1")
    stor4 = MagicMock(energy_source_type="Storage_ET2")

    stor_type_1 = MagicMock(energy_type="ET1")
    stor_type_2 = MagicMock(energy_type="ET2")

    mocked_source_parameters_over_years_query._network.storages = {
        "stor1": stor1,
        "stor2": stor2,
        "stor3": stor3,
        "stor4": stor4,
    }

    mocked_source_parameters_over_years_query._network.storage_types = {
        "Storage_ET1": stor_type_1,
        "Storage_ET2": stor_type_2,
    }

    storage_results = {
        "group1": {
            "stor1": pd.DataFrame(
                {
                    "0": [0, 0, 0],
                    "1": [1, 2, 3],
                    "2": [10, 20, 30],
                },
                index=pd.Index([0, 1, 2], name="Hour"),
            ),
            "stor2": pd.DataFrame(
                {
                    "0": [0, 0, 0],
                    "1": [0, 0, 0],
                    "2": [1, 2, 3],
                },
                index=pd.Index([0, 1, 2], name="Hour"),
            ),
        },
        "group2": {
            "stor3": pd.DataFrame(
                {
                    "0": [0, 0, 0],
                    "1": [10, 20, 30],
                    "2": [100, 200, 300],
                },
                index=pd.Index(
                    [
                        0,
                        1,
                        2,
                    ],
                    name="Hour",
                ),
            ),
            "stor4": pd.DataFrame(
                {
                    "0": [0, 0, 0],
                    "1": [0, 0, 0],
                    "2": [0, 0, 0],
                },
                index=pd.Index([0, 1, 2], name="Hour"),
            ),
        },
    }
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_storage_results",
        storage_results,
    ):
        result = mocked_source_parameters_over_years_query._get_storage_results(
            results_group, storages, is_hours_resolution
        )
        assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "generators, storages, expected_mapping",
    [
        pytest.param(
            [
                {"energy_source_type": "ET1", "name": "gen1"},
                {"energy_source_type": "ET2", "name": "gen2"},
                {"energy_source_type": "ET1", "name": "gen3"},
            ],
            [
                {"energy_source_type": "ET3", "name": "stor1"},
                {"energy_source_type": "ET3", "name": "stor2"},
            ],
            {"ET1": {"gen1", "gen3"}, "ET2": {"gen2"}, "ET3": {"stor1", "stor2"}},
            id="3_gens_2_storages_separated_et",
        ),
        pytest.param([], [], {}, id="empty"),
        pytest.param(
            [{"energy_source_type": "ET1", "name": "gen4"}],
            [
                {"energy_source_type": "ET2", "name": "stor3"},
                {"energy_source_type": "ET1", "name": "stor4"},
            ],
            {"ET1": {"gen4", "stor4"}, "ET2": {"stor3"}},
            id="1_gens_2_storages_same_et",
        ),
    ],
)
def test_create_energy_source_type_mapping(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generators: list[dict[str, str]],
    storages: list[dict[str, str]],
    expected_mapping: dict[str, set[str]],
) -> None:
    mocked_generators = []
    for gen in generators:
        mock_gen = MagicMock(energy_source_type=gen["energy_source_type"])
        mock_gen.name = gen["name"]
        mocked_generators.append(mock_gen)

    mocked_storages = []
    for stor in storages:
        mock_stor = MagicMock(energy_source_type=stor["energy_source_type"])
        mock_stor.name = stor["name"]
        mocked_storages.append(mock_stor)

    mocked_source_parameters_over_years_query._network.generators.values.return_value = (
        mocked_generators
    )
    mocked_source_parameters_over_years_query._network.storages.values.return_value = (
        mocked_storages
    )

    result = (
        mocked_source_parameters_over_years_query._create_energy_source_type_mapping()
    )
    assert result == expected_mapping


@pytest.mark.parametrize(
    "target_list, expected_list",
    [
        pytest.param([[1, 2], [2, 4]], [1, 2, 2, 4], id="inner_list_of_int"),
        pytest.param(
            [[("a", "b")], [("c", "d")]],
            [("a", "b"), ("c", "d")],
            id="inner_list_of_tuple",
        ),
        pytest.param([[[1, 2]], [[2, 4]]], [[1, 2], [2, 4]], id="inner_list_of_list"),
        pytest.param([], [], id="empty_case"),
        pytest.param(["hello"], ["h", "e", "l", "l", "o"], id="string_in_outer_list"),
    ],
)
def test_flatten_2d_list(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    target_list: list[list[Any]],
    expected_list: list[Any],
) -> None:
    result = mocked_source_parameters_over_years_query._flatten_2d_list(target_list)
    assert result == expected_list


@pytest.mark.parametrize(
    "filter_type, filter_names, is_bus_filter, expected_output",
    [
        pytest.param(
            None, None, False, (["gen1", "gen2"], ["stor1", "stor2"]), id="No_filters"
        ),
        pytest.param(
            None,
            None,
            True,
            (["gen1", "gen2"], ["stor1", "stor2"], ["bus1", "bus2"]),
            id="only_bus_filter",
        ),
        pytest.param(
            "aggr", ["aggr1"], False, (["gen1", "gen2"], ["stor1"]), id="aggr_filter"
        ),
        pytest.param(
            "stack",
            ["stack1"],
            True,
            (["gen1"], ["stor1"], ["bus1"]),
            id="stack_filter",
        ),
    ],
)
def test_filter_elements(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    filter_type: None | Literal["aggr", "stack"],
    filter_names: None | list[str],
    is_bus_filter: bool,
    expected_output: tuple[list[str]],
) -> None:
    gen1 = MagicMock(buses={"bus1"})
    gen1.name = "gen1"

    stor1 = MagicMock(bus="bus1")
    stor1.name = "stor1"

    gen2 = MagicMock(buses={"bus2"})
    gen2.name = "gen2"

    mocked_source_parameters_over_years_query._network.generators.values.return_value = [
        gen1,
        gen2,
    ]
    mocked_source_parameters_over_years_query._network.storages.values.return_value = [
        stor1,
    ]
    mocked_source_parameters_over_years_query._network.generators.keys.return_value = [
        "gen1",
        "gen2",
    ]
    mocked_source_parameters_over_years_query._network.storages.keys.return_value = [
        "stor1",
        "stor2",
    ]
    mocked_source_parameters_over_years_query._network.buses.keys.return_value = [
        "bus1",
        "bus2",
    ]

    aggr_mock = MagicMock(stack_base_fraction={"stack1": 0.5, "stack2": 0.1})
    aggr_mock.name = "aggr1"
    mocked_source_parameters_over_years_query._network.aggregated_consumers.values.return_value = [
        aggr_mock
    ]

    stack_mock_1 = MagicMock(buses_out={"bus1": "bus1"})
    stack_mock_1.name = "stack1"
    stack_mock_2 = MagicMock(buses_out={"bus2": "bus2"})
    stack_mock_2.name = "stack2"
    mocked_source_parameters_over_years_query._network.local_balancing_stacks.values.return_value = [
        stack_mock_1,
        stack_mock_2,
    ]

    result = mocked_source_parameters_over_years_query._filter_elements(
        filter_type, filter_names, is_bus_filter
    )
    assert result == expected_output


@pytest.mark.parametrize(
    "df, mapping, expected_output",
    [
        pytest.param(
            pd.DataFrame(
                {
                    ("gen1", "A"): [1, 2, 3],
                    ("gen2", "A"): [4, 5, 6],
                    ("gen3", "B"): [7, 8, 9],
                },
                columns=pd.MultiIndex.from_tuples(
                    [("gen1", "A"), ("gen2", "A"), ("gen3", "B")]
                ),
            ),
            {"type1": {"gen1", "gen2"}, "type2": {"gen3"}},
            pd.DataFrame(
                {("type1", "A"): [5, 7, 9], ("type2", "B"): [7, 8, 9]},
                columns=pd.MultiIndex.from_tuples([("type1", "A"), ("type2", "B")]),
            ),
            id="multiindex_columns",
        ),
        pytest.param(
            pd.DataFrame({"gen1": [1, 2, 3], "gen2": [4, 5, 6], "gen3": [7, 8, 9]}),
            {"type1": {"gen1", "gen2"}, "type2": {"gen3"}},
            pd.DataFrame({"type1": [5, 7, 9], "type2": [7, 8, 9]}),
            id="single_level_columns",
        ),
        pytest.param(
            pd.DataFrame({"gen1": [1, 2, 3], "gen2": [4, 5, 6], "gen3": [7, 8, 9]}),
            {"type1": {"gen4"}},
            pd.DataFrame(),
            id="no_matching_columns",
        ),
    ],
)
def test_aggregate_by_type(
    df: pd.DataFrame, mapping: dict[str, set[str]], expected_output: pd.DataFrame
) -> None:
    result = SourceParametersOverYearsQuery._aggregate_by_type(df, mapping)
    pd.testing.assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "energy_source_df, index_name, level, column_name, filter_type, filter_names, "
    "is_hours_resolution, is_binding_skip, year_aggregation, expected_output",
    [
        pytest.param(
            pd.DataFrame(
                np.array(
                    [[10, 20, 30, 0, 0, 0, 5, 15, 25], [0, 0, 0, 40, 50, 60, 0, 0, 0]]
                ),
                index=["TYPE_1", "TYPE_2"],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen_1", "0"),
                        ("gen_1", "1"),
                        ("gen_1", "2"),
                        ("gen_2", "0"),
                        ("gen_2", "1"),
                        ("gen_2", "2"),
                        ("gen_3", "0"),
                        ("gen_3", "1"),
                        ("gen_3", "2"),
                    ],
                ),
            ),
            "Element Type",
            "type",
            "Year",
            None,
            None,
            False,
            False,
            False,
            pd.DataFrame(
                [[10, 40], [20, 50], [30, 60], [5, 0], [15, 0], [25, 0]],
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
                columns=pd.Index(["TYPE_1", "TYPE_2"], name="Element Type"),
            ),
            id="no_filters",
        ),
        pytest.param(
            pd.DataFrame(
                np.array(
                    [[10, 20, 30, 0, 0, 0, 5, 15, 25], [0, 0, 0, 40, 50, 60, 0, 0, 0]]
                ),
                index=["TYPE_1", "TYPE_2"],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen_1", "0"),
                        ("gen_1", "1"),
                        ("gen_1", "2"),
                        ("gen_2", "0"),
                        ("gen_2", "1"),
                        ("gen_2", "2"),
                        ("gen_3", "0"),
                        ("gen_3", "1"),
                        ("gen_3", "2"),
                    ],
                ),
            ),
            "Element Type",
            "type",
            "Year",
            None,
            ["group1"],
            False,
            False,
            False,
            pd.DataFrame(
                [[10, 40], [20, 50], [30, 60]],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 0),
                        ("group1", 1),
                        ("group1", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["TYPE_1", "TYPE_2"], name="Element Type"),
            ),
            id="aggregate_type",
        ),
        pytest.param(
            pd.DataFrame(
                np.array(
                    [
                        [10, 20, 30, 40, 50, 5, 15, 25, 35],
                        [15, 25, 35, 45, 55, 10, 20, 30, 40],
                        [20, 30, 40, 50, 60, 15, 25, 35, 45],
                        [25, 35, 45, 55, 65, 20, 30, 40, 50],
                        [30, 40, 50, 60, 70, 25, 35, 45, 55],
                        [0, 0, 0, 10, 20, 0, 0, 0, 10],
                        [0, 0, 0, 15, 25, 0, 0, 0, 15],
                        [0, 0, 0, 20, 30, 0, 0, 0, 20],
                        [0, 0, 0, 25, 35, 0, 0, 0, 25],
                        [0, 0, 0, 30, 40, 0, 0, 0, 30],
                    ]
                ),
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "TYPE_1"),
                        (1, "TYPE_1"),
                        (2, "TYPE_1"),
                        (3, "TYPE_1"),
                        (4, "TYPE_1"),
                        (0, "TYPE_2"),
                        (1, "TYPE_2"),
                        (2, "TYPE_2"),
                        (3, "TYPE_2"),
                        (4, "TYPE_2"),
                    ],
                    names=["Hour", "Energy Type"],
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen_1", "0"),
                        ("gen_1", "1"),
                        ("gen_1", "2"),
                        ("gen_2", "0"),
                        ("gen_2", "1"),
                        ("gen_2", "2"),
                        ("gen_3", "0"),
                        ("gen_3", "1"),
                        ("gen_3", "2"),
                    ],
                    names=["Element Type", "Hour"],
                ),
            ),
            "Element Type",
            "type",
            "Year",
            None,
            None,
            True,
            False,
            False,
            pd.DataFrame(
                np.array(
                    [
                        [50, 10],
                        [60, 15],
                        [70, 20],
                        [80, 25],
                        [90, 30],
                        [70, 20],
                        [80, 25],
                        [90, 30],
                        [100, 35],
                        [110, 40],
                        [35, 0],
                        [45, 0],
                        [55, 0],
                        [65, 0],
                        [75, 0],
                        [15, 0],
                        [20, 0],
                        [25, 0],
                        [30, 0],
                        [35, 0],
                        [25, 0],
                        [30, 0],
                        [35, 0],
                        [40, 0],
                        [45, 0],
                        [35, 10],
                        [40, 15],
                        [45, 20],
                        [50, 25],
                        [55, 30],
                    ]
                ),
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 0, 0),
                        ("group1", 0, 1),
                        ("group1", 0, 2),
                        ("group1", 0, 3),
                        ("group1", 0, 4),
                        ("group1", 1, 0),
                        ("group1", 1, 1),
                        ("group1", 1, 2),
                        ("group1", 1, 3),
                        ("group1", 1, 4),
                        ("group1", 2, 0),
                        ("group1", 2, 1),
                        ("group1", 2, 2),
                        ("group1", 2, 3),
                        ("group1", 2, 4),
                        ("group2", 0, 0),
                        ("group2", 0, 1),
                        ("group2", 0, 2),
                        ("group2", 0, 3),
                        ("group2", 0, 4),
                        ("group2", 1, 0),
                        ("group2", 1, 1),
                        ("group2", 1, 2),
                        ("group2", 1, 3),
                        ("group2", 1, 4),
                        ("group2", 2, 0),
                        ("group2", 2, 1),
                        ("group2", 2, 2),
                        ("group2", 2, 3),
                        ("group2", 2, 4),
                    ],
                    names=["Network element type", "Year", "Hour"],
                ),
                columns=pd.Index(["TYPE_1", "TYPE_2"], name="Element Type"),
            ),
            id="hour_resolution",
        ),
        pytest.param(
            pd.DataFrame(
                np.array(
                    [[10, 20, 30, 0, 0, 0, 5, 15, 25], [0, 0, 0, 40, 50, 60, 0, 0, 0]]
                ),
                index=["TYPE_1", "TYPE_2"],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen_1", "0"),
                        ("gen_1", "1"),
                        ("gen_1", "2"),
                        ("gen_2", "0"),
                        ("gen_2", "1"),
                        ("gen_2", "2"),
                        ("gen_3", "0"),
                        ("gen_3", "1"),
                        ("gen_3", "2"),
                    ],
                ),
            ),
            "Element Type",
            "type",
            "Year",
            None,
            None,
            False,
            False,
            True,
            pd.DataFrame(
                [[10, 40], [20, 50], [30, 60], [5, 0], [15, 0], [25, 0]],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group1", 2),
                        ("group1", 4),
                        ("group1", 6),
                        ("group2", 2),
                        ("group2", 4),
                        ("group2", 6),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["TYPE_1", "TYPE_2"], name="Element Type"),
            ),
            id="year_binding",
        ),
        pytest.param(
            pd.DataFrame(
                np.array(
                    [[10, 20, 30, 0, 0, 0, 5, 15, 25], [0, 0, 0, 40, 50, 60, 0, 0, 0]]
                ),
                index=["TYPE_1", "TYPE_2"],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("gen_1", "0"),
                        ("gen_1", "1"),
                        ("gen_1", "2"),
                        ("gen_2", "0"),
                        ("gen_2", "1"),
                        ("gen_2", "2"),
                        ("gen_3", "0"),
                        ("gen_3", "1"),
                        ("gen_3", "2"),
                    ],
                ),
            ),
            "Element Type",
            "type",
            "Year",
            None,
            None,
            False,
            True,
            True,
            pd.DataFrame(
                [[10, 40], [20, 50], [30, 60], [5, 0], [15, 0], [25, 0]],
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
                columns=pd.Index(["TYPE_1", "TYPE_2"], name="Element Type"),
            ),
            id="year_binding_skip_binding",
        ),
    ],
)
def test_aggregate_energy_sources(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    energy_source_df: pd.DataFrame,
    index_name: str,
    level: Literal["element", "type"],
    column_name: str | None,
    filter_type: Literal["bus", "stack", "aggr"] | None,
    filter_names: None | list[str],
    is_hours_resolution: bool,
    is_binding_skip: bool,
    year_aggregation: bool,
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen_1", "gen_2"},
        "group2": {"gen_3"},
    }
    if year_aggregation:
        mocked_source_parameters_over_years_query._years_binding = pd.Series(
            [2, 4, 6], index=[0, 1, 2]
        )
    result = mocked_source_parameters_over_years_query._aggregate_energy_sources(
        energy_source_df,
        index_name,
        level,
        column_name,
        filter_type,
        filter_names,
        is_hours_resolution,
        is_binding_skip,
    )
    assert_frame_equal(result, expected_output)
