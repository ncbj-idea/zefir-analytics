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

from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.mark.parametrize(
    "conversion_rates, generation_df, is_hours_resolution, hour_sample, expected_output",
    [
        pytest.param(
            {
                "ET1": pd.Series([1.0, 1.0, 0.0], index=[0, 1, 2]),
                "ET2": pd.Series([0.5, 0.5, 0.5], index=[0, 1, 2]),
            },
            pd.DataFrame(
                {
                    0: [10.0, 10.0, 10.0, 10.0],
                    1: [10.0, 10.0, 10.0, 10.0],
                    2: [10.0, 10.0, 10.0, 10.0],
                },
                index=pd.Index([0, 1, 2, 3], name="Hour"),
            ),
            True,
            np.array([0, 1]),
            pd.DataFrame(
                data=[
                    [
                        10.0,
                        10.0,
                        10.0,
                    ],
                    [
                        10.0,
                        10.0,
                        10.0,
                    ],
                    [20.0, 20.0, 20.0],
                    [20.0, 20.0, 20.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [(0, "ET1"), (1, "ET1"), (0, "ET2"), (1, "ET2")],
                    names=["Hour", "Energy Type"],
                ),
                columns=[0, 1, 2],
            ),
            id="hour_resolution",
        ),
        pytest.param(
            {
                "ET1": pd.Series([1.0, 1.0, 0.0], index=[0, 1, 2]),
                "ET2": pd.Series([0.5, 0.5, 0.5], index=[0, 1, 2]),
            },
            pd.DataFrame(
                {
                    0: [10.0, 10.0, 10.0, 10.0],
                    1: [10.0, 10.0, 10.0, 10.0],
                    2: [10.0, 10.0, 10.0, 10.0],
                },
                index=pd.Index([0, 1, 2, 3], name="Hour"),
            ),
            False,
            np.array([0, 1]),
            pd.DataFrame(
                data=[[20.0, 40.0], [20.0, 40.0], [20.0, 40.0]],
                columns=["ET1", "ET2"],
                index=[0, 1, 2],
            ),
            id="no_hour_resolution",
        ),
    ],
)
def test_calculate_demand_for_generator(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    conversion_rates: dict[str, pd.Series],
    generation_df: pd.DataFrame,
    is_hours_resolution: bool,
    hour_sample: np.ndarray,
    expected_output: pd.DataFrame,
) -> None:
    result = mocked_source_parameters_over_years_query._calculate_demand_for_generator(
        conversion_rates, generation_df, is_hours_resolution, hour_sample
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "generators, is_hours_resolution, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            False,
            pd.DataFrame(
                [
                    [5.0, 5.0, 5.0, 0.5, 0.5, 0.5],
                    [5.0, 5.0, 5.0, 0.5, 0.5, 0.5],
                ],
                index=["ET1", "ET2"],
                columns=pd.MultiIndex.from_product([["gen1", "gen2"], [0, 1, 2]]),
            ),
            id="all_gens_no_hour_resolution",
        ),
        pytest.param(
            ["gen1"],
            False,
            pd.DataFrame(
                [
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                ],
                index=["ET1", "ET2"],
                columns=pd.MultiIndex.from_product([["gen1"], [0, 1, 2]]),
            ),
            id="gen1_filtered_no_hour_resolution",
        ),
        pytest.param(
            ["gen1", "gen2"],
            True,
            pd.DataFrame(
                np.array(
                    [
                        [5.0, 5.0, 5.0, 0.5, 0.5, 0.5],
                        [5.0, 5.0, 5.0, 0.5, 0.5, 0.5],
                    ]
                ),
                index=pd.MultiIndex.from_product(
                    [[0], ["ET1", "ET2"]], names=["Hour", "Energy Type"]
                ),
                columns=pd.MultiIndex.from_product(
                    [
                        [
                            "gen1",
                            "gen2",
                        ],
                        [0, 1, 2],
                    ]
                ),
            ),
            id="all_gens_hour_resolution",
        ),
        pytest.param(
            [],
            False,
            pd.DataFrame(),
            id="empty_case",
        ),
    ],
)
def test__get_generation_demand(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generator_results_per_year_per_hour: dict[str, pd.DataFrame],
    generators: list[str],
    is_hours_resolution: bool,
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._generator_results = {
        "generation": generator_results_per_year_per_hour
    }

    gen1 = MagicMock(energy_source_type="GEN_ET_1")
    gen1.name = "gen1"

    gen2 = MagicMock(energy_source_type="GEN_ET_2")
    gen2.name = "gen2"

    gen3 = MagicMock(energy_source_type="GEN_ET_3")
    gen3.name = "gen3"

    gen_et_1 = MagicMock(
        conversion_rate={"ET1": pd.Series({0: 2, 1: 4}), "ET2": pd.Series({0: 2, 1: 4})}
    )
    gen_et_1.name = "GEN_ET_1"
    gen_et_2 = MagicMock(
        conversion_rate={
            "ET1": pd.Series({0: 10, 1: 20}),
            "ET2": pd.Series({0: 10, 1: 20}),
        }
    )
    gen_et_2.name = "GEN_ET_2"
    gen_et_3 = MagicMock(conversion_rate=None)
    gen_et_3.name = "GEN_ET_3"

    mocked_source_parameters_over_years_query._network.generators = {
        gen.name: gen for gen in (gen1, gen2, gen3)
    }
    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2, gen_et_3)
    }
    mocked_source_parameters_over_years_query._hour_sample = np.array([0])

    result = mocked_source_parameters_over_years_query._get_generation_demand(
        generators, [], is_hours_resolution
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filtered_generators, level, is_hours_resolution, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "type",
            False,
            pd.DataFrame(
                data=[
                    [5.5, 5.5],
                    [5.5, 5.5],
                    [5.5, 5.5],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
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
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_types",
        ),
        pytest.param(
            ["gen3"],
            "type",
            False,
            pd.DataFrame(
                data=[
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("group2", 0),
                        ("group2", 1),
                        ("group2", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="gen3_filtered_type",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "element",
            False,
            pd.DataFrame(
                data=[
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                        ("gen2", 0),
                        ("gen2", 1),
                        ("gen2", 2),
                        ("gen3", 0),
                        ("gen3", 1),
                        ("gen3", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="all_elements",
        ),
        pytest.param(
            ["gen1", "gen2"],
            "element",
            True,
            pd.DataFrame(
                data=[
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0, 0),
                        ("gen1", 1, 0),
                        ("gen1", 2, 0),
                        ("gen2", 0, 0),
                        ("gen2", 1, 0),
                        ("gen2", 2, 0),
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["ET1", "ET2"], name="Energy Type"),
            ),
            id="gen1_gen2_filtered_hour_resolution",
        ),
        pytest.param(
            [],
            "element",
            False,
            pd.DataFrame(),
            id="empty_case",
        ),
    ],
)
def test_generation_demand(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generator_results_per_year_per_hour: dict[str, pd.DataFrame],
    filtered_generators: list[str],
    level: Literal["type", "element"],
    is_hours_resolution: bool,
    expected_output: pd.DataFrame,
) -> None:
    mocked_source_parameters_over_years_query._generator_results = {
        "generation": generator_results_per_year_per_hour
    }

    gen1 = MagicMock(energy_source_type="GEN_ET_1")
    gen1.name = "gen1"

    gen2 = MagicMock(energy_source_type="GEN_ET_2")
    gen2.name = "gen2"

    gen3 = MagicMock(energy_source_type="GEN_ET_1")
    gen3.name = "gen3"

    gen_et_1 = MagicMock(
        conversion_rate={"ET1": pd.Series({0: 2, 1: 4}), "ET2": pd.Series({0: 2, 1: 4})}
    )
    gen_et_1.name = "GEN_ET_1"
    gen_et_2 = MagicMock(
        conversion_rate={
            "ET1": pd.Series({0: 10, 1: 20}),
            "ET2": pd.Series({0: 10, 1: 20}),
        }
    )
    gen_et_2.name = "GEN_ET_2"

    mocked_source_parameters_over_years_query._network.generators = {
        gen.name: gen for gen in (gen1, gen2, gen3)
    }
    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2)
    }
    mocked_source_parameters_over_years_query._hour_sample = np.array([0])

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen1", "gen2"},
        "group2": {"gen3"},
    }

    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=(filtered_generators, []),
    ):
        result = mocked_source_parameters_over_years_query.get_generation_demand(
            level, None, None, is_hours_resolution
        )
        assert_frame_equal(result, expected_output)
