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

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.fixture
def emission_mock_parameters(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    generator_results_per_year_per_hour: dict[str, pd.DataFrame],
) -> SourceParametersOverYearsQuery:

    mocked_source_parameters_over_years_query._generator_results = {
        "generation": generator_results_per_year_per_hour
    }

    gen1 = MagicMock(energy_source_type="GEN_ET_1")
    gen1.name = "gen1"

    gen2 = MagicMock(energy_source_type="GEN_ET_2")
    gen2.name = "gen2"

    gen3 = MagicMock(energy_source_type="GEN_ET_1")
    gen3.name = "gen3"

    mocked_source_parameters_over_years_query._network.generators = {
        gen.name: gen for gen in (gen1, gen2, gen3)
    }

    gen3.name = "gen3"

    gen_et_1 = MagicMock(
        fuel="fuel1",
        emission_reduction={
            "EMISSION1": pd.Series([0.1, 0.1, 0.1]),
            "EMISSION2": pd.Series([0.2, 0.2, 0.2]),
        },
    )
    gen_et_1.name = "GEN_ET_1"

    gen_et_2 = MagicMock(
        fuel="fuel2",
        emission_reduction={
            "EMISSION1": pd.Series([1, 1, 1]),
            "EMISSION2": pd.Series([0.5, 0.5, 0.5]),
        },
    )
    gen_et_2.name = "GEN_ET_2"

    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2)
    }

    fuel1 = MagicMock(
        energy_per_unit=100,
        cost=pd.Series([100, 200, 300]),
        emission={"EMISSION1": 2, "EMISSION2": 2},
    )
    fuel1.name = "fuel1"

    fuel2 = MagicMock(
        energy_per_unit=100,
        cost=pd.Series([2, 2, 2]),
        emission={"EMISSION1": 5, "EMISSION2": 5},
    )
    fuel2.name = "fuel2"

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "GEN_ET_1": {"gen1", "gen3"},
        "GEN_ET_2": {"gen2"},
    }

    mocked_source_parameters_over_years_query._network.emission_types = [
        "EMISSION1",
        "EMISSION2",
    ]

    mocked_source_parameters_over_years_query._network.fuels = {
        fuel.name: fuel for fuel in (fuel1, fuel2)
    }
    mocked_source_parameters_over_years_query._year_sample = np.array([0, 1, 2])

    mocked_source_parameters_over_years_query._years_binding = None

    return mocked_source_parameters_over_years_query


@pytest.mark.parametrize(
    "is_hours_resolution, fuel_usage_df, fuels_emissions, expected_output",
    [
        pytest.param(
            False,
            pd.DataFrame(
                [
                    [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
                    [10, 10, 10, 2.0, 2.0, 2.0, 4, 4, 4],
                ],
                index=pd.Index(["fuel1", "fuel2"]),
                columns=pd.MultiIndex.from_tuples(
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
                    ]
                ),
            ),
            {
                "fuel1": {"EMISSION1": 2, "EMISSION2": 2},
                "fuel2": {"EMISSION1": 5, "EMISSION2": 5},
            },
            [
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 0,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 1,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 2,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 0,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 1,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 2,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 0,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 1,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 2,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 0,
                    "value": 5.0,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 1,
                    "value": 5.0,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 2,
                    "value": 5.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 0,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 1,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 2,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 0,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 1,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 2,
                    "value": 0.16,
                },
            ],
            id="valid_data",
        ),
        pytest.param(
            True,
            pd.DataFrame(
                [
                    [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "fuel1"),
                        (1, "fuel1"),
                        (0, "fuel2"),
                        (1, "fuel2"),
                    ]
                ),
                columns=pd.MultiIndex.from_product(
                    [["gen1", "gen2", "gen3"], [0, 1, 2]]
                ),
            ),
            {
                "fuel1": {"EMISSION1": 2, "EMISSION2": 2},
                "fuel2": {"EMISSION1": 5, "EMISSION2": 5},
            },
            [
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 0,
                    "hour": 0,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 0,
                    "hour": 1,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 1,
                    "hour": 0,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 1,
                    "hour": 1,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 2,
                    "hour": 0,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen1",
                    "year": 2,
                    "hour": 1,
                    "value": 0.36,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 0,
                    "hour": 0,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 0,
                    "hour": 1,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 1,
                    "hour": 0,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 1,
                    "hour": 1,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 2,
                    "hour": 0,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen1",
                    "year": 2,
                    "hour": 1,
                    "value": 0.32,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 0,
                    "hour": 0,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 0,
                    "hour": 1,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 1,
                    "hour": 0,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 1,
                    "hour": 1,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 2,
                    "hour": 0,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen2",
                    "year": 2,
                    "hour": 1,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 0,
                    "hour": 0,
                    "value": 0.5,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 0,
                    "hour": 1,
                    "value": 1.25,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 1,
                    "hour": 0,
                    "value": 0.5,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 1,
                    "hour": 1,
                    "value": 1.25,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 2,
                    "hour": 0,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen2",
                    "year": 2,
                    "hour": 1,
                    "value": 0.0,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 0,
                    "hour": 0,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 0,
                    "hour": 1,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 1,
                    "hour": 0,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 1,
                    "hour": 1,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 2,
                    "hour": 0,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION1",
                    "generator_name": "gen3",
                    "year": 2,
                    "hour": 1,
                    "value": 0.18,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 0,
                    "hour": 0,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 0,
                    "hour": 1,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 1,
                    "hour": 0,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 1,
                    "hour": 1,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 2,
                    "hour": 0,
                    "value": 0.16,
                },
                {
                    "emission_type": "EMISSION2",
                    "generator_name": "gen3",
                    "year": 2,
                    "hour": 1,
                    "value": 0.16,
                },
            ],
            id="valid_data_hour_resolution",
        ),
        pytest.param(
            False,
            pd.DataFrame(),
            {
                "fuel1": {"EMISSION1": 2, "EMISSION2": 2},
                "fuel2": {"EMISSION1": 5, "EMISSION2": 5},
            },
            [],
            id="empty_fuel_usage_df",
        ),
        pytest.param(
            False,
            pd.DataFrame(
                [
                    [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
                    [10, 10, 10, 2.0, 2.0, 2.0, 4, 4, 4],
                ],
                index=pd.Index(["fuel1", "fuel2"]),
                columns=pd.MultiIndex.from_tuples(
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
                    ]
                ),
            ),
            {},
            [],
            id="empty_emission_dict",
        ),
    ],
)
def test_get_emission_dfs_dicts(
    is_hours_resolution: bool,
    fuel_usage_df: pd.DataFrame,
    fuels_emissions: dict[str, dict[str, float]],
    emission_mock_parameters: SourceParametersOverYearsQuery,
    expected_output: list[dict[str, Any]],
) -> None:
    result = emission_mock_parameters._get_emission_dfs_dicts(
        is_hours_resolution, fuel_usage_df, fuels_emissions
    )
    for res, exp in zip(result, expected_output):
        assert res["emission_type"] == exp["emission_type"]
        assert res["generator_name"] == exp["generator_name"]
        assert res["year"] == exp["year"]
        assert res["value"] == pytest.approx(exp["value"])


@pytest.mark.parametrize(
    "level, generators, filter_names, is_hours_resolution, expected_output",
    [
        pytest.param(
            "type",
            ["gen1", "gen2", "gen3"],
            None,
            False,
            pd.DataFrame(
                data=[
                    [0.36, 0.32],
                    [0.36, 0.32],
                    [0.36, 0.32],
                    [0.0, 0.25],
                    [0.0, 0.25],
                    [0.0, 0.25],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("GEN_ET_1", 0),
                        ("GEN_ET_1", 1),
                        ("GEN_ET_1", 2),
                        ("GEN_ET_2", 0),
                        ("GEN_ET_2", 1),
                        ("GEN_ET_2", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["EMISSION1", "EMISSION2"], name="Emission"),
            ),
            id="valid_data_type",
        ),
        pytest.param(
            "type",
            ["gen1", "gen2", "gen3"],
            None,
            True,
            pd.DataFrame(
                data=[
                    [0.18, 0.16],
                    [0.18, 0.16],
                    [0.18, 0.16],
                    [0.18, 0.16],
                    [0.18, 0.16],
                    [0.18, 0.16],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("GEN_ET_1", 0, 0),
                        ("GEN_ET_1", 0, 1),
                        ("GEN_ET_1", 1, 0),
                        ("GEN_ET_1", 1, 1),
                        ("GEN_ET_1", 2, 0),
                        ("GEN_ET_1", 2, 1),
                        ("GEN_ET_2", 0, 0),
                        ("GEN_ET_2", 0, 1),
                        ("GEN_ET_2", 1, 0),
                        ("GEN_ET_2", 1, 1),
                        ("GEN_ET_2", 2, 0),
                        ("GEN_ET_2", 2, 1),
                    ],
                    names=["Network element type", "Year", "Hour"],
                ),
                columns=pd.Index(["EMISSION1", "EMISSION2"], name="Emission"),
            ),
            id="valid_data_hour_resolution",
        ),
        pytest.param(
            "type",
            ["gen1", "gen2", "gen3"],
            ["GEN_ET_2"],
            False,
            pd.DataFrame(
                data=[
                    [0.0, 0.25],
                    [0.0, 0.25],
                    [0.0, 0.25],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("GEN_ET_2", 0),
                        ("GEN_ET_2", 1),
                        ("GEN_ET_2", 2),
                    ],
                    names=["Network element type", "Year"],
                ),
                columns=pd.Index(["EMISSION1", "EMISSION2"], name="Emission"),
            ),
            id="filtered_GEN_ET_2",
        ),
        pytest.param(
            "element",
            ["gen1", "gen2", "gen3"],
            None,
            False,
            pd.DataFrame(
                data=[
                    [0.36, 0.32],
                    [0.36, 0.32],
                    [0.36, 0.32],
                    [0.0, 0.25],
                    [0.0, 0.25],
                    [0.0, 0.25],
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
                columns=pd.Index(["EMISSION1", "EMISSION2"], name="Emission"),
            ),
            id="valid_data_element",
        ),
        pytest.param(
            "element",
            ["gen1", "gen2", "gen3"],
            ["gen2"],
            True,
            pd.DataFrame(
                data=[
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                    [0.0, 0.125],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen2", 0, 0),
                        ("gen2", 0, 1),
                        ("gen2", 1, 0),
                        ("gen2", 1, 1),
                        ("gen2", 2, 0),
                        ("gen2", 2, 1),
                    ],
                    names=["Network element name", "Year", "Hour"],
                ),
                columns=pd.Index(["EMISSION1", "EMISSION2"], name="Emission"),
            ),
            id="filtred_gen2_hour_resolution",
        ),
        pytest.param(
            "element",
            [],
            None,
            True,
            pd.DataFrame(),
            id="no_generators",
        ),
    ],
)
def test_get_emission(
    is_hours_resolution: bool,
    generators: list[str],
    filter_names: list[str] | None,
    level: Literal["type", "element"],
    emission_mock_parameters: SourceParametersOverYearsQuery,
    expected_output: pd.DataFrame,
) -> None:
    with patch.object(
        emission_mock_parameters,
        "_filter_elements",
        return_value=(generators, []),
    ):
        result = emission_mock_parameters.get_emission(
            level, None, filter_names, is_hours_resolution, False
        )
        assert_frame_equal(result, expected_output)
