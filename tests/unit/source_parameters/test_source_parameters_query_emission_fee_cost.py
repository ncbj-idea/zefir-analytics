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


@pytest.fixture
def emission_cost_mock_parameters(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
) -> SourceParametersOverYearsQuery:

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "GEN_ET_1": {"gen1", "gen3"},
        "GEN_ET_2": {"gen2"},
    }

    EF1 = MagicMock(emission_type="EMISSION1", price=pd.Series([10, 10, 10]))
    EF1.name = "EMISSION_FEE1"
    EF2 = MagicMock(emission_type="EMISSION2", price=pd.Series([2, 2, 2]))
    EF2.name = "EMISSION_FEE2"
    EF3 = MagicMock(emission_type="EMISSION1", price=pd.Series([1000, 1000, 1000]))
    EF3.name = "EMISSION_FEE3"

    mocked_source_parameters_over_years_query._network.emission_fees = {
        EF.name: EF for EF in (EF1, EF2, EF3)
    }

    mocked_source_parameters_over_years_query._year_sample = np.array([0, 1, 2])

    mocked_source_parameters_over_years_query._years_binding = None

    return mocked_source_parameters_over_years_query


@pytest.mark.parametrize(
    "emission_fee_name, gen_name, total_price, expected_output",
    [
        pytest.param(
            "EMISSION_FEE1",
            "gen1",
            None,
            pd.DataFrame(
                {"EMISSION1": [20.0, 40.0, 60.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["gen_name", "year"],
                ),
            ),
            id="gen1_EMISSION_FEE1",
        ),
        pytest.param(
            "EMISSION_FEE3",
            "gen1",
            None,
            pd.DataFrame(
                {"EMISSION1": [2000.0, 4000.0, 6000.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["gen_name", "year"],
                ),
            ),
            id="gen1_EMISSION_FEE3",
        ),
        pytest.param(
            "EMISSION_FEE2",
            "gen1",
            None,
            pd.DataFrame(
                {"EMISSION2": [1.0, 1.0, 1.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["gen_name", "year"],
                ),
            ),
            id="gen1_EMISSION_FEE2",
        ),
        pytest.param(
            "EMISSION_FEE1",
            "gen1",
            pd.Series([100, 100, 100, 100, 100, 100, 100]),
            pd.DataFrame(
                {"EMISSION1": [200.0, 400.0, 600.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["gen_name", "year"],
                ),
            ),
            id="gen1_EMISSION_FEE1_total_cost",
        ),
        pytest.param(
            "EMISSION_FEE_not_in_network",
            "gen1",
            None,
            pd.DataFrame(),
            id="emission_fee_not_in_network",
        ),
        pytest.param(
            "EMISSION_FEE1",
            "im_storage_now",
            None,
            pd.DataFrame(),
            id="gen_not_in_emission_df",
        ),
        pytest.param(
            "EMISSION_FEE1",
            "gen1",
            pd.Series([100]),
            pd.DataFrame(
                {"EMISSION1": [20.0, 40.0, 60.0]},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["gen_name", "year"],
                ),
            ),
            id="gen_1_total_price_shorter_than_year_sample",
        ),
    ],
)
def test_calculate_emission_fee_total_cost(
    emission_cost_mock_parameters: SourceParametersOverYearsQuery,
    emission_fee_name: str,
    gen_name: str,
    total_price: pd.Series | None,
    expected_output: pd.DataFrame,
) -> None:
    emissions_df = pd.DataFrame(
        data=[
            [2, 0.5],
            [4, 0.5],
            [6, 0.5],
            [0.5, 0.25],
            [0.5, 0.25],
            [0.5, 0.25],
            [10.0, 5.0],
            [15.0, 5.0],
            [20.0, 10.0],
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
    )
    result = emission_cost_mock_parameters._calculate_emission_fee_total_cost(
        emissions_df, emission_fee_name, gen_name, total_price
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "level, filter_names, filtered_gens_with_emission_fees, expected_output",
    [
        pytest.param(
            "type",
            None,
            {
                "gen1": {"EMISSION_FEE1"},
                "gen2": {"EMISSION_FEE2"},
                "gen3": {"EMISSION_FEE3"},
            },
            pd.DataFrame(
                {
                    "EMISSION1": [10020.0, 15040.0, 20060.0, 0.0, 0.0, 0.0],
                    "EMISSION2": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                },
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
            "element",
            None,
            {
                "gen1": {"EMISSION_FEE1"},
                "gen2": {"EMISSION_FEE2"},
                "gen3": {"EMISSION_FEE3"},
            },
            pd.DataFrame(
                {
                    "EMISSION1": [
                        20.0,
                        40.0,
                        60.0,
                        0.0,
                        0.0,
                        0.0,
                        10000.0,
                        15000.0,
                        20000.0,
                    ],
                    "EMISSION2": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                },
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
            "type",
            ["GEN_ET_2"],
            {
                "gen1": {"EMISSION_FEE1"},
                "gen2": {"EMISSION_FEE2"},
                "gen3": {"EMISSION_FEE3"},
            },
            pd.DataFrame(
                {
                    "EMISSION1": [0.0, 0.0, 0.0],
                    "EMISSION2": [0.5, 0.5, 0.5],
                },
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
            "type",
            None,
            {},
            pd.DataFrame(
                {
                    "EMISSION1": [np.nan] * 9,
                    "EMISSION2": [np.nan] * 9,
                },
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
            id="empty_emission_fee_dict",
        ),
        pytest.param(
            "type",
            None,
            {
                "gen1": {"EMISSION_FEE1", "EMISSION_FEE3"},
                "gen2": {"EMISSION_FEE2"},
            },
            pd.DataFrame(
                {
                    "EMISSION1": [2020.0, 4040.0, 6060.0, 0.0, 0.0, 0.0],
                    "EMISSION2": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                },
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
            id="same_ET_many_emission_fees",
        ),
        pytest.param(
            "type",
            None,
            {
                "gen1": {"EMISSION_FEE1", "EMISSION_FEE3"},
                "gen2": {"EMISSION_FEE2"},
                "gen4": {"EMISSION_FEE2"},
                "gen5": {"EMISSION_FEE1", "EMISSION_FEE3"},
            },
            pd.DataFrame(
                {
                    "EMISSION1": [2020.0, 4040.0, 6060.0, 0.0, 0.0, 0.0],
                    "EMISSION2": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                },
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
            id="More_gens_at_fee_dict_than_in_emission_results",
        ),
    ],
)
def test_get_emission_fee_total_cost(
    emission_cost_mock_parameters: SourceParametersOverYearsQuery,
    filter_names: list[str] | None,
    level: Literal["type", "element"],
    filtered_gens_with_emission_fees: dict[str, set[str]],
    expected_output: pd.DataFrame,
) -> None:
    emissions_df = pd.DataFrame(
        data=[
            [2, 0.5],
            [4, 0.5],
            [6, 0.5],
            [0.5, 0.25],
            [0.5, 0.25],
            [0.5, 0.25],
            [10.0, 5.0],
            [15.0, 5.0],
            [20.0, 10.0],
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
    )
    with patch.object(
        emission_cost_mock_parameters,
        "get_emission",
        return_value=(emissions_df),
    ), patch(
        "zefir_analytics._engine.data_queries.source_parameters_over_years.get_generators_emission_types",
        return_value=filtered_gens_with_emission_fees,
    ):
        result = emission_cost_mock_parameters.get_emission_fee_total_cost(
            level, filter_names
        )
        assert_frame_equal(result, expected_output, check_dtype=False)
