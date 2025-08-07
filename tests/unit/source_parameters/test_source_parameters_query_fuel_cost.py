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
    "generators, level, year_sample, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "type",
            np.array([0, 1, 2]),
            pd.DataFrame(
                data=[
                    [20.0, 0.0],
                    [40.0, 0.0],
                    [60.0, 0.0],
                    [0.0, 0.2],
                    [0.0, 0.2],
                    [0.0, 0.2],
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
                columns=pd.Index(["fuel1", "fuel2"], name="Fuel"),
            ),
            id="all_gens_type",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            "element",
            np.array([0, 1, 2]),
            pd.DataFrame(
                data=[
                    [20.0, np.nan],
                    [40.0, np.nan],
                    [60.0, np.nan],
                    [np.nan, 0.2],
                    [np.nan, 0.2],
                    [np.nan, 0.2],
                    [0.0, np.nan],
                    [0.0, np.nan],
                    [0.0, np.nan],
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
                columns=pd.Index(["fuel1", "fuel2"], name="Fuel"),
            ),
            id="all_gens_element",
        ),
        pytest.param(
            ["gen1"],
            "element",
            np.array([0, 1, 2]),
            pd.DataFrame(
                data=[
                    [20.0],
                    [40.0],
                    [60.0],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("gen1", 0),
                        ("gen1", 1),
                        ("gen1", 2),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["fuel1"], name="Fuel"),
            ),
            id="only_gen1",
        ),
        pytest.param(
            [],
            "element",
            np.array([0, 1, 2]),
            pd.DataFrame(),
            id="no_gens",
        ),
    ],
)
def test_get_fuel_cost(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    level: Literal["type", "element"],
    generators: list[str],
    year_sample: np.ndarray,
    generator_results_per_year_per_hour: dict[str, pd.DataFrame],
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

    mocked_source_parameters_over_years_query._network.generators = {
        gen.name: gen for gen in (gen1, gen2, gen3)
    }

    gen_et_1 = MagicMock(fuel="fuel1")
    gen_et_1.name = "GEN_ET_1"

    gen_et_2 = MagicMock(fuel="fuel2")
    gen_et_2.name = "GEN_ET_2"

    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2)
    }

    fuel1 = MagicMock(energy_per_unit=100, cost=pd.Series([100, 200, 300]))
    fuel1.name = "fuel1"

    fuel2 = MagicMock(energy_per_unit=100, cost=pd.Series([2, 2, 2]))
    fuel2.name = "fuel2"

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "group1": {"gen1", "gen3"},
        "group2": {"gen2"},
    }

    mocked_source_parameters_over_years_query._network.fuels = {
        fuel.name: fuel for fuel in (fuel1, fuel2)
    }
    mocked_source_parameters_over_years_query._year_sample = year_sample
    with patch.object(
        mocked_source_parameters_over_years_query,
        "_filter_elements",
        return_value=(generators, []),
    ):
        result = mocked_source_parameters_over_years_query.get_fuel_cost(
            level, None, None
        )
        assert_frame_equal(result, expected_output)
