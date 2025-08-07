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

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.fixture
def network_params_mock(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
) -> SourceParametersOverYearsQuery:

    gen_et_1 = MagicMock(
        capex=pd.Series(10000, index=range(5), name="CAPEX"),
        opex=pd.Series(1000, index=range(5), name="OPEX"),
    )
    gen_et_1.name = "GEN_ET_1"
    gen_et_2 = MagicMock(
        capex=pd.Series([100, 200, 300, 400, 500], name="CAPEX"),
        opex=pd.Series([1000, 100, 50, 25, 10], index=range(5), name="OPEX"),
    )
    gen_et_2.name = "GEN_ET_2"

    stor_et1 = MagicMock(
        capex=pd.Series(2, index=range(5), name="CAPEX"),
        opex=pd.Series(5, index=range(5), name="OPEX"),
    )
    stor_et1.name = "STOR_ET_1"

    fuel1 = MagicMock(
        availability=pd.Series([1000, 1000, 1000, 1000, 1000]),
        cost=pd.Series([100, 100, 100, 100, 100]),
    )
    fuel1.name = "FUEL1"

    fuel2 = MagicMock(
        availability=pd.Series([100, 200, 300, 400, 500]),
        cost=pd.Series([1, 2, 3, 4, 5]),
    )
    fuel2.name = "FUEL2"

    mocked_source_parameters_over_years_query._network.storage_types = {
        stor_et1.name: stor_et1
    }
    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2)
    }
    mocked_source_parameters_over_years_query._network.fuels = {
        fuel.name: fuel for fuel in (fuel1, fuel2)
    }
    return mocked_source_parameters_over_years_query


@pytest.mark.parametrize(
    "year_sample, tech_data, keys, expected_output",
    [
        pytest.param(
            np.array([0, 1]),
            [
                ("tech1", {"key1": pd.Series([1, 2]), "key2": pd.Series([3, 4])}),
                ("tech2", {"key1": pd.Series([5, 6]), "key2": pd.Series([7, 8])}),
            ],
            ["key1", "key2"],
            pd.DataFrame(
                {
                    ("tech1", 0): [1, 3],
                    ("tech1", 1): [2, 4],
                    ("tech2", 0): [5, 7],
                    ("tech2", 1): [6, 8],
                },
                index=["key1", "key2"],
            ),
            id="valid_data",
        ),
        pytest.param(
            np.array([0]), [], ["key1", "key2"], pd.DataFrame(), id="empty_techs"
        ),
        pytest.param(
            np.array([0]),
            [
                ("tech1", {"key1": pd.Series([1, 2]), "key2": pd.Series([3, 4])}),
                ("tech2", {"key1": pd.Series([5, 6]), "key2": pd.Series([7, 8])}),
            ],
            [],
            pd.DataFrame(),
            id="empty_keys",
        ),
    ],
)
def test_get_data_per_unit(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
    year_sample: np.ndarray,
    tech_data: list[tuple[str, dict[str, pd.Series]]],
    keys: list[str],
    expected_output: pd.DataFrame,
) -> None:
    techs = []
    for name, data in tech_data:
        mock = MagicMock()
        mock.name = name
        for key, value in data.items():
            setattr(mock, key, value)
        techs.append(mock)

    mocked_source_parameters_over_years_query._year_sample = year_sample
    result = mocked_source_parameters_over_years_query._get_data_per_unit(techs, keys)
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filter_names, year_sample, expected_output",
    [
        pytest.param(
            None,
            np.array([0, 1, 2, 3]),
            pd.DataFrame(
                data=[
                    [10000, 1000],
                    [10000, 1000],
                    [10000, 1000],
                    [10000, 1000],
                    [100, 1000],
                    [200, 100],
                    [300, 50],
                    [400, 25],
                    [2, 5],
                    [2, 5],
                    [2, 5],
                    [2, 5],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("GEN_ET_1", 0),
                        ("GEN_ET_1", 1),
                        ("GEN_ET_1", 2),
                        ("GEN_ET_1", 3),
                        ("GEN_ET_2", 0),
                        ("GEN_ET_2", 1),
                        ("GEN_ET_2", 2),
                        ("GEN_ET_2", 3),
                        ("STOR_ET_1", 0),
                        ("STOR_ET_1", 1),
                        ("STOR_ET_1", 2),
                        ("STOR_ET_1", 3),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["capex", "opex"], name="Cost Type"),
            ),
            id="valid_data",
        ),
        pytest.param(
            None,
            np.array([0]),
            pd.DataFrame(
                data=[
                    [10000, 1000],
                    [100, 1000],
                    [2, 5],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("GEN_ET_1", 0),
                        ("GEN_ET_2", 0),
                        ("STOR_ET_1", 0),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["capex", "opex"], name="Cost Type"),
            ),
            id="only_0_year",
        ),
        pytest.param(
            ["GEN_ET_1", "STOR_ET_1"],
            np.array([0, 1, 2, 3]),
            pd.DataFrame(
                data=[
                    [10000, 1000],
                    [10000, 1000],
                    [10000, 1000],
                    [10000, 1000],
                    [2, 5],
                    [2, 5],
                    [2, 5],
                    [2, 5],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("GEN_ET_1", 0),
                        ("GEN_ET_1", 1),
                        ("GEN_ET_1", 2),
                        ("GEN_ET_1", 3),
                        ("STOR_ET_1", 0),
                        ("STOR_ET_1", 1),
                        ("STOR_ET_1", 2),
                        ("STOR_ET_1", 3),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["capex", "opex"], name="Cost Type"),
            ),
            id="filtered_gen_et1_stor_et_1",
        ),
    ],
)
def test_get_network_costs_per_tech_type(
    network_params_mock: SourceParametersOverYearsQuery,
    filter_names: list[str] | None,
    expected_output: pd.DataFrame,
    year_sample: np.ndarray,
) -> None:
    network_params_mock._year_sample = year_sample
    result = network_params_mock.get_network_costs_per_tech_type(filter_names)
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filter_names, year_sample, expected_output",
    [
        pytest.param(
            None,
            np.array([0, 1, 2, 3]),
            pd.DataFrame(
                data=[
                    [100],
                    [100],
                    [100],
                    [100],
                    [1],
                    [2],
                    [3],
                    [4],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("FUEL1", 0),
                        ("FUEL1", 1),
                        ("FUEL1", 2),
                        ("FUEL1", 3),
                        ("FUEL2", 0),
                        ("FUEL2", 1),
                        ("FUEL2", 2),
                        ("FUEL2", 3),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["cost"], name="Fuel"),
            ),
            id="valid_data",
        ),
        pytest.param(
            None,
            np.array([0, 1]),
            pd.DataFrame(
                data=[
                    [100],
                    [100],
                    [1],
                    [2],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("FUEL1", 0),
                        ("FUEL1", 1),
                        ("FUEL2", 0),
                        ("FUEL2", 1),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["cost"], name="Fuel"),
            ),
            id="years_0_and_1",
        ),
        pytest.param(
            ["FUEL2"],
            np.array([0, 1, 2, 3]),
            pd.DataFrame(
                data=[
                    [1],
                    [2],
                    [3],
                    [4],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("FUEL2", 0),
                        ("FUEL2", 1),
                        ("FUEL2", 2),
                        ("FUEL2", 3),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["cost"], name="Fuel"),
            ),
            id="filtered_fuel2",
        ),
    ],
)
def test_get_network_fuel_cost(
    network_params_mock: SourceParametersOverYearsQuery,
    filter_names: list[str] | None,
    expected_output: pd.DataFrame,
    year_sample: np.ndarray,
) -> None:
    network_params_mock._year_sample = year_sample
    result = network_params_mock.get_network_fuel_cost(filter_names)
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "filter_names, year_sample, expected_output",
    [
        pytest.param(
            None,
            np.array([0, 1, 2, 3]),
            pd.DataFrame(
                data=[
                    [1000],
                    [1000],
                    [1000],
                    [1000],
                    [100],
                    [200],
                    [300],
                    [400],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("FUEL1", 0),
                        ("FUEL1", 1),
                        ("FUEL1", 2),
                        ("FUEL1", 3),
                        ("FUEL2", 0),
                        ("FUEL2", 1),
                        ("FUEL2", 2),
                        ("FUEL2", 3),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["availability"], name="Fuel"),
            ),
            id="valid_data",
        ),
        pytest.param(
            None,
            np.array([0]),
            pd.DataFrame(
                data=[
                    [1000],
                    [100],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("FUEL1", 0),
                        ("FUEL2", 0),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["availability"], name="Fuel"),
            ),
            id="years_0",
        ),
        pytest.param(
            ["FUEL2"],
            np.array([0, 1, 2, 3]),
            pd.DataFrame(
                data=[
                    [100],
                    [200],
                    [300],
                    [400],
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("FUEL2", 0),
                        ("FUEL2", 1),
                        ("FUEL2", 2),
                        ("FUEL2", 3),
                    ],
                    names=["Network element name", "Year"],
                ),
                columns=pd.Index(["availability"], name="Fuel"),
            ),
            id="filtered_fuel2",
        ),
    ],
)
def test_get_network_fuel_availability(
    network_params_mock: SourceParametersOverYearsQuery,
    filter_names: list[str] | None,
    expected_output: pd.DataFrame,
    year_sample: np.ndarray,
) -> None:
    network_params_mock._year_sample = year_sample
    result = network_params_mock.get_network_fuel_availability(filter_names)
    assert_frame_equal(result, expected_output)
