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
from zefir_analytics._engine.data_queries.utils import GeneratorCapacityCostLabel


@pytest.fixture
def capex_opex_mocked_source_parameters(
    mocked_source_parameters_over_years_query: SourceParametersOverYearsQuery,
) -> SourceParametersOverYearsQuery:
    gen1 = MagicMock(energy_source_type="GEN_ET_1")
    gen1.name = "gen1"

    gen2 = MagicMock(energy_source_type="GEN_ET_2")
    gen2.name = "gen2"

    gen3 = MagicMock(energy_source_type="GEN_ET_1")
    gen3.name = "gen3"

    mocked_source_parameters_over_years_query._network.generators = {
        gen.name: gen for gen in (gen1, gen2, gen3)
    }

    efficiency_df = pd.DataFrame(
        [2.0, 2.0, 6.0, 6.0],
        columns=pd.Index(["ET1"], name="energy_type"),
        index=pd.RangeIndex(0, 4, name="hour_idx"),
    )

    opex_series_gen_et_1 = pd.Series([1000] * 3, name="OPEX")
    opex_series_gen_et_2 = pd.Series([100] * 3, name="OPEX")

    gen_et_1 = MagicMock(
        efficiency=efficiency_df, build_time=2, opex=opex_series_gen_et_1
    )
    gen_et_1.name = "GEN_ET_1"

    gen_et_2 = MagicMock(
        efficiency=efficiency_df, build_time=1, opex=opex_series_gen_et_2
    )
    gen_et_2.name = "GEN_ET_2"

    mocked_source_parameters_over_years_query._network.generator_types = {
        gen_et.name: gen_et for gen_et in (gen_et_1, gen_et_2)
    }

    stor1 = MagicMock(energy_source_type="Storage_ET1")
    stor2 = MagicMock(energy_source_type="Storage_ET1")

    opex_series_stor = pd.Series([5] * 3, name="OPEX")
    stor_type_1 = MagicMock(energy_type="ET1", build_time=1, opex=opex_series_stor)

    mocked_source_parameters_over_years_query._network.storages = {
        "stor1": stor1,
        "stor2": stor2,
    }

    mocked_source_parameters_over_years_query._network.storage_types = {
        "Storage_ET1": stor_type_1,
    }

    mocked_source_parameters_over_years_query._energy_source_type_mapping = {
        "GEN_ET_1": {"gen1", "gen3"},
        "GEN_ET_2": {"gen2"},
        "Storage_ET1": {"stor1", "stor2"},
    }
    return mocked_source_parameters_over_years_query


@pytest.mark.parametrize(
    "capacity_label, capacity_gen_df, expected_output",
    [
        pytest.param(
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(),
            pd.DataFrame(),
            id="brutto_empty_df",
        ),
        pytest.param(
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame({"ET1": [1, 2, 3, 4, 5]}),
            pd.DataFrame({"ET1": [1, 2, 3, 4, 5]}),
            id="brutto_not_empty_df",
        ),
        pytest.param(
            GeneratorCapacityCostLabel.netto,
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "gen2": [10.0, 20.0, 30.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {
                    "gen1": [4.0, 8.0, 12.0],
                    "gen2": [40.0, 80.0, 120.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="netto_df_all_gens",
        ),
        pytest.param(
            GeneratorCapacityCostLabel.netto,
            pd.DataFrame(
                {
                    "gen1": [1.0, 1.0, 1.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {
                    "gen1": [4.0, 4.0, 4.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="netto_df_only_gen_1",
        ),
        pytest.param(
            GeneratorCapacityCostLabel.netto,
            pd.DataFrame(),
            pd.DataFrame(),
            id="netto_empty_df",
        ),
    ],
)
def test_handle_brutto_netto_generator(
    capacity_label: GeneratorCapacityCostLabel,
    capacity_gen_df: pd.DataFrame,
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    expected_output: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._capa_cost_label = capacity_label
    result = capex_opex_mocked_source_parameters._handle_brutto_netto_generator(
        capacity_gen_df
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "generator_capacity_df, storages_capacity_df, expected_gen_cap_plus_df, expected_stor_cap_plus_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                    "gen2": [10.0, 20.0, 30.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {"stor1": [0.0, 0.0, 0.0], "stor2": [5.0, 10.0, 15.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {
                    "gen1": [1.0, 0.0, 0.0],
                    "gen2": [10.0, 10.0, 0.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {"stor1": [0.0, 0.0, 0.0], "stor2": [5.0, 5.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="all_gens_stors",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [10.0, 110.0, 1100.0],
                    "gen2": [0.0, 0.0, 0.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {"stor1": [200.0, 100.0, 10.0], "stor2": [0.0, 0.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {
                    "gen1": [990.0, 0.0, 0.0],
                    "gen2": [0.0, 0.0, 0.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {"stor1": [0.0, 0.0, 0.0], "stor2": [0.0, 0.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="gen_1_stor_1",
        ),
    ],
)
def test_calculate_cap_plus(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    generator_capacity_df: pd.DataFrame,
    storages_capacity_df: pd.DataFrame,
    expected_gen_cap_plus_df: pd.DataFrame,
    expected_stor_cap_plus_df: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_capacity_df}
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storages_capacity_df}
    }
    result_gen, result_stor = capex_opex_mocked_source_parameters._calculate_cap_plus()
    assert_frame_equal(result_gen, expected_gen_cap_plus_df)
    assert_frame_equal(result_stor, expected_stor_cap_plus_df)


@pytest.mark.parametrize(
    "generator_capacity_df, storages_capacity_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [1.0, 2.0, 3.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {"stor1": [0.0, 0.0, 0.0], "stor2": [0.0, 0.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="only_gen_1",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [1.0, 0.0, 0.0],
                    "gen2": [10.0, 10.0, 0.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(),
            id="storage_empty_df,",
        ),
        pytest.param(
            pd.DataFrame(),
            pd.DataFrame(),
            id="both_empty_df,",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "gen4": [1.0, 2.0, 3.0],
                    "gen5": [1.0, 2.0, 3.0],
                    "gen6": [1.0, 2.0, 3.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {"stor1": [0.0, 0.0, 0.0], "stor2": [0.0, 0.0, 0.0]},
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="gens_not_mapped",
        ),
    ],
)
def test_calculate_cap_plus_with_errors(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    generator_capacity_df: pd.DataFrame,
    storages_capacity_df: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_capacity_df}
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storages_capacity_df}
    }
    with pytest.raises(KeyError):
        capex_opex_mocked_source_parameters._calculate_cap_plus()


@pytest.mark.parametrize(
    "generators, storages, cast_to_energy_source_type, year_sample, capacity_label, expected_output",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            False,
            None,
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(
                {
                    "gen1": [1000.0, 2000.0, 3000.0],
                    "gen2": [1000.0, 2000.0, 3000.0],
                    "gen3": [0.0, 0.0, 0.0],
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [25.0, 25.0, 25.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="all_gens_and_storage_brutto",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            False,
            None,
            GeneratorCapacityCostLabel.netto,
            pd.DataFrame(
                {
                    "gen1": [4000.0, 8000.0, 12000.0],
                    "gen2": [4000.0, 8000.0, 12000.0],
                    "gen3": [0.0, 0.0, 0.0],
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [25.0, 25.0, 25.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="all_gens_and_storage_netto",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            False,
            np.array([0, 1]),
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(
                {
                    "gen1": [1000.0, 2000.0],
                    "gen2": [1000.0, 2000.0],
                    "gen3": [0.0, 0.0],
                    "stor1": [0.0, 0.0],
                    "stor2": [25.0, 25.0],
                },
                index=pd.Index([0, 1], name="Year"),
            ),
            id="all_gens_and_storage_brutto_year_sample",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            True,
            None,
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(
                {
                    "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                    "GEN_ET_2": [1000.0, 2000.0, 3000.0],
                    "Storage_ET1": [25.0, 25.0, 25.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="all_gens_and_storage_brutto_cast_to_source_type",
        ),
        pytest.param(
            ["gen1"],
            ["stor1"],
            True,
            None,
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(
                {
                    "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                    "Storage_ET1": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="gen1_stor1_filtered_brutto_cast_to_source_type",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            [],
            False,
            None,
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(
                {
                    "gen1": [1000.0, 2000.0, 3000.0],
                    "gen2": [1000.0, 2000.0, 3000.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            id="empty_storage",
        ),
        pytest.param(
            [],
            [],
            False,
            None,
            GeneratorCapacityCostLabel.brutto,
            pd.DataFrame(
                index=pd.Index([0, 1, 2], name="Year"),
                columns=pd.Index([], dtype="object"),
            ),
            id="empty_case",
        ),
    ],
)
def test_calculate_opex(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    generators: list[str],
    storages: list[str],
    cast_to_energy_source_type: bool,
    year_sample: np.ndarray | None,
    expected_output: pd.DataFrame,
    capacity_label: GeneratorCapacityCostLabel,
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._capa_cost_label = capacity_label
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year}
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year}
    }
    capex_opex_mocked_source_parameters._year_sample = year_sample

    result = capex_opex_mocked_source_parameters._calculate_opex(
        generators, storages, cast_to_energy_source_type
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.parametrize(
    "capex_df, opex_df, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [1000.0, 2000.0, 3000.0],
                    "gen2": [1000.0, 2000.0, 3000.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=[0, 1, 2],
            ),
            pd.DataFrame(
                {
                    "gen1": [0.0, 0.0, 0.0],
                    "gen2": [25.0, 25.0, 25.0],
                    "gen3": [10.0, 10.0, 10.0],
                },
                index=[0, 1, 2],
            ),
            pd.DataFrame(
                {
                    ("gen1", 0): [1000.0, 0.0],
                    ("gen1", 1): [2000.0, 0.0],
                    ("gen1", 2): [3000.0, 0.0],
                    ("gen2", 0): [1000.0, 25.0],
                    ("gen2", 1): [2000.0, 25.0],
                    ("gen2", 2): [3000.0, 25.0],
                    ("gen3", 0): [0.0, 10.0],
                    ("gen3", 1): [0.0, 10.0],
                    ("gen3", 2): [0.0, 10.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="basic_case_with_more_cols",
        ),
        pytest.param(
            pd.DataFrame({"gen1": [1000.0, 2000.0]}, index=[0, 1]),
            pd.DataFrame({"gen1": [10.0, 20.0]}, index=[0, 1]),
            pd.DataFrame(
                {("gen1", 0): [1000.0, 10.0], ("gen1", 1): [2000.0, 20.0]},
                index=pd.Index(["capex", "opex"]),
            ),
            id="single_column",
        ),
        pytest.param(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            id="empty_dfs",
        ),
        pytest.param(
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "gen1": [0.0, 0.0, 0.0],
                    "gen2": [25.0, 25.0, 25.0],
                    "gen3": [10.0, 10.0, 10.0],
                },
                index=[0, 1, 2],
            ),
            pd.DataFrame(
                {
                    ("gen1", 0): [0.0],
                    ("gen1", 1): [0.0],
                    ("gen1", 2): [0.0],
                    ("gen2", 0): [25.0],
                    ("gen2", 1): [25.0],
                    ("gen2", 2): [25.0],
                    ("gen3", 0): [10.0],
                    ("gen3", 1): [10.0],
                    ("gen3", 2): [10.0],
                },
                index=pd.Index(["opex"]),
            ),
            id="empty_capex",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [1000.0, 2000.0, 3000.0],
                    "gen2": [1000.0, 2000.0, 3000.0],
                    "gen3": [0.0, 0.0, 0.0],
                },
                index=[0, 1, 2],
            ),
            pd.DataFrame(),
            pd.DataFrame(
                {
                    ("gen1", 0): [1000.0],
                    ("gen1", 1): [2000.0],
                    ("gen1", 2): [3000.0],
                    ("gen2", 0): [1000.0],
                    ("gen2", 1): [2000.0],
                    ("gen2", 2): [3000.0],
                    ("gen3", 0): [0.0],
                    ("gen3", 1): [0.0],
                    ("gen3", 2): [0.0],
                },
                index=pd.Index(["capex"]),
            ),
            id="empty_opex",
        ),
    ],
)
def test_format_capex_opex_dfs(
    capex_df: pd.DataFrame,
    opex_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
) -> None:
    result = capex_opex_mocked_source_parameters._format_capex_opex_dfs(
        capex_df, opex_df
    )
    assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "capex_gen, capex_stor, generators, storages, filter_names, expected_df",
    [
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            None,
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [1001.0, 1000.0],
                    ("GEN_ET_1", 1): [2002.0, 2000.0],
                    ("GEN_ET_1", 2): [3003.0, 3000.0],
                    ("GEN_ET_2", 0): [0.0, 1000.0],
                    ("GEN_ET_2", 1): [0.0, 2000.0],
                    ("GEN_ET_2", 2): [0.0, 3000.0],
                    ("Storage_ET1", 0): [11.0, 25.0],
                    ("Storage_ET1", 1): [11.0, 25.0],
                    ("Storage_ET1", 2): [11.0, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="all_gens_and_storages",
        ),
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "Aggr1",
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [1000.0, 1000.0],
                    ("GEN_ET_1", 1): [2000.0, 2000.0],
                    ("GEN_ET_1", 2): [3000.0, 3000.0],
                    ("GEN_ET_2", 0): [0.0, 1000.0],
                    ("GEN_ET_2", 1): [0.0, 2000.0],
                    ("GEN_ET_2", 2): [0.0, 3000.0],
                    ("Storage_ET1", 0): [10.0, 25.0],
                    ("Storage_ET1", 1): [10.0, 25.0],
                    ("Storage_ET1", 2): [10.0, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="filtered_Aggr1",
        ),
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "Aggr2",
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [1.0, 1000.0],
                    ("GEN_ET_1", 1): [2.0, 2000.0],
                    ("GEN_ET_1", 2): [3.0, 3000.0],
                    ("GEN_ET_2", 0): [0.0, 1000.0],
                    ("GEN_ET_2", 1): [0.0, 2000.0],
                    ("GEN_ET_2", 2): [0.0, 3000.0],
                    ("Storage_ET1", 0): [1.0, 25.0],
                    ("Storage_ET1", 1): [1.0, 25.0],
                    ("Storage_ET1", 2): [1.0, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="filtered_Aggr2",
        ),
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {},
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            None,
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [1001.0, 1000.0],
                    ("GEN_ET_1", 1): [2002.0, 2000.0],
                    ("GEN_ET_1", 2): [3003.0, 3000.0],
                    ("GEN_ET_2", 0): [0.0, 1000.0],
                    ("GEN_ET_2", 1): [0.0, 2000.0],
                    ("GEN_ET_2", 2): [0.0, 3000.0],
                    ("Storage_ET1", 0): [np.nan, 25.0],
                    ("Storage_ET1", 1): [np.nan, 25.0],
                    ("Storage_ET1", 2): [np.nan, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="no_storage_capex_results",
        ),
        pytest.param(
            {},
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            None,
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [np.nan, 1000.0],
                    ("GEN_ET_1", 1): [np.nan, 2000.0],
                    ("GEN_ET_1", 2): [np.nan, 3000.0],
                    ("GEN_ET_2", 0): [np.nan, 1000.0],
                    ("GEN_ET_2", 1): [np.nan, 2000.0],
                    ("GEN_ET_2", 2): [np.nan, 3000.0],
                    ("Storage_ET1", 0): [11.0, 25.0],
                    ("Storage_ET1", 1): [11.0, 25.0],
                    ("Storage_ET1", 2): [11.0, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="no_capex_capex_results",
        ),
        pytest.param(
            {},
            {},
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            None,
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [1000.0],
                    ("GEN_ET_1", 1): [2000.0],
                    ("GEN_ET_1", 2): [3000.0],
                    ("GEN_ET_2", 0): [1000.0],
                    ("GEN_ET_2", 1): [2000.0],
                    ("GEN_ET_2", 2): [3000.0],
                    ("Storage_ET1", 0): [25.0],
                    ("Storage_ET1", 1): [25.0],
                    ("Storage_ET1", 2): [25.0],
                },
                index=pd.Index(["opex"]),
            ).rename_axis(columns=[None, "Year"]),
            id="both_capex_result_empty",
        ),
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1"],
            ["stor1"],
            None,
            pd.DataFrame(
                {
                    ("GEN_ET_1", 0): [1001.0, 1000.0],
                    ("GEN_ET_1", 1): [2002.0, 2000.0],
                    ("GEN_ET_1", 2): [3003.0, 3000.0],
                    ("GEN_ET_2", 0): [0.0, np.nan],
                    ("GEN_ET_2", 1): [0.0, np.nan],
                    ("GEN_ET_2", 2): [0.0, np.nan],
                    ("Storage_ET1", 0): [11.0, 0.0],
                    ("Storage_ET1", 1): [11.0, 0.0],
                    ("Storage_ET1", 2): [11.0, 0.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="only_gen1_stor1",
        ),
    ],
)
def test__get_local_capex_opex(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    generators: list[str],
    storages: list[str],
    filter_names: list[str] | None,
    expected_df: pd.DataFrame,
    capex_gen: dict[str, pd.DataFrame],
    capex_stor: dict[str, pd.DataFrame],
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._capa_cost_label = (
        GeneratorCapacityCostLabel.brutto
    )
    capex_opex_mocked_source_parameters._year_sample = None
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year},
        "local_capex": capex_gen,
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year},
        "local_capex": capex_stor,
    }

    result = capex_opex_mocked_source_parameters._get_local_capex_opex(
        generators, storages, "aggr", filter_names
    )
    assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "capex_gen, capex_stor, generators, storages, filter_names, expected_df",
    [
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            None,
            pd.DataFrame(
                {
                    "capex": {
                        ("GEN_ET_1", 0): 1001.0,
                        ("GEN_ET_1", 1): 2002.0,
                        ("GEN_ET_1", 2): 3003.0,
                        ("GEN_ET_2", 0): 0.0,
                        ("GEN_ET_2", 1): 0.0,
                        ("GEN_ET_2", 2): 0.0,
                        ("Storage_ET1", 0): 11.0,
                        ("Storage_ET1", 1): 11.0,
                        ("Storage_ET1", 2): 11.0,
                    },
                    "opex": {
                        ("GEN_ET_1", 0): 1000.0,
                        ("GEN_ET_1", 1): 2000.0,
                        ("GEN_ET_1", 2): 3000.0,
                        ("GEN_ET_2", 0): 1000.0,
                        ("GEN_ET_2", 1): 2000.0,
                        ("GEN_ET_2", 2): 3000.0,
                        ("Storage_ET1", 0): 25.0,
                        ("Storage_ET1", 1): 25.0,
                        ("Storage_ET1", 2): 25.0,
                    },
                }
            ).rename_axis(index=["Network element name", "Year"]),
            id="all_gens_and_storages",
        ),
        pytest.param(
            {
                "Aggr1": pd.DataFrame(
                    {
                        "GEN_ET_1": [1000.0, 2000.0, 3000.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "GEN_ET_1": [1.0, 2.0, 3.0],
                        "GEN_ET_2": [0.0, 0.0, 0.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            {
                "Aggr1": pd.DataFrame(
                    {
                        "Storage_ET1": [10.0, 10.0, 10.0],
                    },
                    index=[0, 1, 2],
                ),
                "Aggr2": pd.DataFrame(
                    {
                        "Storage_ET1": [1.0, 1.0, 1.0],
                    },
                    index=[0, 1, 2],
                ),
            },
            ["gen1"],
            ["stor1"],
            None,
            pd.DataFrame(
                {
                    "capex": {
                        ("GEN_ET_1", 0): 1001.0,
                        ("GEN_ET_1", 1): 2002.0,
                        ("GEN_ET_1", 2): 3003.0,
                        ("GEN_ET_2", 0): 0.0,
                        ("GEN_ET_2", 1): 0.0,
                        ("GEN_ET_2", 2): 0.0,
                        ("Storage_ET1", 0): 11.0,
                        ("Storage_ET1", 1): 11.0,
                        ("Storage_ET1", 2): 11.0,
                    },
                    "opex": {
                        ("GEN_ET_1", 0): 1000.0,
                        ("GEN_ET_1", 1): 2000.0,
                        ("GEN_ET_1", 2): 3000.0,
                        ("GEN_ET_2", 0): np.nan,
                        ("GEN_ET_2", 1): np.nan,
                        ("GEN_ET_2", 2): np.nan,
                        ("Storage_ET1", 0): 0.0,
                        ("Storage_ET1", 1): 0.0,
                        ("Storage_ET1", 2): 0.0,
                    },
                }
            ).rename_axis(index=["Network element name", "Year"]),
            id="only_gen1_stor1",
        ),
        pytest.param(
            {},
            {},
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            None,
            pd.DataFrame(
                {
                    "opex": {
                        ("GEN_ET_1", 0): 1000.0,
                        ("GEN_ET_1", 1): 2000.0,
                        ("GEN_ET_1", 2): 3000.0,
                        ("GEN_ET_2", 0): 1000.0,
                        ("GEN_ET_2", 1): 2000.0,
                        ("GEN_ET_2", 2): 3000.0,
                        ("Storage_ET1", 0): 25.0,
                        ("Storage_ET1", 1): 25.0,
                        ("Storage_ET1", 2): 25.0,
                    },
                }
            ).rename_axis(index=["Network element name", "Year"]),
            id="both_capex_result_empty",
        ),
    ],
)
def test_get_local_capex_opex(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    generators: list[str],
    storages: list[str],
    filter_names: list[str] | None,
    expected_df: pd.DataFrame,
    capex_gen: dict[str, pd.DataFrame],
    capex_stor: dict[str, pd.DataFrame],
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._capa_cost_label = (
        GeneratorCapacityCostLabel.brutto
    )
    capex_opex_mocked_source_parameters._year_sample = None
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year},
        "local_capex": capex_gen,
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year},
        "local_capex": capex_stor,
    }
    with patch.object(
        capex_opex_mocked_source_parameters,
        "_filter_elements",
        return_value=(generators, storages),
    ), patch.object(
        capex_opex_mocked_source_parameters,
        "_get_global_generators_and_storage",
        return_value=([], []),
    ):
        result = capex_opex_mocked_source_parameters.get_local_capex_opex(
            "aggr", filter_names
        )
        assert_frame_equal(result, expected_df, check_names=False)


@pytest.mark.parametrize(
    "gen_capex_df, stor_capex_df, generators, storages, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [150.0, 224.0, 312.0],
                    "gen2": [123.0, 120.0, 320.0],
                    "gen3": [10.0, 10.0, 20.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [12.0, 12.0, 12.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    ("gen1", 0): [150.0, 1000.0],
                    ("gen1", 1): [224.0, 2000.0],
                    ("gen1", 2): [312.0, 3000.0],
                    ("gen2", 0): [123.0, 1000.0],
                    ("gen2", 1): [120.0, 2000.0],
                    ("gen2", 2): [320.0, 3000.0],
                    ("gen3", 0): [10.0, 0.0],
                    ("gen3", 1): [10.0, 0.0],
                    ("gen3", 2): [20.0, 0.0],
                    ("stor1", 0): [0.0, 0.0],
                    ("stor1", 1): [0.0, 0.0],
                    ("stor1", 2): [0.0, 0.0],
                    ("stor2", 0): [12.0, 25.0],
                    ("stor2", 1): [12.0, 25.0],
                    ("stor2", 2): [12.0, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="all_gens_and_storages",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [150.0, 224.0, 312.0],
                    "gen2": [123.0, 120.0, 320.0],
                    "gen3": [10.0, 10.0, 20.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(),
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    ("gen1", 0): [150.0, 1000.0],
                    ("gen1", 1): [224.0, 2000.0],
                    ("gen1", 2): [312.0, 3000.0],
                    ("gen2", 0): [123.0, 1000.0],
                    ("gen2", 1): [120.0, 2000.0],
                    ("gen2", 2): [320.0, 3000.0],
                    ("gen3", 0): [10.0, 0.0],
                    ("gen3", 1): [10.0, 0.0],
                    ("gen3", 2): [20.0, 0.0],
                    ("stor1", 0): [np.nan, 0.0],
                    ("stor1", 1): [np.nan, 0.0],
                    ("stor1", 2): [np.nan, 0.0],
                    ("stor2", 0): [np.nan, 25.0],
                    ("stor2", 1): [np.nan, 25.0],
                    ("stor2", 2): [np.nan, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="no_storage_results",
        ),
        pytest.param(
            pd.DataFrame(),
            pd.DataFrame(),
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            pd.DataFrame(
                {
                    ("gen1", 0): [1000.0],
                    ("gen1", 1): [2000.0],
                    ("gen1", 2): [3000.0],
                    ("gen2", 0): [1000.0],
                    ("gen2", 1): [2000.0],
                    ("gen2", 2): [3000.0],
                    ("gen3", 0): [0.0],
                    ("gen3", 1): [0.0],
                    ("gen3", 2): [0.0],
                    ("stor1", 0): [0.0],
                    ("stor1", 1): [0.0],
                    ("stor1", 2): [0.0],
                    ("stor2", 0): [25.0],
                    ("stor2", 1): [25.0],
                    ("stor2", 2): [25.0],
                },
                index=pd.Index(["opex"]),
            ),
            id="both_no_results",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "gen1": [150.0, 224.0, 312.0],
                    "gen2": [123.0, 120.0, 320.0],
                    "gen3": [10.0, 10.0, 20.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            pd.DataFrame(
                {
                    "stor1": [0.0, 0.0, 0.0],
                    "stor2": [12.0, 12.0, 12.0],
                },
                index=pd.Index([0, 1, 2], name="Year"),
            ),
            ["gen2"],
            ["stor2"],
            pd.DataFrame(
                {
                    ("gen2", 0): [123.0, 1000.0],
                    ("gen2", 1): [120.0, 2000.0],
                    ("gen2", 2): [320.0, 3000.0],
                    ("stor2", 0): [12.0, 25.0],
                    ("stor2", 1): [12.0, 25.0],
                    ("stor2", 2): [12.0, 25.0],
                },
                index=pd.Index(["capex", "opex"]),
            ),
            id="gen2_stor2_filtered",
        ),
    ],
)
def test__get_global_capex_opex(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    generators: list[str],
    storages: list[str],
    gen_capex_df: pd.DataFrame,
    stor_capex_df: pd.DataFrame,
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._capa_cost_label = (
        GeneratorCapacityCostLabel.brutto
    )
    capex_opex_mocked_source_parameters._year_sample = None
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year},
        "global_capex": {"global_capex": gen_capex_df},
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year},
        "global_capex": {"global_capex": stor_capex_df},
    }
    result = capex_opex_mocked_source_parameters._get_global_capex_opex(
        generators, storages
    )
    assert_frame_equal(result, expected_df, check_names=False)


@pytest.mark.parametrize(
    "generators, storages, level, expected_df",
    [
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "element",
            pd.DataFrame(
                {
                    "capex": {
                        ("gen1", 0): 100.0,
                        ("gen1", 1): 100.0,
                        ("gen1", 2): 100.0,
                        ("gen2", 0): 0.0,
                        ("gen2", 1): 0.0,
                        ("gen2", 2): 0.0,
                        ("gen3", 0): 200.0,
                        ("gen3", 1): 200.0,
                        ("gen3", 2): 200.0,
                        ("stor1", 0): 10.0,
                        ("stor1", 1): 10.0,
                        ("stor1", 2): 10.0,
                        ("stor2", 0): 20.0,
                        ("stor2", 1): 20.0,
                        ("stor2", 2): 20.0,
                    },
                    "opex": {
                        ("gen1", 0): 1000.0,
                        ("gen1", 1): 2000.0,
                        ("gen1", 2): 3000.0,
                        ("gen2", 0): 1000.0,
                        ("gen2", 1): 2000.0,
                        ("gen2", 2): 3000.0,
                        ("gen3", 0): 0.0,
                        ("gen3", 1): 0.0,
                        ("gen3", 2): 0.0,
                        ("stor1", 0): 0.0,
                        ("stor1", 1): 0.0,
                        ("stor1", 2): 0.0,
                        ("stor2", 0): 25.0,
                        ("stor2", 1): 25.0,
                        ("stor2", 2): 25.0,
                    },
                }
            ).rename_axis(index=["Network element name", "Year"]),
            id="all_gens_and_storages_element",
        ),
        pytest.param(
            ["gen1", "gen2", "gen3"],
            ["stor1", "stor2"],
            "type",
            pd.DataFrame(
                {
                    "capex": {
                        ("GEN_ET_1", 0): 300.0,
                        ("GEN_ET_1", 1): 300.0,
                        ("GEN_ET_1", 2): 300.0,
                        ("GEN_ET_2", 0): 0.0,
                        ("GEN_ET_2", 1): 0.0,
                        ("GEN_ET_2", 2): 0.0,
                        ("Storage_ET1", 0): 30.0,
                        ("Storage_ET1", 1): 30.0,
                        ("Storage_ET1", 2): 30.0,
                    },
                    "opex": {
                        ("GEN_ET_1", 0): 1000.0,
                        ("GEN_ET_1", 1): 2000.0,
                        ("GEN_ET_1", 2): 3000.0,
                        ("GEN_ET_2", 0): 1000.0,
                        ("GEN_ET_2", 1): 2000.0,
                        ("GEN_ET_2", 2): 3000.0,
                        ("Storage_ET1", 0): 25.0,
                        ("Storage_ET1", 1): 25.0,
                        ("Storage_ET1", 2): 25.0,
                    },
                }
            ).rename_axis(index=["Network element name", "Year"]),
            id="all_gens_and_storages_type",
        ),
        pytest.param(
            [
                "gen2",
            ],
            ["stor1"],
            "element",
            pd.DataFrame(
                {
                    "capex": {
                        ("gen2", 0): 0.0,
                        ("gen2", 1): 0.0,
                        ("gen2", 2): 0.0,
                        ("stor1", 0): 10.0,
                        ("stor1", 1): 10.0,
                        ("stor1", 2): 10.0,
                    },
                    "opex": {
                        ("gen2", 0): 1000.0,
                        ("gen2", 1): 2000.0,
                        ("gen2", 2): 3000.0,
                        ("stor1", 0): 0.0,
                        ("stor1", 1): 0.0,
                        ("stor1", 2): 0.0,
                    },
                }
            ).rename_axis(index=["Network element name", "Year"]),
            id="gen2_stor1_filtered",
        ),
    ],
)
def test_get_global_capex_opex(
    capex_opex_mocked_source_parameters: SourceParametersOverYearsQuery,
    level: Literal["type", "element"],
    generators: list[str],
    storages: list[str],
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
    expected_df: pd.DataFrame,
) -> None:
    capex_opex_mocked_source_parameters._capa_cost_label = (
        GeneratorCapacityCostLabel.brutto
    )
    gen_capex_df = pd.DataFrame(
        {
            "gen1": [100.0, 100.0, 100.0],
            "gen2": [0.0, 0.0, 0.0],
            "gen3": [200.0, 200.0, 200.0],
        },
        index=pd.Index([0, 1, 2], name="Year"),
    )

    stor_capex_df = pd.DataFrame(
        {
            "stor1": [10.0, 10.0, 10.0],
            "stor2": [20.0, 20.0, 20.0],
        },
        index=pd.Index([0, 1, 2], name="Year"),
    )

    capex_opex_mocked_source_parameters._year_sample = None
    capex_opex_mocked_source_parameters._generator_results = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year},
        "global_capex": {"global_capex": gen_capex_df},
    }
    capex_opex_mocked_source_parameters._storage_results = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year},
        "global_capex": {"global_capex": stor_capex_df},
    }
    with patch.object(
        capex_opex_mocked_source_parameters,
        "_get_global_generators_and_storage",
        return_value=(generators, storages),
    ):
        result = capex_opex_mocked_source_parameters.get_global_capex_opex(level)
        assert_frame_equal(result, expected_df, check_names=False)
