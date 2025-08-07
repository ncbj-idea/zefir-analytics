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

from zefir_analytics._engine.data_queries.aggregated_consumer_parameters_over_years import (
    AggregatedConsumerParametersOverYearsQuery,
)


@pytest.fixture
def fraction_results_per_per_lbs() -> pd.DataFrame:
    return pd.DataFrame(
        {"lbs1": [10.0, 0.0, 0.0], "lbs2": [5.0, 5.0, 5.0]},
        index=pd.Index([0, 1, 2], name="Year"),
    )


def create_mocked_network() -> MagicMock:
    gens = {
        "gen1": MagicMock(energy_source_type="generator_type_1"),
        "gen2": MagicMock(energy_source_type="generator_type_2"),
        "gen3": MagicMock(energy_source_type="generator_type_3"),
    }
    stors = {
        "stor1": MagicMock(energy_source_type="storage_type_1"),
        "stor2": MagicMock(energy_source_type="storage_type_2"),
        "stor3": MagicMock(energy_source_type="storage_type_1"),
    }

    bus1 = MagicMock(generators={"gen1", "gen2"}, storages={"stor1", "stor2"})
    bus1.name = "bus1"

    bus2 = MagicMock(generators={"gen3"}, storages={"stor3"})
    bus2.name = "bus2"

    buses = {"bus1": bus1, "bus2": bus2}

    local_balancing_stacks = {
        "lkt1": MagicMock(buses={"energy_source_1": {"bus1"}}),
        "lkt2": MagicMock(buses={"energy_source_2": {"bus2"}}),
    }
    aggr1 = MagicMock(
        available_stacks={"lkt1"},
        n_consumers=pd.Series([10, 10, 10]),
        average_area=2,
        yearly_energy_usage={
            "ET1": pd.Series([2, 2, 2]),
            "ET2": pd.Series([10, 10, 10]),
        },
    )
    aggr1.name = "aggr1"

    aggr2 = MagicMock(
        available_stacks={"lkt2"},
        n_consumers=pd.Series([100, 100, 100]),
        average_area=2,
        yearly_energy_usage={
            "ET1": pd.Series([2, 2, 2]),
            "ET3": pd.Series([100, 100, 100]),
        },
    )
    aggr2.name = "aggr2"

    aggregated_consumers = {"aggr1": aggr1, "aggr2": aggr2}

    return MagicMock(
        aggregated_consumers=aggregated_consumers,
        local_balancing_stacks=local_balancing_stacks,
        buses=buses,
        generators=gens,
        storages=stors,
    )


@pytest.fixture
def mocked_aggr_parameters_over_year_query(
    fraction_results_per_per_lbs: pd.DataFrame,
) -> AggregatedConsumerParametersOverYearsQuery:
    network = create_mocked_network()
    fraction_results: dict[str, dict[str, pd.DataFrame]] = {
        "fraction": {
            "aggr1": fraction_results_per_per_lbs,
            "aggr2": fraction_results_per_per_lbs,
        }
    }
    years_binding = None
    return AggregatedConsumerParametersOverYearsQuery(
        network=network,
        fraction_results=fraction_results,
        years_binding=years_binding,
    )
