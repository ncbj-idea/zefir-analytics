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

import numpy as np
import pandas as pd
import pytest

from zefir_analytics._engine.data_queries.source_parameters_over_years import (
    SourceParametersOverYearsQuery,
)


@pytest.fixture
def mocked_source_parameters_over_years_query() -> SourceParametersOverYearsQuery:
    network = MagicMock()
    generator_results: dict[str, dict[str, pd.DataFrame]] = {}
    storage_results: dict[str, dict[str, pd.DataFrame]] = {}
    bus_results: dict[str, dict[str, pd.DataFrame]] = {}
    year_sample = pd.Series()
    discount_rate = pd.Series()
    hourly_scale = 1.0
    hour_sample = np.array([])
    generator_capacity_cost_label = MagicMock()
    years_binding = None

    with patch.object(
        SourceParametersOverYearsQuery,
        "_create_energy_source_type_mapping",
        return_value=dict(),
    ), patch.object(
        SourceParametersOverYearsQuery,
        "_calculate_cap_plus",
        return_value=(pd.DataFrame(), pd.DataFrame()),
    ):
        return SourceParametersOverYearsQuery(
            network=network,
            generator_results=generator_results,
            storage_results=storage_results,
            bus_results=bus_results,
            year_sample=year_sample,
            discount_rate=discount_rate,
            hourly_scale=hourly_scale,
            hour_sample=hour_sample,
            generator_capacity_cost_label=generator_capacity_cost_label,
            years_binding=years_binding,
        )


@pytest.fixture
def generator_results_per_et_per_year_per_hour() -> dict[str, pd.DataFrame]:
    return {
        "gen1": pd.DataFrame(
            {
                "Energy Type": ["ET1", "ET1", "ET2", "ET2"],
                0: [10.0, 10.0, 10.0, 10.0],
                1: [0.0, 0.0, 0.0, 0.0],
                2: [15.0, 15.0, 15.0, 15.0],
            },
            index=pd.Index([0, 1, 0, 1], name="Hour"),
        ),
        "gen2": pd.DataFrame(
            {
                "Energy Type": ["ET1", "ET1", "ET2", "ET2"],
                0: [10.0, 10.0, 10.0, 10.0],
                1: [11.0, 11.0, 11.0, 11.0],
                2: [12.0, 12.0, 12.0, 12.0],
            },
            index=pd.Index([0, 1, 0, 1], name="Hour"),
        ),
        "gen3": pd.DataFrame(
            {
                "Energy Type": ["ET1", "ET1", "ET2", "ET2"],
                0: [0.0, 0.0, 0.0, 0.0],
                1: [0.0, 0.0, 0.0, 0.0],
                2: [0.0, 0.0, 0.0, 0.0],
            },
            index=pd.Index([0, 1, 0, 1], name="Hour"),
        ),
    }


@pytest.fixture
def generator_results_per_year_per_hour() -> dict[str, pd.DataFrame]:
    return {
        "gen1": pd.DataFrame(
            {
                0: [10.0, 10.0],
                1: [10.0, 10.0],
                2: [10.0, 10.0],
            },
            index=pd.Index([0, 1], name="Hour"),
        ),
        "gen2": pd.DataFrame(
            {
                0: [5.0, 5.0],
                1: [5.0, 5.0],
                2: [5.0, 5.0],
            },
            index=pd.Index([0, 1], name="Hour"),
        ),
        "gen3": pd.DataFrame(
            {
                0: [0.0, 0.0],
                1: [0.0, 0.0],
                2: [0.0, 0.0],
            },
            index=pd.Index([0, 1], name="Hour"),
        ),
    }


@pytest.fixture
def storage_results_per_year_per_hour() -> dict[str, pd.DataFrame]:
    return {
        "stor1": pd.DataFrame(
            {
                0: [0.0, 0.0],
                1: [0.0, 0.0],
                2: [2.0, 2.0],
            },
            index=pd.Index([0, 1], name="Hour"),
        ),
        "stor2": pd.DataFrame(
            {
                0: [0.0, 0.0],
                1: [3.0, 3.0],
                2: [7.0, 7.0],
            },
            index=pd.Index([0, 1], name="Hour"),
        ),
    }


@pytest.fixture
def generator_results_per_gen_name_per_year() -> pd.DataFrame:
    return pd.DataFrame(
        {"gen1": [1.0, 2.0, 3.0], "gen2": [10.0, 20.0, 30.0], "gen3": [0.0, 0.0, 0.0]},
        index=pd.Index([0, 1, 2], name="Year"),
    )


@pytest.fixture
def storage_results_per_gen_name_per_year() -> pd.DataFrame:
    return pd.DataFrame(
        {"stor1": [0.0, 0.0, 0.0], "stor2": [5.0, 5.0, 5.0]},
        index=pd.Index([0, 1, 2], name="Year"),
    )
