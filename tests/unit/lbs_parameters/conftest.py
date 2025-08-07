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

from zefir_analytics._engine.data_queries.lbs_parameters_over_years import (
    LbsParametersOverYearsQuery,
)


@pytest.fixture
def generator_results_per_gen_name_per_year() -> pd.DataFrame:
    return pd.DataFrame(
        {"gen1": [1.0, 2.0, 3.0], "gen2": [10.0, 20.0, 30.0], "gen3": [0.0, 0.0, 8.0]},
        index=pd.Index([0, 1, 2], name="Year"),
    )


@pytest.fixture
def storage_results_per_gen_name_per_year() -> pd.DataFrame:
    return pd.DataFrame(
        {"stor1": [10.0, 0.0, 0.0], "stor2": [5.0, 5.0, 5.0]},
        index=pd.Index([0, 1, 2], name="Year"),
    )


@pytest.fixture
def fraction_results_per_per_lbs() -> pd.DataFrame:
    return pd.DataFrame(
        {"lbs1": [10.0, 0.0, 0.0], "lbs2": [0.0, 0.0, 0.0]},
        index=pd.Index([0, 1, 2], name="Year"),
    )


@pytest.fixture
def mocked_lbs_parameters_over_year_query(
    generator_results_per_gen_name_per_year: pd.DataFrame,
    storage_results_per_gen_name_per_year: pd.DataFrame,
    fraction_results_per_per_lbs: pd.DataFrame,
) -> LbsParametersOverYearsQuery:
    network = MagicMock()
    fractions_results: dict[str, dict[str, pd.DataFrame]] = {
        "fraction": {
            "aggr1": fraction_results_per_per_lbs,
            "aggr2": fraction_results_per_per_lbs,
        }
    }

    generator_results: dict[str, dict[str, pd.DataFrame]] = {
        "capacity": {"capacity": generator_results_per_gen_name_per_year}
    }
    storage_results: dict[str, dict[str, pd.DataFrame]] = {
        "capacity": {"capacity": storage_results_per_gen_name_per_year}
    }
    years_binding = None

    return LbsParametersOverYearsQuery(
        network=network,
        fractions_results=fractions_results,
        generator_results=generator_results,
        storage_results=storage_results,
        years_binding=years_binding,
    )
