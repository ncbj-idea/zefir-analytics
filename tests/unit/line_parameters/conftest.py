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

from zefir_analytics._engine.data_queries.line_parameters_over_years import (
    LineParametersOverYearsQuery,
)


@pytest.fixture
def flow_results_per_year_per_hour() -> dict[str, pd.DataFrame]:
    return {
        "line_1": pd.DataFrame(
            {
                0: [10.0, 10.0, 10.0],
                1: [10.0, 10.0, 10.0],
                2: [10.0, 10.0, 10.0],
            },
            index=pd.Index([0, 1, 2], name="Hour"),
        ),
        "line_2": pd.DataFrame(
            {
                0: [0.0, 5.0, 5.0],
                1: [0.0, 5.0, 5.0],
                2: [0.0, 5.0, 5.0],
            },
            index=pd.Index([0, 1, 2], name="Hour"),
        ),
        "line_3": pd.DataFrame(
            {
                0: [2.0, 0.0, 0.0],
                1: [2.0, 0.0, 0.0],
                2: [2.0, 0.0, 0.0],
            },
            index=pd.Index([0, 1, 2], name="Hour"),
        ),
    }


@pytest.fixture
def mocked_line_parameters_over_year_query(
    flow_results_per_year_per_hour: dict[str, pd.DataFrame]
) -> LineParametersOverYearsQuery:
    network = MagicMock()
    line_results: dict[str, dict[str, pd.DataFrame]] = {
        "flow": flow_results_per_year_per_hour
    }
    hourly_scale = 1.0
    hour_sample = np.array([])
    years_binding = None

    return LineParametersOverYearsQuery(
        network=network,
        line_results=line_results,
        hourly_scale=hourly_scale,
        years_binding=years_binding,
        hour_sample=hour_sample,
    )
