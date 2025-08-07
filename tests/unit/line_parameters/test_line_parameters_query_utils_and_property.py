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

import pandas as pd
import pytest

from zefir_analytics._engine.data_queries.line_parameters_over_years import (
    LineParametersOverYearsQuery,
)


def test_property_line_names_mocked_object(
    mocked_line_parameters_over_year_query: LineParametersOverYearsQuery,
) -> None:
    expected_lines_names = ["line_1", "line_2", "line_3"]
    assert mocked_line_parameters_over_year_query.line_names == expected_lines_names


@pytest.mark.parametrize(
    "results",
    [
        pytest.param({"line_1": pd.DataFrame()}, id="one_line"),
        pytest.param(
            {
                "line_a": pd.DataFrame(),
                "line_b": pd.DataFrame(),
                "line_c": pd.DataFrame(),
                "line_d": pd.DataFrame(),
            },
            id="few_line",
        ),
        pytest.param(
            {},
            id="no_results",
        ),
    ],
)
def test_property_lines_names(
    mocked_line_parameters_over_year_query: LineParametersOverYearsQuery,
    results: dict[str, pd.DataFrame],
) -> None:
    mocked_line_parameters_over_year_query._line_results = {"flow": results}
    assert mocked_line_parameters_over_year_query.line_names == list(results.keys())
