# ZefirAnalytics
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
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
from pandas.testing import assert_frame_equal

from tests.integration.utils import assert_analytics_result
from zefir_analytics.zefir_engine import ZefirEngine


@pytest.fixture
def line_names() -> list[str | list[str]]:
    return [
        "DH -> MF_BASIC_H",
        "DH -> OS_BASIC_H",
        ["DH -> MF_BASIC_H", "DH -> OS_BASIC_H"],
        ["KSE -> MF_GAS_EE", "KSE -> MF_HP_EE"],
    ]


@pytest.fixture
def lines_without_tf() -> list[str]:
    return ["KSE -> SF_GAS_EE", "KSE -> SF_HP_EE"]


def test_variability_of_lines_parameters(
    zefir_engine: ZefirEngine,
    line_names: list[str | list[str]],
) -> None:
    ze = zefir_engine
    for name in line_names:
        assert_analytics_result([ze.line_params.get_flow(name)])
        assert_analytics_result([ze.line_params.get_transmission_fee(name)])


def test_cost_without_transmission_fee(
    zefir_engine: ZefirEngine, lines_without_tf: list[str]
) -> None:
    ze = zefir_engine
    results: list[pd.DataFrame] = []  # no comprehension coz of return type
    for line_name in lines_without_tf:
        results.append(ze.line_params.get_transmission_fee(line_name))

    for df in results:
        df_zero = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        df.equals(df_zero)


@pytest.mark.parametrize(
    ("df", "operation", "expected_df"),
    [
        pytest.param(
            pd.DataFrame({1: [1, 1], 3: [2, 2], 7: [3, 3]}),
            "sum",
            pd.DataFrame({"test": [2, 4, 6]}, index=pd.Index([1, 3, 7], name="Year")),
            id="df_sum_values",
        ),
        pytest.param(
            pd.DataFrame({1: [1, 3], 3: [2, 2], 7: [5, 5]}),
            "mean",
            pd.DataFrame(
                {"test": [2.0, 2.0, 5.0]}, index=pd.Index([1, 3, 7], name="Year")
            ),
            id="df_mean_values",
        ),
    ],
)
def test_get_yearly_summary(
    zefir_engine: ZefirEngine,
    df: pd.DataFrame,
    expected_df: pd.DataFrame,
    operation: str,
) -> None:
    ze = zefir_engine
    result = ze.line_params._get_yearly_summary(df, "test", operation)
    assert_frame_equal(result, expected_df)


def test_line_parameters_none(
    zefir_engine: ZefirEngine,
) -> None:
    ze = zefir_engine
    assert_analytics_result([ze.line_params.get_flow()])
    assert_analytics_result([ze.line_params.get_transmission_fee()])


def test_get_flow_lines_over_hours(
    zefir_engine: ZefirEngine,
    line_names: list[str | list[str]],
) -> None:
    ze = zefir_engine
    zefir_results: list[pd.DataFrame] = []
    for name in line_names:
        zefir_results.append(
            ze.line_params.get_flow(line_name=name, is_hours_resolution=True)
        )
    zefir_results.append(ze.line_params.get_flow(is_hours_resolution=True))
    zefir_results = [
        value
        for element in zefir_results
        if isinstance(element, dict)
        for value in element.values()
    ]
    assert len(zefir_results)
    assert all(not df.empty for df in zefir_results)
    assert all("Hour" in index for df in zefir_results for index in [df.index.names])
