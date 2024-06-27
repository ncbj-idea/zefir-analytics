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


import pytest

from tests.integration.utils import assert_analytics_result
from zefir_analytics.zefir_engine import ZefirEngine


def test_line_parameters_n_sample(zefir_engine_n_sampled: ZefirEngine) -> None:
    ze = zefir_engine_n_sampled
    results = [ze.line_params.get_flow(), ze.line_params.get_transmission_fee()]
    assert_analytics_result(results)


@pytest.mark.parametrize(
    ("func_input"),
    [
        pytest.param(
            [
                "KSE -> B2_I__HEAT_PUMP_BASE__EE_OUTER__EE_OUTER",
                "HEATING_SUBSYSTEM -> B2_II__CENTRAL_HEATING_BASE__HEAT_OUTER__HEAT_OUTER",
            ],
            id="2 lines",
        ),
        pytest.param("KSE -> B2_III__HEAT_PUMP_BASE__EE_OUTER__EE_OUTER", id="1 line"),
    ],
)
def test_line_parameters_n_sample_for_chosen_lines(
    zefir_engine_n_sampled: ZefirEngine, func_input: str | list[str]
) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.line_params.get_flow(func_input),
        ze.line_params.get_transmission_fee(func_input),
    ]
    assert_analytics_result(results)
