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


def test_aggregate_consumer_n_sample(zefir_engine_n_sampled: ZefirEngine) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.lbs_params.get_lbs_fraction(),
        ze.lbs_params.get_lbs_capacity(),
    ]
    assert_analytics_result(results)


@pytest.mark.parametrize(
    ("func_input"),
    [
        pytest.param(
            [
                "B2_I__ELECTRIC_HEATING_BASE",
                "B2_IV__BJ_K4",
            ],
            id="2 lbs",
        ),
        pytest.param("B5_IV__ELECTRIC_HEATING_BASE", id="1 lbs"),
    ],
)
def test_line_parameters_n_sample_for_chosen_lines(
    zefir_engine_n_sampled: ZefirEngine, func_input: str | list[str]
) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.lbs_params.get_lbs_fraction(func_input),
        ze.lbs_params.get_lbs_capacity(func_input),
    ]
    assert_analytics_result(results)
