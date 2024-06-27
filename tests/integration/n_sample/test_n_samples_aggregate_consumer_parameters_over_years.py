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
        ze.aggregated_consumer_params.get_fractions(),
        ze.aggregated_consumer_params.get_n_consumers(),
        ze.aggregated_consumer_params.get_yearly_energy_usage(),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage(),
        ze.aggregated_consumer_params.get_aggregate_parameters(),
        ze.aggregated_consumer_params.get_aggregate_elements_type_attachments(),
    ]
    assert_analytics_result(results)


@pytest.mark.parametrize(
    ("func_input"),
    [
        pytest.param(["B2_II", "B6_I"], id="2 aggr"),
        pytest.param("B5_II", id="1 aggr"),
    ],
)
def test_aggregate_consumer_n_sample_for_chosen_aggr(
    zefir_engine_n_sampled: ZefirEngine, func_input: str | list[str]
) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.aggregated_consumer_params.get_fractions(func_input),
        ze.aggregated_consumer_params.get_n_consumers(func_input),
        ze.aggregated_consumer_params.get_yearly_energy_usage(func_input),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage(func_input),
        ze.aggregated_consumer_params.get_aggregate_parameters(func_input),
        ze.aggregated_consumer_params.get_aggregate_elements_type_attachments(
            func_input
        ),
    ]
    assert_analytics_result(results)
