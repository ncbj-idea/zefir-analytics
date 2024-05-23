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

from tests.integration.utils import assert_analytics_result
from zefir_analytics import ZefirEngine


def test_aggregated_consumer_params_over_years(zefir_engine: ZefirEngine) -> None:
    ze = zefir_engine
    zefir_results_dataframe_res = [
        ze.aggregated_consumer_params.get_aggregate_elements_type_attachments(),
        ze.aggregated_consumer_params.get_aggregate_parameters(),
        ze.aggregated_consumer_params.get_aggregate_elements_type_attachments(
            ["MULTI_FAMILY", "SINGLE_FAMILY"]
        ),
        ze.aggregated_consumer_params.get_aggregate_parameters(
            ["MULTI_FAMILY", "SINGLE_FAMILY"]
        ),
        ze.aggregated_consumer_params.get_aggregate_elements_type_attachments(
            "MULTI_FAMILY"
        ),
        ze.aggregated_consumer_params.get_aggregate_parameters("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_fractions("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_n_consumers("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_yearly_energy_usage("SINGLE_FAMILY"),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_fractions(),
        ze.aggregated_consumer_params.get_n_consumers(),
        ze.aggregated_consumer_params.get_yearly_energy_usage(),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage(),
        ze.aggregated_consumer_params.get_fractions(["MULTI_FAMILY"]),
        ze.aggregated_consumer_params.get_n_consumers(
            ["MULTI_FAMILY", "SINGLE_FAMILY"]
        ),
        ze.aggregated_consumer_params.get_yearly_energy_usage(["MULTI_FAMILY"]),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage(
            ["MULTI_FAMILY", "SINGLE_FAMILY"]
        ),
    ]

    assert_analytics_result(
        zefir_results_dataframe_res,
    )
