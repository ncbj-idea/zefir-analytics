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

from typing import Any

import pytest

from tests.integration.utils import assert_analytics_result
from zefir_analytics.zefir_engine import ZefirEngine


@pytest.mark.parametrize(
    ("source_params_condition"),
    [
        pytest.param({"level": "element"}, id="element_condition"),
        pytest.param({"level": "type"}, id="type_condition"),
        pytest.param(
            {"level": "type", "filter_type": "aggr", "filter_names": ["B2_I", "B5_II"]},
            id="filter_condition",
        ),
    ],
)
def test_source_parameters_n_sample_over_years(
    zefir_engine_n_sampled: ZefirEngine,
    source_params_condition: dict,
) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.source_params.get_generation_sum(**source_params_condition),
        ze.source_params.get_dump_energy_sum(**source_params_condition),
        ze.source_params.get_load_sum(**source_params_condition),
        ze.source_params.get_installed_capacity(**source_params_condition),
        ze.source_params.get_generation_demand(**source_params_condition),
        ze.source_params.get_fuel_usage(**source_params_condition),
        ze.source_params.get_emission(**source_params_condition),
        ze.source_params.get_fuel_cost(**source_params_condition),
        ze.source_params.get_network_costs_per_tech_type(**source_params_condition),
        ze.source_params.get_ets_cost(**source_params_condition),
        ze.source_params.get_state_of_charge(**source_params_condition),
    ]
    if "filter_type" not in source_params_condition:
        results.append(
            ze.source_params.get_global_capex_opex(**source_params_condition)
        )
    assert_analytics_result(results)


def test_source_parameters_n_sample_without_required_args(
    zefir_engine_n_sampled: ZefirEngine,
) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.source_params.get_local_capex_opex(),
        ze.source_params.get_network_fuel_cost(),
        ze.source_params.get_network_fuel_availability(),
        ze.source_params.get_ens(),
    ]
    assert_analytics_result(results)


def test_source_parameters_n_sample_over_years_filter_empty_aggr(
    zefir_engine_n_sampled: ZefirEngine,
) -> None:
    ze = zefir_engine_n_sampled
    setup: dict[str, Any] = {
        "level": "type",
        "filter_type": "aggr",
        "filter_names": ["TARYFA_B"],
    }
    zefir_results = [
        ze.source_params.get_generation_sum(**setup),
        ze.source_params.get_dump_energy_sum(**setup),
        ze.source_params.get_load_sum(**setup),
        ze.source_params.get_installed_capacity(**setup),
        ze.source_params.get_generation_demand(**setup),
        ze.source_params.get_fuel_usage(**setup),
        ze.source_params.get_emission(**setup),
        ze.source_params.get_fuel_cost(**setup),
        ze.source_params.get_network_costs_per_tech_type(**setup),
        ze.source_params.get_ets_cost(**setup),
        ze.source_params.get_state_of_charge(**setup),
    ]
    assert len(zefir_results)


def test_source_parameters_n_sample_over_years_hourly_resolution(
    zefir_engine_n_sampled: ZefirEngine,
) -> None:
    ze = zefir_engine_n_sampled
    results = [
        ze.source_params.get_generation_sum(level="type", is_hours_resolution=True),
        ze.source_params.get_dump_energy_sum(level="type", is_hours_resolution=True),
        ze.source_params.get_load_sum(level="type", is_hours_resolution=True),
        ze.source_params.get_generation_demand(level="type", is_hours_resolution=True),
        ze.source_params.get_fuel_usage(level="type", is_hours_resolution=True),
        ze.source_params.get_ens(is_hours_resolution=True),
        ze.source_params.get_state_of_charge(level="type", is_hours_resolution=True),
    ]
    assert_analytics_result(results)
