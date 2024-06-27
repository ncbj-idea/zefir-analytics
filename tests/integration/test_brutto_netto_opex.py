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

from tests.integration.conftest import get_paths_and_data_for_engine
from zefir_analytics.zefir_engine import ZefirEngine


def test_opex_brutto_netto() -> None:
    input_path, results_path, year_sample, hour_sample, discount_rate = (
        get_paths_and_data_for_engine("simple-data-case")
    )
    ze_brutto = ZefirEngine(
        source_path=input_path,
        result_path=results_path / "csv",
        scenario_name="scenario_1",
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        used_hourly_scale=True,
        generator_capacity_cost="brutto",
    )
    ze_netto = ZefirEngine(
        source_path=input_path,
        result_path=results_path / "csv",
        scenario_name="scenario_1",
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        used_hourly_scale=True,
        generator_capacity_cost="netto",
    )

    opex_brutto_global = ze_brutto.source_params.get_global_capex_opex(level="type")[
        "opex"
    ]
    opex_netto_global = ze_netto.source_params.get_global_capex_opex(level="type")[
        "opex"
    ]

    opex_brutto_local = ze_brutto.source_params.get_local_capex_opex()["opex"]
    opex_netto_local = ze_netto.source_params.get_local_capex_opex()["opex"]

    for brutto, netto in zip(
        [opex_brutto_global, opex_brutto_local], [opex_netto_global, opex_netto_local]
    ):
        assert len(brutto) == len(netto)
        assert sorted(brutto.index) == sorted(netto.index)
        assert pytest.approx(brutto.values) != netto.values

    for gen_et, year in opex_netto_global.index:
        if gen_et not in ze_netto._network.generator_types:
            continue
        eff = ze_netto._network.generator_types[gen_et].efficiency.iloc[:, 0].mean()
        opex_netto_global[gen_et][year] = opex_netto_global[gen_et][year] / eff
    assert pytest.approx(opex_netto_global.values) == opex_brutto_global.values


@pytest.mark.parametrize(
    ("generator_capacity_cost", "error_msg"),
    [
        pytest.param(
            "semi-brutto",
            "Invalid generator_capacity_cost: semi-brutto. Must be one of ['brutto', 'netto']",
            id="semi-brutto",
        ),
        pytest.param(
            "neto",
            "Invalid generator_capacity_cost: neto. Must be one of ['brutto', 'netto']",
            id="neto",
        ),
        pytest.param(
            "NO",
            "Invalid generator_capacity_cost: NO. Must be one of ['brutto', 'netto']",
            id="NO",
        ),
    ],
)
def test_generator_capacity_cost_create_ze_wrong_cost_label(
    generator_capacity_cost: str, error_msg: str
) -> None:
    input_path, results_path, year_sample, hour_sample, discount_rate = (
        get_paths_and_data_for_engine("simple-data-case")
    )
    with pytest.raises(ValueError) as exc_info:
        ZefirEngine(
            source_path=input_path,
            result_path=results_path / "csv",
            scenario_name="scenario_1",
            year_sample=year_sample,
            hour_sample=hour_sample,
            discount_rate=discount_rate,
            used_hourly_scale=True,
            generator_capacity_cost=generator_capacity_cost,
        )
    assert exc_info.value.args[0] == error_msg


def test_generator_capacity_cost_create_ze_property() -> None:
    input_path, results_path, year_sample, hour_sample, discount_rate = (
        get_paths_and_data_for_engine("simple-data-case")
    )

    ze = ZefirEngine(
        source_path=input_path,
        result_path=results_path / "csv",
        scenario_name="scenario_1",
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        used_hourly_scale=True,
        generator_capacity_cost="brutto",
    )
    assert ze.generator_capacity_cost
    assert ze.generator_capacity_cost == "brutto"
