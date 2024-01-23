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

import numpy as np
import pytest

from tests.integration.utils import assert_analytics_result
from zefir_analytics.zefir_engine import ZefirEngine


@pytest.fixture
def lbs_names() -> list[str | list[str]]:
    return [
        "MF_BASIC",
        "OS_HP",
        "SF_GAS",
        ["MF_BASIC", "MF_GAS", "MF_HP"],
        ["OS_BASIC", "OS_HP"],
    ]


def test_variability_of_local_balancing_stacks(
    zefir_engine: ZefirEngine,
    lbs_names: list[str | list[str]],
) -> None:
    for name in lbs_names:
        assert_analytics_result([zefir_engine.lbs_params.get_lbs_fraction(name)])
        assert_analytics_result([zefir_engine.lbs_params.get_lbs_capacity(name)])


def test_variability_of_local_balancing_stacks_none(zefir_engine: ZefirEngine) -> None:
    assert_analytics_result([zefir_engine.lbs_params.get_lbs_fraction()])
    assert_analytics_result([zefir_engine.lbs_params.get_lbs_capacity()])


@pytest.mark.parametrize(
    ("lbs_name", "expected_gens", "expected_stors"),
    [
        pytest.param(
            "MF_BASIC", ["PV_MF_BASIC"], ["EE_STOR_MF_BASIC"], id="lbs_name_MF_BASIC"
        ),
        pytest.param(
            "SF_GAS",
            ["PV_SF_GAS", "BOILER_GAS_SF_GAS"],
            ["EE_STOR_SF_GAS"],
            id="lbs_name_SF_GAS",
        ),
        pytest.param(
            "OS_HP",
            ["HEAT_PUMP_OS_HP", "PV_OS_HP", "HEAT_PUMP_OS_HP"],
            ["EE_STOR_OS_HP", "H_STOR_OS_HP"],
            id="lbs_name_OS_HP",
        ),
    ],
)
def test_get_attached_sources(
    zefir_engine: ZefirEngine,
    lbs_name: str,
    expected_gens: list[str],
    expected_stors: list[str],
) -> None:
    gens, stors = zefir_engine.lbs_params._get_attached_sources(lbs_name)

    assert sorted(gens) == sorted(expected_gens)
    assert sorted(stors) == sorted(expected_stors)


@pytest.mark.parametrize(
    (
        "lbs_name",
        "expected_array",
    ),
    [
        pytest.param(
            "MF_BASIC", np.array([24000.0, 27755.55, 30000.0]), id="lbs_name_MF_BASIC"
        ),
        pytest.param("SF_GAS", np.array([400.0, 0.0, 0.0]), id="lbs_name_SF_GAS"),
        pytest.param("OS_HP", np.array([2000.0, 2000.0, 12000.0]), id="lbs_name_OS_HP"),
    ],
)
def test_get_fraction_factor(
    zefir_engine: ZefirEngine,
    lbs_name: str,
    expected_array: np.ndarray,
) -> None:
    result = zefir_engine.lbs_params._get_fraction_factor(lbs_name)

    assert np.allclose(result, expected_array)
