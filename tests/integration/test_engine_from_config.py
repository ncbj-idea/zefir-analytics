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

import configparser
import tempfile
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from zefir_analytics.zefir_engine import ZefirEngine


@pytest.fixture
def config_parser(
    input_path: Path,
    results_path: Path,
    parameters_path: Path,
) -> configparser.ConfigParser:
    """Simple configuration file for pipeline test run."""
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "input": {
                "input_path": str(input_path),
                "input_format": "csv",
                "scenario": "scenario_1",
            },
            "output": {
                "output_path": str(results_path),
            },
            "parameters": {
                "hour_sample": str(parameters_path / "hour_sample.csv"),
                "year_sample": str(parameters_path / "year_sample.csv"),
                "discount_rate": str(parameters_path / "discount_rate.csv"),
            },
            "optimization": {
                "binary_fraction": False,
                "money_scale": 100.0,
                "ens": False,
                "use_hourly_scale": True,
            },
        }
    )
    return config


@pytest.fixture
def config_ini_path() -> Path:
    """Create *.ini file."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".ini", delete=False
    ) as temp_file:
        return Path(temp_file.name)


@pytest.fixture
def zefir_engine_from_config(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
) -> ZefirEngine:
    set_up_config_ini(config_ini_path, config_parser)
    ze = ZefirEngine.create_from_config(config_ini_path)
    return ze


def set_up_config_ini(path: Path, config_parser: configparser.ConfigParser) -> None:
    with open(path, mode="w") as file_handler:
        config_parser.write(file_handler)


def test_check_zefir_engines_creation_method(
    zefir_engine_from_config: ZefirEngine, zefir_engine: ZefirEngine
) -> None:
    zec = zefir_engine_from_config
    ze = zefir_engine
    ze_config_results = [
        ze.source_params.get_generation_sum(level="element"),
        ze.source_params.get_emission(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        ze.lbs_params.get_lbs_fraction("OS_HP"),
        ze.line_params.get_transmission_fee("DH -> MF_BASIC_H"),
    ]
    ze_config_results = [
        zec.source_params.get_generation_sum(level="element"),
        zec.source_params.get_emission(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        zec.lbs_params.get_lbs_fraction("OS_HP"),
        zec.line_params.get_transmission_fee("DH -> MF_BASIC_H"),
    ]

    for df_ze, df_zec in zip(ze_config_results, ze_config_results):
        assert_frame_equal(df_ze, df_zec)
