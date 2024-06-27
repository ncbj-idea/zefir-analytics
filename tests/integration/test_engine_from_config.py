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

from tests.integration.conftest import get_paths_for_config
from zefir_analytics.zefir_engine import ZefirEngine


@pytest.fixture
def config_parser() -> configparser.ConfigParser:
    """Simple configuration file for pipeline test run."""
    input_path, results_path, parameters_path = get_paths_for_config("simple-data-case")
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
                "use_hourly_scale": True,
                "generator_capacity_cost": "brutto",
            },
        }
    )
    return config


@pytest.fixture
def xlsx_config_parser() -> configparser.ConfigParser:
    """Simple configuration file for pipeline test run."""
    input_path, results_path, parameters_path = get_paths_for_config("simple-data-case")
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "input": {
                "input_path": str(input_path),
                "input_format": "xlsx",
                "scenario": "scenario_1",
            },
            "output": {
                "output_path": str(results_path),
                "csv_dump_path": str(input_path),
            },
            "parameters": {
                "hour_sample": str(parameters_path / "hour_sample.csv"),
                "year_sample": str(parameters_path / "year_sample.csv"),
                "discount_rate": str(parameters_path / "discount_rate.csv"),
            },
            "optimization": {
                "binary_fraction": False,
                "money_scale": 100.0,
                "use_hourly_scale": True,
                "generator_capacity_cost": "brutto",
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
def zefir_engine_from_config_csv(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
) -> ZefirEngine:
    set_up_config_ini(config_ini_path, config_parser)
    ze = ZefirEngine.create_from_config(config_ini_path)
    return ze


@pytest.fixture
def zefir_engine_from_config_xlsx(
    config_ini_path: Path,
    xlsx_config_parser: configparser.ConfigParser,
) -> ZefirEngine:
    set_up_config_ini(config_ini_path, xlsx_config_parser)
    ze = ZefirEngine.create_from_config(config_ini_path)
    return ze


def set_up_config_ini(path: Path, config_parser: configparser.ConfigParser) -> None:
    with open(path, mode="w") as file_handler:
        config_parser.write(file_handler)


def test_check_zefir_engines_creation_method(
    zefir_engine_from_config_csv: ZefirEngine,
    zefir_engine: ZefirEngine,
    zefir_engine_from_config_xlsx: ZefirEngine,
) -> None:
    zec = zefir_engine_from_config_csv
    zee = zefir_engine_from_config_xlsx
    ze = zefir_engine
    ze_config_results = [
        ze.source_params.get_generation_sum(level="element"),
        ze.source_params.get_emission(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        ze.lbs_params.get_lbs_fraction("OS_HP"),
        ze.line_params.get_transmission_fee("DH -> MF_BASIC_H"),
    ]
    zec_config_results = [
        zec.source_params.get_generation_sum(level="element"),
        zec.source_params.get_emission(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        zec.lbs_params.get_lbs_fraction("OS_HP"),
        zec.line_params.get_transmission_fee("DH -> MF_BASIC_H"),
    ]
    assert (
        ze.objective_function_value
        == zec.objective_function_value
        == zee.objective_function_value
    )
    zee_config_results = [
        zee.source_params.get_generation_sum(level="element"),
        zee.source_params.get_emission(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        zee.lbs_params.get_lbs_fraction("OS_HP"),
        zee.line_params.get_transmission_fee("DH -> MF_BASIC_H"),
    ]

    for df_ze, df_zec, df_zee in zip(
        ze_config_results, zec_config_results, zee_config_results
    ):
        assert_frame_equal(df_ze, df_zec)
        assert_frame_equal(df_ze, df_zee)
        assert_frame_equal(df_zec, df_zee)
