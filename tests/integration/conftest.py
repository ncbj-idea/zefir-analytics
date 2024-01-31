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

import configparser
import tempfile
from pathlib import Path

import pytest

from tests.utils import get_resources
from zefir_analytics import ZefirEngine

data = get_resources("simple-data-case")
parameters_path = data / "parameters"
input_path = data / "source_csv"


@pytest.fixture
def output_path() -> Path:
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def config_parser(output_path: Path) -> configparser.ConfigParser:
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
                "output_path": str(output_path),
                "sol_dump_path": str(output_path / "file.sol"),
                "opt_logs_path": str(output_path / "file.log"),
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
def config_ini_path(config_parser: configparser.ConfigParser) -> Path:
    """Create *.ini file."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".ini", delete=False
    ) as temp_file:
        yield Path(temp_file.name)


def set_up_config_ini(path: Path, config_parser: configparser.ConfigParser) -> None:
    with open(path, mode="w") as file_handler:
        config_parser.write(file_handler)


@pytest.fixture
def zefir_engine(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
) -> ZefirEngine:
    set_up_config_ini(config_ini_path, config_parser)
    ze = ZefirEngine(
        source_path=input_path,
        result_path=data / "results",
        scenario_name="scenario_1",
        parameter_path=data / "parameters",
        config_path=config_ini_path,
    )
    yield ze
