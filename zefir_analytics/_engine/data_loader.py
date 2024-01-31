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

from dataclasses import InitVar, dataclass, field, fields
from enum import StrEnum, unique
from pathlib import Path

import pandas as pd
from pyzefir.model.network import Network
from pyzefir.parser.csv_parser import CsvParser
from pyzefir.parser.network_creator import NetworkCreator
from pyzefir.postprocessing.results_handler import GeneralResultDirectory
from pyzefir.utils.config_parser import ConfigLoader, ConfigParams
from pyzefir.utils.path_manager import CsvPathManager


class MissingParametersFileException(Exception):
    pass


@unique
class ParameterNames(StrEnum):
    discount_rate = "discount_rate"
    hour_sample = "hour_sample"
    year_sample = "year_sample"


@dataclass(frozen=True)
class ParametersPath:
    discount_rate: Path = field(init=False)
    hour_sample: Path = field(init=False)
    year_sample: Path = field(init=False)
    parameters_path: InitVar[Path | None] = None
    discount_rate_path: InitVar[Path | None] = None
    hour_sample_path: InitVar[Path | None] = None
    year_sample_path: InitVar[Path | None] = None

    @staticmethod
    def _path_or_default(
        parameters_dir: Path | None,
        parameter_path: Path | None,
        parameter_name: ParameterNames,
    ) -> Path:
        if parameter_path is not None:
            if parameter_path.exists():
                return parameter_path
            raise MissingParametersFileException(
                f"Parameters file {parameter_name} not found in given path {parameter_path}"
            )
        if parameters_dir is not None:
            path = parameters_dir / f"{parameter_name}.csv"
            if path.exists():
                return path
            raise MissingParametersFileException(
                f"Parameters file {parameter_name} not found in default path {path}"
            )
        raise MissingParametersFileException(
            f"Parameters file {parameter_name} not found. "
            f"Either specify direct path to the file or pass path to parameters directory"
        )

    def __post_init__(
        self,
        parameters_path: Path | None,
        discount_rate_path: Path | None,
        hour_sample_path: Path | None,
        year_sample_path: Path | None,
    ) -> None:
        discount_rate = self._path_or_default(
            parameters_path, discount_rate_path, ParameterNames.discount_rate
        )
        hour_sample = self._path_or_default(
            parameters_path, hour_sample_path, ParameterNames.hour_sample
        )
        year_sample = self._path_or_default(
            parameters_path, year_sample_path, ParameterNames.year_sample
        )

        # bypassing dataclass param frozen=True
        object.__setattr__(self, "discount_rate", discount_rate)
        object.__setattr__(self, "hour_sample", hour_sample)
        object.__setattr__(self, "year_sample", year_sample)


class DataLoader:
    @classmethod
    def load_data(
        cls,
        source_path: Path,
        result_path: Path,
        scenario_name: str,
        parameters_path: ParametersPath,
        config_path: Path,
    ) -> tuple[
        dict[str, dict[str, pd.DataFrame]],
        Network,
        dict[str, dict[str, dict[str, pd.DataFrame]]],
        dict[str, pd.Series],
        ConfigParams,
    ]:
        source_data = cls._load_source_data(source_path, scenario_name)
        network = cls._create_network(source_data)
        result_data = cls._load_result_data(result_path)
        parameters = cls._load_parameters(parameters_path)
        config = cls._load_config(config_path)

        return source_data, network, result_data, parameters, config

    @staticmethod
    def _load_config(config_path: Path) -> ConfigParams:
        config = ConfigLoader(config_path).load()
        return config

    @staticmethod
    def _load_source_data(
        source_path: Path,
        scenario_name: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return CsvParser(
            path_manager=CsvPathManager(
                dir_path=source_path,
                scenario_name=scenario_name,
            )
        ).load_dfs()

    @staticmethod
    def _create_network(df_dict: dict[str, dict[str, pd.DataFrame]]) -> Network:
        return NetworkCreator.create(df_dict)

    @classmethod
    def _load_result_data(
        cls,
        result_path: Path,
    ) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
        result_dict: dict[str, dict[str, dict[str, pd.DataFrame]]] = dict()
        for group in GeneralResultDirectory:
            group_str = group.value  # noqa
            group_path = result_path / group_str
            result_dict[group_str] = dict()
            for data_category in group_path.glob("*"):
                result_dict[group_str][data_category.stem] = dict()
                for csv_file in data_category.glob("*.csv"):
                    df = pd.read_csv(csv_file, index_col=0)
                    if group_str == GeneralResultDirectory.LINES_RESULTS:
                        result_dict[group_str][data_category.stem][
                            csv_file.stem.replace("-", "->")
                        ] = df
                    else:
                        result_dict[group_str][data_category.stem][csv_file.stem] = df
        return result_dict

    @classmethod
    def _load_parameters(cls, parameters_path: ParametersPath) -> dict[str, pd.Series]:
        parameter_dict = dict()
        for parameter_field in fields(parameters_path):
            parameter_name = parameter_field.name
            parameter_filepath = getattr(parameters_path, parameter_name)
            parameter_dict[parameter_name] = pd.read_csv(
                parameter_filepath, header=None
            ).squeeze("columns")

        return parameter_dict
