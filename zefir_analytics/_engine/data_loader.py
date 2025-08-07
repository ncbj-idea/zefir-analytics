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


from pathlib import Path

import pandas as pd
from pyzefir.model.network import Network
from pyzefir.parser.csv_parser import CsvParser
from pyzefir.parser.network_creator import NetworkCreator
from pyzefir.postprocessing.results_handler import GeneralResultDirectory
from pyzefir.utils.path_manager import CsvPathManager

from zefir_analytics._engine.constants import OBJECTIVE_FUNCTION_FILE_NAME


class MissingParametersFileException(Exception):
    pass


class DataLoader:
    @classmethod
    def load_data(
        cls,
        source_path: Path,
        result_path: Path,
        scenario_name: str,
    ) -> tuple[
        dict[str, dict[str, pd.DataFrame]],
        Network,
        dict[str, dict[str, dict[str, pd.DataFrame]]],
        float,
    ]:
        source_data = cls._load_source_data(source_path, scenario_name)
        network = cls._create_network(source_data)
        if result_path.name == "feather":
            result_data = cls._load_feather_result_data(result_path)
            objective_func_value = cls._load_objective_function_from_feather(
                result_path
            )
        else:
            result_data = cls._load_csv_result_data(result_path)
            objective_func_value = cls._load_objective_function_from_csv(result_path)

        return source_data, network, result_data, objective_func_value

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
    def _load_objective_function_from_csv(cls, result_path: Path) -> float:
        return pd.read_csv(
            result_path / f"{OBJECTIVE_FUNCTION_FILE_NAME}.csv", index_col=0
        ).squeeze()

    @classmethod
    def _load_objective_function_from_feather(cls, result_path: Path) -> float:
        df = pd.read_feather(result_path / f"{OBJECTIVE_FUNCTION_FILE_NAME}.feather")
        return df.set_index("index").squeeze()

    @classmethod
    def _load_csv_result_data(
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
                    objective_func_df = pd.read_csv(csv_file, index_col=0)
                    if group_str == GeneralResultDirectory.LINES_RESULTS:
                        result_dict[group_str][data_category.stem][
                            csv_file.stem.replace("-", "->")
                        ] = objective_func_df
                    else:
                        result_dict[group_str][data_category.stem][
                            csv_file.stem
                        ] = objective_func_df
        return result_dict

    @classmethod
    def _load_feather_result_data(
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
                for feather_file in data_category.glob("*.feather"):
                    objective_func_df = pd.read_feather(feather_file)
                    df = objective_func_df.set_index(objective_func_df.columns[0])
                    if group_str == GeneralResultDirectory.LINES_RESULTS:
                        result_dict[group_str][data_category.stem][
                            feather_file.stem.replace("-", "->")
                        ] = df
                    else:
                        result_dict[group_str][data_category.stem][
                            feather_file.stem
                        ] = df
        return result_dict
