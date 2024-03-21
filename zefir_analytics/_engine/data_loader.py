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
    ]:
        source_data = cls._load_source_data(source_path, scenario_name)
        network = cls._create_network(source_data)
        result_data = cls._load_result_data(result_path)

        return source_data, network, result_data

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
