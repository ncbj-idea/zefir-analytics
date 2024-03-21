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
from typing import Self

import numpy as np
import pandas as pd
from pyzefir.model.network import Network
from pyzefir.optimization.opt_config import OptConfig
from pyzefir.postprocessing.results_handler import GeneralResultDirectory
from pyzefir.utils.config_parser import ConfigLoader

from zefir_analytics import _engine as _d


class ZefirEngine:
    def __init__(
        self,
        source_path: Path,
        result_path: Path,
        scenario_name: str,
        discount_rate: np.ndarray[float],
        year_sample: np.ndarray[int],
        hour_sample: np.ndarray[int],
        used_hourly_scale: bool,
    ) -> None:
        self._scenario_name = scenario_name
        self._year_sample = self._validate_parameter_array(
            year_sample, np.int64, "year_sample"
        )
        self._hour_sample = self._validate_parameter_array(
            hour_sample, np.int64, "hour_sample"
        )
        self._discount_rate = self._validate_parameter_array(
            discount_rate, np.float64, "discount_rate"
        )
        (
            self._source_dict,
            self._network,
            self._result_dict,
        ) = self._load_input_data(
            source_path=source_path,
            result_path=result_path,
            scenario_name=scenario_name,
        )
        self._opt_config = OptConfig(
            hours=self._network.constants.n_hours,
            years=self._network.constants.n_years,
            hour_sample=self._hour_sample,
            use_hourly_scale=used_hourly_scale,
        )

        self._source_parameters_over_years = _d.SourceParametersOverYearsQuery(
            network=self.network,
            generator_results=self.result_dict[
                GeneralResultDirectory.GENERATORS_RESULTS
            ],
            storage_results=self.result_dict[GeneralResultDirectory.STORAGES_RESULTS],
            year_sample=self._year_sample,
            discount_rate=self._discount_rate,
            hourly_scale=self._opt_config.hourly_scale,
            hour_sample=self._opt_config.hour_sample,
        )

        self._line_parameters_over_years = _d.LineParametersOverYearsQuery(
            network=self.network,
            line_results=self.result_dict[GeneralResultDirectory.LINES_RESULTS],
            hourly_scale=self._opt_config.hourly_scale,
            hour_sample=self._opt_config.hour_sample,
        )

        self._aggregated_consumer_parameters_over_years = (
            _d.AggregatedConsumerParametersOverYearsQuery(
                network=self.network,
                fraction_results=self.result_dict[
                    GeneralResultDirectory.FRACTIONS_RESULTS
                ],
            )
        )
        self._variability_of_lbs = _d.LbsParametersOverYearsQuery(
            network=self.network,
            fractions_results=self.result_dict[
                GeneralResultDirectory.FRACTIONS_RESULTS
            ],
            generator_results=self.result_dict[
                GeneralResultDirectory.GENERATORS_RESULTS
            ],
            storage_results=self.result_dict[GeneralResultDirectory.STORAGES_RESULTS],
        )

    @classmethod
    def create_from_config(cls, config_path: Path) -> Self:
        config = ConfigLoader(config_path).load()
        return cls(
            source_path=config.input_path,
            result_path=config.output_path,
            scenario_name=config.scenario,
            discount_rate=config.discount_rate,
            year_sample=config.year_sample,
            hour_sample=config.hour_sample,
            used_hourly_scale=config.use_hourly_scale,
        )

    @property
    def network(self) -> Network:
        return self._network

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @property
    def source_dict(self) -> dict[str, dict[str, pd.DataFrame]]:
        return self._source_dict

    @property
    def result_dict(self) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
        return self._result_dict

    @property
    def source_params(self) -> _d.SourceParametersOverYearsQuery:
        return self._source_parameters_over_years

    @property
    def line_params(self) -> _d.LineParametersOverYearsQuery:
        return self._line_parameters_over_years

    @property
    def aggregated_consumer_params(
        self,
    ) -> _d.AggregatedConsumerParametersOverYearsQuery:
        return self._aggregated_consumer_parameters_over_years

    @property
    def lbs_params(self) -> _d.LbsParametersOverYearsQuery:
        return self._variability_of_lbs

    @staticmethod
    def _load_input_data(
        source_path: Path,
        result_path: Path,
        scenario_name: str,
    ) -> tuple[
        dict[str, dict[str, pd.DataFrame]],
        Network,
        dict[str, dict[str, dict[str, pd.DataFrame]]],
    ]:
        return _d.DataLoader.load_data(
            source_path,
            result_path,
            scenario_name,
        )

    @staticmethod
    def _validate_parameter_array(
        array: np.ndarray, dtype: type, attr_name: str
    ) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Array {attr_name} must be a numpy array")
        if array.dtype != dtype:
            raise TypeError(
                f"Elements of the array {attr_name} must be of type {dtype.__name__}"
            )
        return array
