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
from pyzefir.optimization.opt_config import OptConfig
from pyzefir.postprocessing.results_handler import GeneralResultDirectory
from pyzefir.utils.config_parser import ConfigParams

from zefir_analytics import _engine as _d


class ZefirEngine:
    def __init__(
        self,
        source_path: Path,
        result_path: Path,
        scenario_name: str,
        config_path: Path,
        parameter_path: Path | None = None,
        discount_rate_path: Path | None = None,
        year_sample_path: Path | None = None,
        hour_sample_path: Path | None = None,
    ) -> None:
        (
            self._source_dict,
            self._network,
            self._result_dict,
            self._params,
            self._config,
        ) = self._load_input_data(
            source_path=source_path,
            result_path=result_path,
            scenario_name=scenario_name,
            config_path=config_path,
            parameter_path=parameter_path,
            discount_rate_path=discount_rate_path,
            year_sample_path=year_sample_path,
            hour_sample_path=hour_sample_path,
        )

        self._scenario_name = scenario_name
        self._opt_config = OptConfig(
            hours=self._network.constants.n_hours,
            years=self._network.constants.n_years,
            hour_sample=self._params["hour_sample"].values,
        )

        self._source_parameters_over_years = _d.SourceParametersOverYearsQuery(
            network=self.network,
            generator_results=self.result_dict[
                GeneralResultDirectory.GENERATORS_RESULTS
            ],
            storage_results=self.result_dict[GeneralResultDirectory.STORAGES_RESULTS],
            year_sample=self._params["year_sample"],
            discount_rate=self._params["discount_rate"],
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
        config_path: Path,
        parameter_path: Path | None = None,
        discount_rate_path: Path | None = None,
        year_sample_path: Path | None = None,
        hour_sample_path: Path | None = None,
    ) -> tuple[
        dict[str, dict[str, pd.DataFrame]],
        Network,
        dict[str, dict[str, dict[str, pd.DataFrame]]],
        dict[str, pd.Series],
        ConfigParams,
    ]:
        parameters_path = _d.ParametersPath(
            parameter_path,
            discount_rate_path,
            year_sample_path,
            hour_sample_path,
        )
        return _d.DataLoader.load_data(
            source_path,
            result_path,
            scenario_name,
            parameters_path,
            config_path,
        )
