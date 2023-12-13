from pathlib import Path

import pandas as pd
from pyzefir.model.network import Network
from pyzefir.postprocessing.results_handler import GeneralResultDirectory

from zefir_analytics import _engine as _d


class ZefirEngine:
    def __init__(
        self,
        source_path: Path,
        result_path: Path,
        scenario_name: str,
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
        ) = self._load_input_data(
            source_path,
            result_path,
            scenario_name,
            parameter_path,
            discount_rate_path,
            year_sample_path,
            hour_sample_path,
        )

        self._scenario_name = scenario_name

        self._source_parameters_over_years = _d.SourceParametersOverYearsQuery(
            network=self.network,
            generator_results=self.result_dict[
                GeneralResultDirectory.GENERATORS_RESULTS
            ],
            storage_results=self.result_dict[GeneralResultDirectory.STORAGES_RESULTS],
            year_sample=self._params["year_sample"],
            discount_rate=self._params["discount_rate"],
        )

        self._line_parameters_over_years = _d.LineParametersOverYearsQuery(
            network=self.network,
            line_results=self.result_dict[GeneralResultDirectory.LINES_RESULTS],
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
        parameter_path: Path | None = None,
        discount_rate_path: Path | None = None,
        year_sample_path: Path | None = None,
        hour_sample_path: Path | None = None,
    ) -> tuple[
        dict[str, dict[str, pd.DataFrame]],
        Network,
        dict[str, dict[str, dict[str, pd.DataFrame]]],
        dict[str, pd.Series],
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
        )
