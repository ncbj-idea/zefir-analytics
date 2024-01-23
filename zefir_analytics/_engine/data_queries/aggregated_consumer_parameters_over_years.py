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

from typing import Final

import pandas as pd
from pyzefir.model.network import Network

from zefir_analytics._engine.data_queries import utils as data_utils

FRACTION_RESULTS_KEY: Final[str] = "fraction"


class AggregatedConsumerParametersOverYearsQuery:
    def __init__(
        self,
        network: Network,
        fraction_results: dict[str, dict[str, pd.DataFrame]],
    ) -> None:
        self._network = network
        self._fraction_results = fraction_results

    def get_fractions(
        self,
        aggregated_consumers_names: list[str] | str | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return data_utils.dict_filter(
            dictionary=dict(self._fraction_results[FRACTION_RESULTS_KEY]),
            keys=aggregated_consumers_names,
        ).copy()

    def get_n_consumers(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        return (
            {key: value.n_consumers.rename("N_consumers") for key, value in res.items()}
            if isinstance(res, dict)
            else res.n_consumers.rename("N_consumers")
        )

    def get_yearly_energy_usage(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        return (
            {
                key: pd.DataFrame(
                    {col: values for col, values in value.yearly_energy_usage.items()}
                )
                for key, value in res.items()
            }
            if isinstance(res, dict)
            else pd.DataFrame(
                {col: values for col, values in res.yearly_energy_usage.items()}
            )
        )

    def get_total_yearly_energy_usage(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        return (
            {
                key: pd.DataFrame(
                    {
                        col: values * value.n_consumers
                        for col, values in value.yearly_energy_usage.items()
                    }
                )
                for key, value in res.items()
            }
            if isinstance(res, dict)
            else pd.DataFrame(
                {
                    col: values * res.n_consumers
                    for col, values in res.yearly_energy_usage.items()
                }
            )
        )
