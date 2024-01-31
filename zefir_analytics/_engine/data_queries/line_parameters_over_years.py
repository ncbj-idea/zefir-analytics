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

import numpy as np
import pandas as pd
from pyzefir.model.network import Network

from zefir_analytics._engine.constants import YEARS_LABEL
from zefir_analytics._engine.data_queries import utils as data_utils


class LineParametersOverYearsQuery:
    def __init__(
        self,
        network: Network,
        line_results: dict[str, dict[str, pd.DataFrame]],
        hour_sample: np.ndarray,
        hourly_scale: float,
    ) -> None:
        self._network = network
        self._line_results = line_results
        self._hourly_scale = hourly_scale
        self._hour_sample = hour_sample

    @property
    def line_names(self) -> list[str]:
        return list(self._line_results["flow"].keys())

    @staticmethod
    def _get_yearly_summary(
        df: pd.DataFrame, column_name: str, operation: str
    ) -> pd.DataFrame:
        df = df.agg(axis=0, func=operation).to_frame(column_name)
        df.index.name = YEARS_LABEL
        df.index = df.index.astype(np.integer)
        return df

    def _get_flow(self, line_name: str) -> pd.DataFrame:
        return (
            self._get_yearly_summary(
                self._line_results["flow"][line_name], "Total energy volume", "sum"
            )
            * self._hourly_scale
        )

    def _get_transmission_fee(self, line_name: str) -> pd.DataFrame:
        df_flow = self._line_results["flow"][line_name]
        if tf_name := self._network.lines[line_name].transmission_fee:
            series_tf = (
                self._network.transmission_fees[tf_name]
                .fee.iloc[self._hour_sample]
                .reset_index(drop=True)
            )
            df = df_flow.mul(series_tf.values[:None], axis=0)
        else:
            df = df_flow * 0.0
        return (
            self._get_yearly_summary(df, "Transmission fee total cost", "sum")
            * self._hourly_scale
        )

    def get_flow(
        self, line_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return (
            data_utils.argument_condition(line_name, self._get_flow)
            if line_name is not None
            else data_utils.argument_condition(self.line_names, self._get_flow)
        )

    def get_capacity(
        self, line_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        raise NotImplementedError(
            "This method will be implemented later when basecode will be ready"
        )

    def get_transmission_fee(
        self, line_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return (
            data_utils.argument_condition(line_name, self._get_transmission_fee)
            if line_name is not None
            else data_utils.argument_condition(
                self.line_names, self._get_transmission_fee
            )
        )
