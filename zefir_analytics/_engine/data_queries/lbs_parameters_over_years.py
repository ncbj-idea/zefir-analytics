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

from itertools import chain

import numpy as np
import pandas as pd
from pyzefir.model.network import Network

from zefir_analytics._engine.data_queries import utils as data_utils


class LbsParametersOverYearsQuery:
    def __init__(
        self,
        network: Network,
        fractions_results: dict[str, dict[str, pd.DataFrame]],
        generator_results: dict[str, dict[str, pd.DataFrame]],
        storage_results: dict[str, dict[str, pd.DataFrame]],
    ) -> None:
        self._network = network
        self._fractions_results = fractions_results
        self._generator_results = generator_results
        self._storage_results = storage_results

    @property
    def lbs_names(self) -> list[str]:
        lbs_names = {
            col
            for df in self._fractions_results["fraction"].values()
            for col in df.columns
        }
        return list(lbs_names)

    def _get_lbs_fraction(self, lbs_name: str) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                key: df[lbs_name]
                for key, df in self._fractions_results["fraction"].items()
            }
        )
        df_filtered = df.loc[:, (df != 0.0).any()]
        if df_filtered.empty:
            for agg in df.columns:
                if lbs_name in self._network.aggregated_consumers[agg].available_stacks:
                    return pd.DataFrame({agg: np.zeros(len(df.index))}, index=df.index)
        return df_filtered

    def _get_lbs_capacity(self, lbs_name: str) -> pd.DataFrame:
        gen_attach, storage_attach = self._get_attached_sources(lbs_name)
        fraction_factor = self._get_fraction_factor(lbs_name)
        df_gen = self._generator_results["capacity"]["capacity"][gen_attach]
        df_stor = self._storage_results["capacity"]["capacity"][storage_attach]
        df_gen = df_gen.div(fraction_factor, axis=0).fillna(0.0).replace(np.inf, 0.0)
        df_stor = df_stor.div(fraction_factor, axis=0).fillna(0.0).replace(np.inf, 0.0)
        return pd.concat([df_gen, df_stor], axis=1)

    def _get_attached_sources(self, lbs_name: str) -> tuple[list[str], list[str]]:
        buses_attached = [
            bus
            for buses in self._network.local_balancing_stacks[lbs_name].buses.values()
            for bus in buses
        ]

        generators = list(
            chain.from_iterable(
                self._network.buses[bus_name].generators for bus_name in buses_attached
            )
        )

        storages = list(
            chain.from_iterable(
                self._network.buses[bus_name].storages for bus_name in buses_attached
            )
        )

        return generators, storages

    def _get_fraction_factor(self, lbs_name: str) -> np.ndarray:
        df_fraction = self._get_lbs_fraction(lbs_name)
        df_fraction = df_fraction.loc[:, (df_fraction != 0.0).any()]
        if df_fraction.empty:
            return np.zeros(len(df_fraction.index))
        n_con_series = self._network.aggregated_consumers[
            list(df_fraction.columns).pop()
        ].n_consumers
        return (df_fraction.squeeze() * n_con_series[df_fraction.index]).to_numpy()

    def get_lbs_fraction(
        self, lbs_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return (
            data_utils.argument_condition(lbs_name, self._get_lbs_fraction)
            if lbs_name is not None
            else data_utils.argument_condition(self.lbs_names, self._get_lbs_fraction)
        )

    def get_lbs_capacity(
        self, lbs_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return (
            data_utils.argument_condition(lbs_name, self._get_lbs_capacity)
            if lbs_name is not None
            else data_utils.argument_condition(self.lbs_names, self._get_lbs_capacity)
        )
