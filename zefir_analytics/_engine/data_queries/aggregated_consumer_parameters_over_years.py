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
from pyzefir.model.network_elements import AggregatedConsumer

from zefir_analytics._engine.data_queries import utils as data_utils

FRACTION_RESULTS_KEY: Final[str] = "fraction"


class AggregatedConsumerParametersOverYearsQuery:

    def __init__(
        self,
        network: Network,
        fraction_results: dict[str, dict[str, pd.DataFrame]],
        years_binding: pd.Series | None = None,
    ) -> None:
        self._network = network
        self._fraction_results = fraction_results
        self._years_binding = years_binding

    def get_fractions(
        self,
        aggregated_consumers_names: list[str] | str | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        df_dict = data_utils.dict_filter(
            dictionary=dict(self._fraction_results[FRACTION_RESULTS_KEY]),
            keys=aggregated_consumers_names,
        ).copy()
        return data_utils.handle_n_sample_results(df_dict, self._years_binding)

    def get_n_consumers(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        n_consumers = (
            {key: value.n_consumers.rename("N_consumers") for key, value in res.items()}
            if isinstance(res, dict)
            else res.n_consumers.rename("N_consumers")
        )
        return data_utils.handle_n_sample_results(n_consumers, self._years_binding)

    def get_yearly_energy_usage(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        yearly_energy_usage = (
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
        return data_utils.handle_n_sample_results(
            yearly_energy_usage, self._years_binding
        )

    def get_total_yearly_energy_usage(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        total_yearly_energy_usage = (
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
        return data_utils.handle_n_sample_results(
            total_yearly_energy_usage, self._years_binding
        )

    def get_aggregate_parameters(
        self, aggregated_consumers_names: list[str] | str | None = None
    ) -> pd.DataFrame:
        res = data_utils.dict_filter(
            dictionary=dict(self._network.aggregated_consumers),
            keys=aggregated_consumers_names,
        )
        if isinstance(res, dict):
            dfs = [
                self._create_aggregate_parameters_dataframe(agg) for agg in res.values()
            ]
            return data_utils.handle_n_sample_results(
                pd.concat(dfs), self._years_binding, is_multiindex=True
            )
        else:
            return data_utils.handle_n_sample_results(
                self._create_aggregate_parameters_dataframe(res),
                self._years_binding,
                is_multiindex=True,
            )

    @staticmethod
    def _create_aggregate_parameters_dataframe(agg: AggregatedConsumer) -> pd.DataFrame:
        series = agg.n_consumers.rename("n_consumers")
        series.index.name = "Year"
        df = pd.DataFrame(series)
        df = df.assign(
            total_usable_area=(
                df["n_consumers"] * agg.average_area
                if agg.average_area is not None
                else df["n_consumers"]
            )
        )
        df["Aggregate Name"] = agg.name
        return df.set_index(["Aggregate Name", df.index])

    def get_aggregate_elements_type_attachments(
        self,
        aggregated_consumers_names: list[str] | str | None = None,
    ) -> pd.DataFrame:
        if aggregated_consumers_names is None:
            aggregated_consumers_names = list(self._network.aggregated_consumers.keys())

        if isinstance(aggregated_consumers_names, list):
            dfs: list[pd.DataFrame] = [
                self._get_single_aggregate_elements_type_attachments_dataframe(agg_name)
                for agg_name in aggregated_consumers_names
            ]
            return pd.concat(dfs).fillna(0)
        else:
            return self._get_single_aggregate_elements_type_attachments_dataframe(
                aggregated_consumers_names
            )

    def _get_single_aggregate_elements_type_attachments_dataframe(
        self, aggregate_name: str
    ) -> pd.DataFrame:
        agg = self._network.aggregated_consumers[aggregate_name]
        dfs: list[pd.DataFrame] = []
        for lbs_name in agg.available_stacks:
            lbs = self._network.local_balancing_stacks[lbs_name]
            type_set: set[str] = set()
            for buses in lbs.buses.values():
                type_set = type_set.union(
                    self._get_unique_element_type_from_buses(buses)
                )
            df = pd.DataFrame(
                {
                    "agg_name": aggregate_name,
                    "lbs_name": lbs_name,
                    "attached_tech": list(type_set),
                }
            )
            df = pd.pivot_table(
                df,
                index=["agg_name", "lbs_name"],
                columns="attached_tech",
                aggfunc="size",
                fill_value=0,
            )

            dfs.append(df)
        return pd.concat(dfs).fillna(0)

    def _get_unique_element_type_from_buses(self, buses: set[str]) -> set[str]:
        unique_types = set()
        for bus_name in buses:
            bus = self._network.buses[bus_name]
            for gen_name in bus.generators:
                unique_types.add(self._network.generators[gen_name].energy_source_type)
            for stor_name in bus.storages:
                unique_types.add(self._network.storages[stor_name].energy_source_type)
        return unique_types
