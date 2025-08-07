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

from collections import defaultdict
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from pyzefir.model.network import Network

from zefir_analytics._engine.constants import (
    COST_TYPE_LABEL,
    EMISSION_LABEL,
    ENERGY_TYPE_LABEL,
    FUEL_LABEL,
    HOUR_LABEL,
    NETWORK_ELEMENT_NAME,
    NETWORK_ELEMENT_TYPE,
    YEARS_LABEL,
)
from zefir_analytics._engine.data_queries.utils import (
    GeneratorCapacityCostLabel,
    assign_multiindex,
    get_generators_emission_types,
    handle_n_sample_results,
)

source_parameter_levels = Literal["element", "type"]
source_parameter_filter_types = Literal["bus", "stack", "aggr"]


class SourceParametersOverYearsQuery:

    def __init__(
        self,
        network: Network,
        generator_results: dict[str, dict[str, pd.DataFrame]],
        storage_results: dict[str, dict[str, pd.DataFrame]],
        bus_results: dict[str, dict[str, pd.DataFrame]],
        year_sample: pd.Series,
        discount_rate: pd.Series,
        hourly_scale: float,
        hour_sample: np.ndarray,
        generator_capacity_cost_label: GeneratorCapacityCostLabel,
        years_binding: pd.Series | None = None,
    ):
        self._network = network
        self._generator_results = generator_results
        self._storage_results = storage_results
        self._bus_results = bus_results
        self._year_sample = year_sample
        self._discount_rate = discount_rate
        self._energy_source_type_mapping = self._create_energy_source_type_mapping()
        (
            self._generator_capacity_plus,
            self._storage_capacity_plus,
        ) = self._calculate_cap_plus()
        self._hourly_scale = hourly_scale
        self._hour_sample = hour_sample
        self._capa_cost_label = generator_capacity_cost_label
        self._years_binding = years_binding

    def _get_generator_and_storage_from_set_of_buses(
        self,
        bus_set: set[str],
    ) -> tuple[list[str], list[str]]:
        generators = [
            g.name
            for g in self._network.generators.values()
            if g.buses.intersection(bus_set)
        ]
        storages = [s.name for s in self._network.storages.values() if s.bus in bus_set]
        return generators, storages

    def _get_global_generators_and_storage(
        self,
    ) -> tuple[list[str], list[str]]:
        all_buses_out = {
            item
            for stack in self._network.local_balancing_stacks.values()
            for item in list(stack.buses_out.values())
        }
        all_global_buses = set(self._network.buses).difference(all_buses_out)
        return self._get_generator_and_storage_from_set_of_buses(all_global_buses)

    def _get_generator_et_results(
        self,
        results_group: str,
        generators: list[str],
        is_hours_resolution: bool,
    ) -> pd.DataFrame:
        if not generators:
            return pd.DataFrame()
        generator_dfs = self._generator_results[results_group]
        generator_dfs = {
            gen_name: (
                df.pivot(columns=ENERGY_TYPE_LABEL).stack()
                if is_hours_resolution
                else df.pivot(columns=ENERGY_TYPE_LABEL).stack().groupby(level=1).sum()
            )
            for gen_name, df in generator_dfs.items()
            if gen_name in generators
        }
        return pd.concat(generator_dfs, axis=1)

    def _get_storage_results(
        self,
        results_group: str,
        storages: list[str],
        is_hours_resolution: bool,
    ) -> pd.DataFrame:
        if not storages:
            return pd.DataFrame()
        storage_dfs = self._storage_results[results_group]

        storage_dfs = {
            storage_name: df if is_hours_resolution else df.sum()
            for storage_name, df in storage_dfs.items()
            if storage_name in storages
        }
        for storage_name, df in storage_dfs.items():
            storage_type = self._network.storages[storage_name].energy_source_type
            energy_type = self._network.storage_types[storage_type].energy_type
            if is_hours_resolution:
                df = df.set_index(df.index.map(lambda x: (x, energy_type)))
                df.index.names = [HOUR_LABEL, ENERGY_TYPE_LABEL]
            else:
                df = df.to_frame().T
                df.index = [energy_type]
            storage_dfs[storage_name] = df
        return pd.concat(storage_dfs, axis=1).fillna(0.0)

    def _create_energy_source_type_mapping(self) -> dict[str, set[str]]:
        mapping = defaultdict(set)
        for gen in self._network.generators.values():
            mapping[gen.energy_source_type].add(gen.name)

        for stor in self._network.storages.values():
            mapping[stor.energy_source_type].add(stor.name)

        return dict(mapping)

    def _aggregate_energy_sources(
        self,
        energy_source_df: pd.DataFrame,
        index_name: str,
        level: source_parameter_levels,
        column_name: str | None = None,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
        is_binding_skip: bool = False,
    ) -> pd.DataFrame:
        if energy_source_df.empty:
            return energy_source_df
        level_name = NETWORK_ELEMENT_NAME
        if level == "type":
            energy_source_df = self._aggregate_by_type(
                energy_source_df, self._energy_source_type_mapping
            )
            level_name = NETWORK_ELEMENT_TYPE

        column_names = (
            [level_name, column_name] if column_name is not None else level_name
        )
        energy_source_df = energy_source_df.rename_axis(
            columns=column_names,
            index=[HOUR_LABEL, index_name] if is_hours_resolution else index_name,
        )

        if filter_type is None and filter_names is not None:
            filtered_columns = energy_source_df.columns.get_level_values(0).isin(
                filter_names
            )
            energy_source_df = energy_source_df.loc[:, filtered_columns]

        energy_source_df = energy_source_df.T.sort_index()

        energy_source_df.index = energy_source_df.index.set_levels(
            energy_source_df.index.get_level_values(1).astype(int),
            level=1,
            verify_integrity=False,
        )
        if self._years_binding is not None and not is_binding_skip:
            energy_source_df = handle_n_sample_results(
                energy_source_df, self._years_binding, is_multiindex=True
            )

        if is_hours_resolution:
            return energy_source_df.stack(level=0).fillna(0.0)

        return energy_source_df

    @staticmethod
    def _aggregate_by_type(
        df: pd.DataFrame, mapping: dict[str, set[str]]
    ) -> pd.DataFrame:
        result_dfs = []
        for gen_type, gen_names in mapping.items():
            columns = list(gen_names.intersection(df.columns.get_level_values(level=0)))
            if not columns:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                type_df = df[columns].groupby(level=1, axis=1).sum()
                multiindex = pd.MultiIndex.from_tuples(
                    [(gen_type, col) for col in type_df.columns]
                )
                type_df.columns = multiindex
            else:
                type_df = df[columns].sum(axis=1).to_frame(gen_type)
            result_dfs.append(type_df)

        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        return pd.DataFrame()

    @staticmethod
    def _flatten_2d_list(data: list[list]) -> list:
        return [ele for d in data for ele in d]

    def _filter_elements(  # noqa: R901
        self,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_bus_filter: bool = False,
    ) -> tuple[list[str], ...]:
        generators = list(self._network.generators.keys())
        storages = list(self._network.storages.keys())

        if filter_type is None or filter_names is None:
            if is_bus_filter:
                return generators, storages, list(self._network.buses.keys())
            return generators, storages

        available_buses, available_stacks = set(filter_names), set(filter_names)
        if filter_type == "aggr":
            available_stacks_2d: list[list[str]] = [
                list(aggr.stack_base_fraction.keys())
                for aggr in self._network.aggregated_consumers.values()
                if aggr.name in filter_names
            ]
            available_stacks = set(self._flatten_2d_list(available_stacks_2d))
        if filter_type in ["stack", "aggr"]:
            available_buses_2d: list[list[str]] = [
                list(stack.buses_out.values())
                for stack in self._network.local_balancing_stacks.values()
                if stack.name in available_stacks
            ]
            available_buses = set(self._flatten_2d_list(available_buses_2d))

        generators, storages = self._get_generator_and_storage_from_set_of_buses(
            available_buses
        )
        if is_bus_filter:
            return generators, storages, list(available_buses)
        return generators, storages

    def _get_generation_sum(
        self, generators: list[str], storages: list[str], is_hours_resolution: bool
    ) -> pd.DataFrame:
        generator_df = self._get_generator_et_results(
            "generation_per_energy_type", generators, is_hours_resolution
        )
        storage_df = self._get_storage_results(
            "generation", storages, is_hours_resolution
        )
        generation_sum_df = pd.concat([generator_df, storage_df], axis=1)
        if not is_hours_resolution:
            return generation_sum_df * self._hourly_scale
        return generation_sum_df

    def get_generation_sum(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_generation_sum(generators, storages, is_hours_resolution)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def get_dump_energy_sum(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        generators, _ = self._filter_elements(filter_type, filter_names)
        df = self._get_generator_et_results(
            "dump_energy_per_energy_type", generators, is_hours_resolution
        )
        if not is_hours_resolution:
            df *= self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def get_load_sum(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        _, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_storage_results("load", storages, is_hours_resolution)
        if not is_hours_resolution:
            df *= self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def _get_installed_capacity(
        self, generators: list[str], storages: list[str]
    ) -> pd.DataFrame:
        generator_df = self._generator_results["capacity"]["capacity"]
        storage_df = self._storage_results["capacity"]["capacity"]
        generator_df = generator_df.loc[:, generator_df.columns.isin(generators)]
        storage_df = storage_df.loc[:, storage_df.columns.isin(storages)]
        return pd.concat([generator_df, storage_df], axis=1)

    def get_installed_capacity(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = (
            self._get_installed_capacity(generators, storages)
            .stack()
            .swaplevel()
            .to_frame()
            .T
        )
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _get_generation_demand(
        self,
        generators: list[str],
        _storages: list[str],
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        if not generators:
            return pd.DataFrame()
        generator_dfs = self._generator_results["generation"]
        dfs = []
        for generator, generation_df in generator_dfs.items():
            if generator not in generators:
                continue
            gen_type_name = self._network.generators[generator].energy_source_type
            conversion_rates = self._network.generator_types[
                gen_type_name
            ].conversion_rate
            if not conversion_rates:
                continue
            dem_df = self._calculate_demand_for_generator(
                conversion_rates, generation_df, is_hours_resolution, self._hour_sample
            )
            multiindex = pd.MultiIndex.from_tuples(
                [(generator, col) for col in dem_df.columns]
            )
            dem_df.columns = multiindex
            if not is_hours_resolution:
                dem_df = dem_df.stack().swaplevel().unstack()
            dfs.append(dem_df)
        return pd.concat(dfs, axis=1)

    @staticmethod
    def _calculate_demand_for_generator(
        conversion_rates: dict[str, pd.Series],
        generation_df: pd.DataFrame,
        is_hours_resolution: bool,
        hour_sample: np.ndarray,
    ) -> pd.DataFrame:
        dem_series_list = []
        for en_type, conv_en_rate in conversion_rates.items():
            conv_en_rate_resampled = conv_en_rate.iloc[hour_sample]
            if is_hours_resolution:
                dem_series = generation_df.div(conv_en_rate_resampled, axis=0).dropna()
                dem_series.index = pd.MultiIndex.from_tuples(
                    [(hour, en_type) for hour in dem_series.index],
                    names=[HOUR_LABEL, ENERGY_TYPE_LABEL],
                )
            else:
                dem_series = (
                    generation_df.div(conv_en_rate_resampled, axis=0).dropna().sum()
                )
                dem_series.name = en_type
            dem_series_list.append(dem_series)
        if is_hours_resolution:
            dem_df = pd.concat(dem_series_list)
        else:
            dem_df = pd.concat(dem_series_list, axis=1)
        return dem_df

    def get_generation_demand(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_generation_demand(generators, storages, is_hours_resolution)
        if not is_hours_resolution:
            df *= self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def _handle_brutto_netto_generator(self, gen_df: pd.DataFrame) -> pd.DataFrame:
        if self._capa_cost_label == GeneratorCapacityCostLabel.brutto:
            return gen_df
        df = gen_df.copy(deep=True)
        for gen in df.columns:
            _get_et = (
                self._network.generators[gen].energy_source_type
                if gen not in self._network.generator_types
                else gen
            )
            eff = (
                self._network.generator_types[_get_et]
                .efficiency.iloc[
                    :, 0
                ]  # If netto we are sure that efficiency is defined for one ET only
                .mean()
            )
            df[gen] = df[gen] * eff
        return df

    def _calculate_opex(
        self,
        generators: list[str],
        storages: list[str],
        cast_to_energy_source_type: bool = False,
    ) -> pd.DataFrame:
        generator_df = self._generator_results["capacity"]["capacity"]
        storage_df = self._storage_results["capacity"]["capacity"]
        generator_df = generator_df.loc[:, generator_df.columns.isin(generators)]
        storage_df = storage_df.loc[:, storage_df.columns.isin(storages)]
        for generator_name in generator_df.columns:
            generator_type_name = self._network.generators[
                generator_name
            ].energy_source_type
            opex = self._network.generator_types[generator_type_name].opex
            generator_df.loc[:, generator_name] = (
                (
                    generator_df[generator_name][self._year_sample]
                    * opex[self._year_sample].values
                )
                if self._year_sample is not None
                else generator_df[generator_name] * opex.values
            )
        for storage_name in storage_df.columns:
            storage_type_name = self._network.storages[storage_name].energy_source_type
            opex = self._network.storage_types[storage_type_name].opex
            storage_df.loc[:, storage_name] = (
                (
                    storage_df[storage_name][self._year_sample]
                    * opex[self._year_sample].values
                )
                if self._year_sample is not None
                else storage_df[storage_name] * opex.values
            )
        if cast_to_energy_source_type:
            generator_df = self._aggregate_by_type(
                generator_df, self._energy_source_type_mapping
            )
            storage_df = self._aggregate_by_type(
                storage_df, self._energy_source_type_mapping
            )
        generator_df = self._handle_brutto_netto_generator(generator_df)

        return pd.concat([generator_df, storage_df], axis=1).dropna()

    # TODO: we do not used cap plus in any func, just assign to class fields. Should we leave this as it is ?
    def _calculate_cap_plus(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        gen_cap_plus_df = (
            self._generator_results["capacity"]["capacity"].diff().fillna(0)
        )
        gen_cap_plus_df[gen_cap_plus_df[gen_cap_plus_df.columns] < 0] = 0
        stor_cap_plus_df = (
            self._storage_results["capacity"]["capacity"].diff().fillna(0)
        )
        stor_cap_plus_df[stor_cap_plus_df[stor_cap_plus_df.columns] < 0] = 0

        for (
            energy_source_type,
            energy_sources,
        ) in self._energy_source_type_mapping.items():
            unit_list = list(energy_sources)
            if energy_source_type in self._network.generator_types:
                gen_type_build_time = self._network.generator_types[
                    energy_source_type
                ].build_time
                gen_cap_plus_df[unit_list] = (
                    gen_cap_plus_df[unit_list].shift(-gen_type_build_time).fillna(0)
                )
            else:
                stor_type_build_time = self._network.storage_types[
                    energy_source_type
                ].build_time
                stor_cap_plus_df[unit_list] = (
                    stor_cap_plus_df[unit_list].shift(-stor_type_build_time).fillna(0)
                )

        return gen_cap_plus_df, stor_cap_plus_df

    @staticmethod
    def _format_capex_opex_dfs(
        capex_df: pd.DataFrame, opex_df: pd.DataFrame
    ) -> pd.DataFrame:
        if capex_df.empty and opex_df.empty:
            return pd.DataFrame()
        capex_df = assign_multiindex(capex_df, "capex")
        opex_df = assign_multiindex(opex_df, "opex")
        df = pd.concat([capex_df, opex_df], axis=1)
        df = df.reindex(
            columns=df.columns.get_level_values(0).sort_values().unique(), level=0
        )
        df = df.stack().swaplevel().unstack()
        return df

    def _get_global_capex_opex(
        self,
        generators: list[str],
        storages: list[str],
    ) -> pd.DataFrame:
        capex_df = self._generator_results["global_capex"]["global_capex"].join(
            self._storage_results["global_capex"]["global_capex"]
        )
        capex_df = capex_df.loc[:, capex_df.columns.isin(generators + storages)]
        opex_df = self._calculate_opex(generators, storages)
        return self._format_capex_opex_dfs(capex_df, opex_df)

    def _get_local_capex_opex(
        self,
        generators: list[str],
        storages: list[str],
        filter_type: Literal["aggr"] | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        if filter_type is not None and filter_names is not None:
            generator_values = [
                value
                for key, value in self._generator_results["local_capex"].items()
                if key in filter_names
            ]
            storage_values = [
                value
                for key, value in self._storage_results["local_capex"].items()
                if key in filter_names
            ]
        else:
            generator_values = list(self._generator_results["local_capex"].values())
            storage_values = list(self._storage_results["local_capex"].values())

        if not storage_values:
            capex_df = (
                pd.concat(generator_values, axis=0).groupby(level=0).sum()
                if generator_values
                else pd.DataFrame()
            )
        else:
            capex_df = (
                (pd.concat(generator_values, axis=0).groupby(level=0).sum()).join(
                    pd.concat(storage_values, axis=0).groupby(level=0).sum()
                )
                if generator_values
                else pd.concat(storage_values, axis=0).groupby(level=0).sum()
            )

        opex_df = self._calculate_opex(
            generators, storages, cast_to_energy_source_type=True
        )
        return self._format_capex_opex_dfs(capex_df, opex_df)

    def get_global_capex_opex(
        self,
        level: source_parameter_levels,
    ) -> pd.DataFrame:
        generators, storages = self._get_global_generators_and_storage()
        df = self._get_global_capex_opex(generators, storages)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=COST_TYPE_LABEL,
            level=level,
            filter_type=None,
            filter_names=None,
        )

    def get_local_capex_opex(
        self,
        filter_type: Literal["aggr"] | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        global_generators, global_storages = self._get_global_generators_and_storage()
        generators = list(set(generators) - set(global_generators))
        storages = list(set(storages) - set(global_storages))
        df = self._get_local_capex_opex(generators, storages, filter_type, filter_names)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=COST_TYPE_LABEL,
            level="element",  # element coz data is already in type but when level=type func cast element-> type
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _get_fuel_usage(
        self,
        generators: list[str],
        _storages: list[str],
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        if not generators:
            return pd.DataFrame()

        generator_dfs = self._generator_results["generation"]
        generator_dfs = {
            k: v if is_hours_resolution else v.sum()
            for k, v in generator_dfs.items()
            if k in generators
        }
        dfs = []
        for generator, generation_df in generator_dfs.items():
            generator_type = self._network.generators[generator].energy_source_type
            fuel_name = self._network.generator_types[generator_type].fuel
            if fuel_name is None:
                continue
            energy_per_unit = self._network.fuels[fuel_name].energy_per_unit
            df = generation_df / energy_per_unit
            if is_hours_resolution:
                df = self._format_hourly_resolution(df, generator, fuel_name)
            else:
                df = self._format_total_resolution(df, generator, fuel_name)
            dfs.append(df)

        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def _format_hourly_resolution(
        self, df: pd.DataFrame, generator: str, fuel_name: str
    ) -> pd.DataFrame:
        df.columns = df.columns.astype(int)
        df.columns = pd.MultiIndex.from_tuples([(generator, col) for col in df.columns])
        df.index = pd.MultiIndex.from_tuples([(hour, fuel_name) for hour in df.index])
        return df

    def _format_total_resolution(
        self, df: pd.Series, generator: str, fuel_name: str
    ) -> pd.DataFrame:
        df = df.to_frame()
        multiindex = pd.MultiIndex.from_tuples([(generator, fuel_name)])
        df.columns = multiindex
        df = df.stack().swaplevel().unstack()
        return df * self._hourly_scale

    def get_fuel_usage(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_fuel_usage(generators, storages, is_hours_resolution)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=FUEL_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def get_emission(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
        is_binding_skip: bool = False,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        fuel_usage_df = self._get_fuel_usage(generators, storages, is_hours_resolution)
        if fuel_usage_df.empty:
            return fuel_usage_df
        fuel_usage_df.columns = fuel_usage_df.columns.set_levels(
            fuel_usage_df.columns.levels[1].astype(int), level=1
        )
        fuels_emissions: dict[str, dict[str, float]] = {
            fuel_name: {
                emission_type: (
                    fuel.emission[emission_type]
                    if emission_type in fuel.emission
                    else 0
                )
                for emission_type in self._network.emission_types
            }
            for fuel_name, fuel in self._network.fuels.items()
        }

        dfs: list[dict[str, Any]] = self._get_emission_dfs_dicts(
            is_hours_resolution, fuel_usage_df, fuels_emissions
        )
        index_columns: list[str] | Literal["emission_type"] = (
            ["hour", "emission_type"] if is_hours_resolution else "emission_type"
        )

        df = (
            pd.DataFrame(dfs)
            .set_index(index_columns)
            .pivot(columns=["generator_name", "year"])
            .droplevel(0, axis=1)
        )

        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=EMISSION_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
            is_binding_skip=is_binding_skip,
        )

    def _get_emission_dfs_dicts(
        self,
        is_hours_resolution: bool,
        fuel_usage_df: pd.DataFrame,
        fuels_emissions: dict[str, dict[str, float]],
    ) -> list[dict[str, Any]]:
        if fuel_usage_df.empty or not fuels_emissions:
            return []
        dfs: list[dict[str, Any]] = []
        for generator_name in fuel_usage_df.columns.levels[0]:
            generator_em_reduction = self._network.generator_types[
                self._network.generators[generator_name].energy_source_type
            ].emission_reduction
            generator_fuel = self._network.generator_types[
                self._network.generators[generator_name].energy_source_type
            ].fuel

            if is_hours_resolution:
                dfs.extend(
                    [
                        {
                            "emission_type": emission_type,
                            "generator_name": generator_name,
                            "year": year,
                            "hour": hour,
                            "value": (
                                fuel_usage_df[generator_name].loc[
                                    (hour, generator_fuel)
                                ][year]
                                * fuels_emissions[generator_fuel][emission_type]
                                * (1 - generator_em_reduction[emission_type][year])
                            ),
                        }
                        for emission_type in fuels_emissions[generator_fuel].keys()
                        for year in self._year_sample
                        for hour in fuel_usage_df.index.get_level_values(0)[
                            fuel_usage_df.index.get_level_values(1) == generator_fuel
                        ]
                    ]
                )

            else:
                dfs.extend(
                    [
                        {
                            "emission_type": emission_type,
                            "generator_name": generator_name,
                            "year": year,
                            "value": (
                                fuel_usage_df[generator_name].loc[generator_fuel][year]
                                * fuels_emissions[generator_fuel][emission_type]
                                * (1 - generator_em_reduction[emission_type][year])
                            ),
                        }
                        for emission_type in fuels_emissions[generator_fuel].keys()
                        for year in self._year_sample
                    ]
                )

        return dfs

    def get_fuel_cost(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_fuel_usage(generators, storages)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.set_levels(df.columns.levels[1].astype(int), level=1)
        for fuel_name in df.index:
            fuel = self._network.fuels[fuel_name]
            for year in self._year_sample:
                df.loc[fuel.name, (slice(None), year)] *= fuel.cost[year]

        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=FUEL_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _get_data_per_unit(self, techs: Iterable, keys: list[str]) -> pd.DataFrame:
        if not techs or not keys:
            return pd.DataFrame()
        df_dict = {
            key: pd.concat(
                [getattr(t, key).to_frame(name=t.name) for t in techs], axis=1
            )
            for key in keys
        }
        df_dict = {
            key: v.iloc[self._year_sample].unstack() for key, v in df_dict.items()
        }
        return pd.concat(df_dict, axis=1).T

    def get_network_costs_per_tech_type(
        self,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        techs = list(self._network.generator_types.values()) + list(
            self._network.storage_types.values()
        )
        df = self._get_data_per_unit(
            techs=techs,
            keys=["capex", "opex"],
        )
        # we have input df for gen or stor type only so level must be element coz we dont want
        # to translate type -> element -> type
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=COST_TYPE_LABEL,
            level="element",
            filter_type=None,
            filter_names=filter_names,
        )

    def get_network_fuel_cost(
        self,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        fuels = self._network.fuels.values()
        df = self._get_data_per_unit(techs=fuels, keys=["cost"])
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=FUEL_LABEL,
            level="element",
            filter_type=None,
            filter_names=filter_names,
        )

    def get_network_fuel_availability(
        self,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        fuels = self._network.fuels.values()
        df = self._get_data_per_unit(techs=fuels, keys=["availability"])
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=FUEL_LABEL,
            level="element",
            filter_type=None,
            filter_names=filter_names,
        )

    def get_emission_fee_total_cost(
        self,
        level: source_parameter_levels,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        emissions_df = self.get_emission(level="element", is_binding_skip=True)
        filtered_gens_with_emission_fees = get_generators_emission_types(self._network)
        result = []
        for gen_name, emission_fee_names in filtered_gens_with_emission_fees.items():
            if gen_name not in emissions_df.index.get_level_values(0):
                continue
            emission_fees_emission_types = {
                self._network.emission_fees[ets_name].emission_type
                for ets_name in emission_fee_names
            }
            if len(emission_fees_emission_types) == 1 and len(emission_fee_names) > 1:
                total_price: pd.Series = sum(
                    self._network.emission_fees[ets_name].price
                    for ets_name in emission_fee_names
                )
                result.append(
                    self._calculate_emission_fee_total_cost(
                        emissions_df,
                        next(iter(emission_fee_names)),
                        gen_name,
                        total_price,
                    )
                )
            else:
                for emission_fee_name in emission_fee_names:
                    result.append(
                        self._calculate_emission_fee_total_cost(
                            emissions_df, emission_fee_name, gen_name
                        )
                    )
        if not result:
            return pd.DataFrame(columns=emissions_df.columns, index=emissions_df.index)
        df = pd.concat(result, axis=1).fillna(0).groupby(level=0, axis=1).sum()
        return self._aggregate_energy_sources(
            energy_source_df=df.T,
            column_name=YEARS_LABEL,
            index_name=EMISSION_LABEL,
            level=level,
            filter_type=None,
            filter_names=filter_names,
        )

    def _calculate_emission_fee_total_cost(
        self,
        emissions_df: pd.DataFrame,
        emission_fee_name: str,
        gen_name: str,
        total_price: pd.Series | None = None,
    ) -> pd.DataFrame:
        if (
            emission_fee_name not in self._network.emission_fees
            or gen_name not in emissions_df.index.get_level_values(0)
        ):
            return pd.DataFrame()
        emission_fee = self._network.emission_fees[emission_fee_name]
        mul_factor = (
            total_price[self._year_sample]
            if total_price is not None and len(total_price) >= len(self._year_sample)
            else emission_fee.price[self._year_sample]
        )
        series = (
            emissions_df.loc[gen_name, emission_fee.emission_type]
            .mul(mul_factor)
            .dropna()
        )
        index = pd.MultiIndex.from_product(
            [[gen_name], series.index], names=["gen_name", "year"]
        )
        return pd.DataFrame({emission_fee.emission_type: series.values}, index=index)

    def get_state_of_charge(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        _, storages = self._filter_elements(filter_type, filter_names)
        soc_df = self._get_storage_results(
            "state_of_charge", storages, is_hours_resolution
        )
        if not is_hours_resolution:
            soc_df *= self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=soc_df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def get_ens(
        self,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
        is_hours_resolution: bool = False,
    ) -> pd.DataFrame:
        *_, buses = self._filter_elements(filter_type, filter_names, is_bus_filter=True)
        ens_df = self._get_ens(is_hours_resolution, buses)
        if not is_hours_resolution:
            ens_df *= self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=ens_df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level="element",
            filter_type=filter_type,
            filter_names=filter_names,
            is_hours_resolution=is_hours_resolution,
        )

    def _get_ens(self, is_hours_resolution: bool, buses: list[str]) -> pd.DataFrame:
        ens_dfs = self._bus_results["generation_ens"]
        ens_dfs = {
            bus_name: df if is_hours_resolution else df.sum()
            for bus_name, df in ens_dfs.items()
            if bus_name in buses and bus_name in self._network.buses
        }
        for bus_name, ens_df in ens_dfs.items():
            energy_type = self._network.buses[bus_name].energy_type
            if is_hours_resolution:
                ens_df = ens_df.set_index(ens_df.index.map(lambda x: (x, energy_type)))
                ens_df.index.names = [HOUR_LABEL, ENERGY_TYPE_LABEL]
            else:
                ens_df = ens_df.to_frame().T
                ens_df.index = [energy_type]
            ens_dfs[bus_name] = ens_df
        return pd.concat(ens_dfs, axis=1).fillna(0.0) if ens_dfs else pd.DataFrame()
