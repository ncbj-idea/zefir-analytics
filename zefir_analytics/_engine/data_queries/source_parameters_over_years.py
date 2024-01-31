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
from typing import Literal

import numpy as np
import pandas as pd
from pyzefir.model.network import Network

from zefir_analytics._engine.constants import (
    COST_TYPE_LABEL,
    EMISSION_LABEL,
    ENERGY_TYPE_LABEL,
    FUEL_LABEL,
    NETWORK_ELEMENT_NAME,
    NETWORK_ELEMENT_TYPE,
    YEARS_LABEL,
)

source_parameter_levels = Literal["element", "type"]
source_parameter_filter_types = Literal["bus", "stack", "aggr"]


class SourceParametersOverYearsQuery:
    def __init__(
        self,
        network: Network,
        generator_results: dict[str, dict[str, pd.DataFrame]],
        storage_results: dict[str, dict[str, pd.DataFrame]],
        year_sample: pd.Series,
        discount_rate: pd.Series,
        hourly_scale: float,
        hour_sample: np.ndarray,
    ):
        self._network = network
        self._generator_results = generator_results
        self._storage_results = storage_results
        self._year_sample = year_sample
        self._discount_rate = discount_rate
        self._energy_source_type_mapping = self._create_energy_source_type_mapping()
        (
            self._generator_capacity_plus,
            self._storage_capacity_plus,
        ) = self._calculate_cap_plus()
        self._hourly_scale = hourly_scale
        self._hour_sample = hour_sample

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
    ) -> pd.DataFrame:
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
            index=index_name,
        )

        if filter_type is None and filter_names is not None:
            filtered_columns = energy_source_df.columns.get_level_values(0).isin(
                filter_names
            )
            energy_source_df = energy_source_df.loc[:, filtered_columns]
        energy_source_df = energy_source_df.T
        energy_source_df.index = energy_source_df.index.set_levels(
            energy_source_df.index.get_level_values(1).astype(np.integer),
            level=1,
            verify_integrity=False,
        )
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
    ) -> tuple[list[str], list[str]]:
        generators = list(self._network.generators.keys())
        storages = list(self._network.storages.keys())

        if filter_type is None or filter_names is None:
            return generators, storages

        available_buses, available_stacks = filter_names, filter_names
        if filter_type == "aggr":
            available_stacks_2d = [
                list(aggr.stack_base_fraction.keys())
                for aggr in self._network.aggregated_consumers.values()
                if aggr.name in filter_names
            ]
            available_stacks = self._flatten_2d_list(available_stacks_2d)
        if filter_type in ["stack", "aggr"]:
            available_buses_2d = [
                list(stack.buses_out.values())
                for stack in self._network.local_balancing_stacks.values()
                if stack.name in available_stacks
            ]
            available_buses = self._flatten_2d_list(available_buses_2d)

        generators = [
            g
            for g in generators
            if self._network.generators[g].buses.intersection(available_buses)
        ]
        storages = [
            s for s in storages if self._network.storages[s].bus in available_buses
        ]

        return generators, storages

    def _get_generation_sum(
        self,
        generators: list[str],
        storages: list[str],
    ) -> pd.DataFrame:
        generator_dfs = self._generator_results["generation_per_energy_type"]
        storage_dfs = self._storage_results["generation"]

        generator_dfs = {
            k: v.pivot(columns=ENERGY_TYPE_LABEL).stack().groupby(level=1).sum()
            for k, v in generator_dfs.items()
            if k in generators
        }
        storage_dfs = {k: v.sum() for k, v in storage_dfs.items() if k in storages}

        for storage_name in storage_dfs.keys():
            storage_type = self._network.storages[storage_name].energy_source_type
            energy_type = self._network.storage_types[storage_type].energy_type
            df = storage_dfs[storage_name].to_frame().T
            df.index = [energy_type]
            storage_dfs[storage_name] = df

        generator_generation_sum = pd.concat(generator_dfs, axis=1)
        storage_generation_sum = pd.concat(storage_dfs, axis=1)
        generation_sum_df = pd.concat(
            [generator_generation_sum, storage_generation_sum], axis=1
        )

        return generation_sum_df * self._hourly_scale

    def get_generation_sum(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_generation_sum(generators, storages)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _get_dump_energy_sum(
        self, generators: list[str], _storages: list[str]
    ) -> pd.DataFrame:
        generator_dfs = self._generator_results["dump_energy_per_energy_type"]
        generator_dfs = {
            k: v.pivot(columns=ENERGY_TYPE_LABEL).stack().groupby(level=1).sum()
            for k, v in generator_dfs.items()
            if k in generators
        }
        generator_sum = pd.concat(generator_dfs, axis=1)
        return generator_sum

    def get_dump_energy_sum(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_dump_energy_sum(generators, storages) * self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _get_load_sum(
        self, _generators: list[str], storages: list[str]
    ) -> pd.DataFrame:
        storage_dfs = self._storage_results["load"]
        storage_dfs = {k: v.sum() for k, v in storage_dfs.items() if k in storages}
        for storage_name in storage_dfs.keys():
            storage_type = self._network.storages[storage_name].energy_source_type
            energy_type = self._network.storage_types[storage_type].energy_type
            df = storage_dfs[storage_name].to_frame().T
            df.index = [energy_type]
            storage_dfs[storage_name] = df
        storage_sum = pd.concat(storage_dfs, axis=1)
        return storage_sum

    def get_load_sum(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_load_sum(generators, storages) * self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
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
        self, generators: list[str], _storages: list[str]
    ) -> pd.DataFrame:
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
            dem_series_list = []
            for en_type, conv_en_rate in conversion_rates.items():
                conv_en_rate_resampled = conv_en_rate.iloc[
                    self._hour_sample
                ].reset_index(drop=True)
                dem_series = (
                    generation_df.mul(conv_en_rate_resampled, axis=0).dropna().sum()
                )
                dem_series.name = en_type
                dem_series_list.append(dem_series)
            dem_df = pd.concat(dem_series_list)
            if len(dem_df.shape) == 1:
                dem_df = dem_df.to_frame()
            multiindex = pd.MultiIndex.from_tuples(
                [(generator, col) for col in dem_df.columns]
            )
            dem_df.columns = multiindex
            dem_df = dem_df.stack().swaplevel().unstack()
            dfs.append(dem_df)
        return pd.concat(dfs, axis=1)

    def get_generation_demand(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_generation_demand(generators, storages) * self._hourly_scale
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=ENERGY_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _calculate_opex(
        self, generators: list[str], storages: list[str]
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
                generator_df[generator_name] * opex[self._year_sample].values
            )
        for storage_name in storage_df.columns:
            storage_type_name = self._network.storages[storage_name].energy_source_type
            opex = self._network.storage_types[storage_type_name].opex
            storage_df.loc[:, storage_name] = (
                storage_df[storage_name] * opex[self._year_sample].values
            )
        return pd.concat([generator_df, storage_df], axis=1)

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

    def _get_capex_opex(
        self, generators: list[str], storages: list[str]
    ) -> pd.DataFrame:
        capex_df = self._generator_results["capex"]["capex"].join(
            self._storage_results["capex"]["capex"]
        )
        opex_df = self._calculate_opex(generators, storages)
        capex_df.columns = pd.MultiIndex.from_tuples(
            [(col, "capex") for col in capex_df.columns]
        )
        opex_df.columns = pd.MultiIndex.from_tuples(
            [(col, "opex") for col in opex_df.columns]
        )
        df = pd.concat([capex_df, opex_df], axis=1)
        df = df.reindex(
            columns=df.columns.get_level_values(0).sort_values().unique(), level=0
        )
        df = df.stack().swaplevel().unstack()
        return df

    def get_capex_opex(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_capex_opex(generators, storages)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=COST_TYPE_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def _get_fuel_usage(
        self, generators: list[str], _storages: list[str]
    ) -> pd.DataFrame:
        generator_dfs = self._generator_results["generation"]
        generator_dfs = {
            k: v.sum() for k, v in generator_dfs.items() if k in generators
        }
        dfs = []
        for generator, generation_df in generator_dfs.items():
            generator_type = self._network.generators[generator].energy_source_type
            fuel_name = self._network.generator_types[generator_type].fuel
            if fuel_name is None:
                continue
            energy_per_unit = self._network.fuels[fuel_name].energy_per_unit
            df = generation_df / energy_per_unit
            df = df.to_frame()
            multiindex = pd.MultiIndex.from_tuples([(generator, fuel_name)])
            df.columns = multiindex
            df = df.stack().swaplevel().unstack() * self._hourly_scale
            dfs.append(df)

        return pd.concat(dfs, axis=1)

    def get_fuel_usage(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_fuel_usage(generators, storages)
        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=FUEL_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )

    def get_emission(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        fuel_usage_df = self._get_fuel_usage(generators, storages)

        fuels_emissions: dict[str, dict[str, float]] = {
            fuel_name: {
                emission_type: fuel.emission[emission_type]
                if emission_type in fuel.emission
                else 0
                for emission_type in self._network.emission_types
            }
            for fuel_name, fuel in self._network.fuels.items()
        }

        dfs = []
        for generator_name in fuel_usage_df.columns.levels[0]:
            generator_em_reduction = self._network.generator_types[
                self._network.generators[generator_name].energy_source_type
            ].emission_reduction
            generator_fuel = self._network.generator_types[
                self._network.generators[generator_name].energy_source_type
            ].fuel

            dfs.extend(
                [
                    {
                        "emission_type": emission_type,
                        "generator_name": generator_name,
                        "year": year,
                        "value": fuel_usage_df[generator_name].loc[generator_fuel][
                            str(year)
                        ]
                        * fuels_emissions[generator_fuel][emission_type]
                        * (1 - generator_em_reduction[emission_type]),
                    }
                    for emission_type in fuels_emissions[generator_fuel].keys()
                    for year in self._year_sample
                ]
            )

        df = (
            pd.DataFrame(dfs)
            .set_index("emission_type")
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
        )

    def get_fuel_cost(
        self,
        level: source_parameter_levels,
        filter_type: source_parameter_filter_types | None = None,
        filter_names: list[str] | None = None,
    ) -> pd.DataFrame:
        generators, storages = self._filter_elements(filter_type, filter_names)
        df = self._get_fuel_usage(generators, storages)

        for fuel_name in df.index:
            fuel = self._network.fuels[fuel_name]
            for year in self._year_sample:
                df.loc[fuel.name, (slice(None), str(year))] *= fuel.cost[year]

        return self._aggregate_energy_sources(
            energy_source_df=df,
            column_name=YEARS_LABEL,
            index_name=FUEL_LABEL,
            level=level,
            filter_type=filter_type,
            filter_names=filter_names,
        )
