# ZefirAnalytics
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

# ZefirAnalytics
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of theTrue)nse, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from enum import StrEnum, auto
from typing import Callable, TypeVar

import pandas as pd
from pyzefir.model.network import Network

T = TypeVar("T")


class GeneratorCapacityCostLabel(StrEnum):
    brutto = auto()
    netto = auto()


def argument_condition(
    name: str | list[str], function: Callable[[str], pd.DataFrame]
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    if isinstance(name, str):
        return function(name)
    return {el_name: function(el_name) for el_name in name}


def dict_filter(
    dictionary: dict[str, T], keys: list[str] | str | None
) -> dict[str, T] | T:
    if keys is None:
        return dictionary
    elif isinstance(keys, str):
        return dictionary.get(keys, {})
    else:
        return {key: dictionary[key] for key in keys if key in dictionary}


def assign_multiindex(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if not df.empty:
        df.columns = pd.MultiIndex.from_tuples([(col, label) for col in df.columns])
    return df


def reindex_single_df_binding_years(
    df: pd.DataFrame, binding_years: pd.Series
) -> pd.DataFrame:
    df_reindex = df.copy(deep=True)
    df_reindex.index = df_reindex.index.map(lambda x: binding_years[x])
    return df_reindex


def reindex_multiindex_df_binding_years(
    df: pd.DataFrame, binding_years: pd.Series, level: str = "Year"
) -> pd.DataFrame:
    df_reindex = df.copy(deep=True)
    years = df_reindex.index.get_level_values(level).map(lambda x: binding_years[x])
    df_reindex.index = df_reindex.index.set_levels(
        years, level=level, verify_integrity=False
    )
    return df_reindex


def handle_n_sample_results(
    results: dict[str, pd.DataFrame] | pd.DataFrame,
    year_binding: pd.Series | None,
    is_multiindex: bool = False,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    if year_binding is None:
        return results
    if isinstance(results, dict):
        return {
            key: reindex_single_df_binding_years(value, year_binding)
            for key, value in results.items()
        }
    else:
        if is_multiindex:
            return reindex_multiindex_df_binding_years(results, year_binding)
        return reindex_single_df_binding_years(results, year_binding)


def get_generators_emission_types(network: Network) -> dict[str, set[str]]:
    return {
        key: obj.emission_fee
        for key, obj in network.generators.items()
        if obj.emission_fee
    }
