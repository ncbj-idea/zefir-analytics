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

from typing import Callable, TypeVar

import pandas as pd

T = TypeVar("T")


def argument_condition(
    name: str | list[str], function: Callable[[str], pd.DataFrame]
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    if isinstance(name, str):
        return function(name)
    return {el_name: function(el_name) for el_name in name}


def dict_filter(
    dictionary: dict[str, T], keys: list[str] | str | None
) -> dict[str, T] | T:
    return (
        dictionary
        if keys is None
        else dictionary[keys]
        if isinstance(keys, str)
        else {key: dictionary[key] for key in keys}
    )
