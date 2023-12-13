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
