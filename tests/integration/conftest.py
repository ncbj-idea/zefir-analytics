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

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from tests.utils import get_resources
from zefir_analytics import ZefirEngine


def data_path(resource_name: str) -> Path:
    return get_resources(resource_name)


def input_path(data_path: Path) -> Path:
    return data_path / "source_csv"


def parameters_path(data_path: Path) -> Path:
    return data_path / "parameters"


def results_path(data_path: Path) -> Path:
    return data_path / "results"


def parameters(parameters_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    year_sample: np.ndarray = (
        pd.read_csv(parameters_path / "year_sample.csv", header=None)
        .squeeze()
        .to_numpy()
    )
    hour_sample: np.ndarray = (
        pd.read_csv(parameters_path / "hour_sample.csv", header=None)
        .squeeze()
        .to_numpy()
    )
    discount_rate: np.ndarray = (
        pd.read_csv(parameters_path / "discount_rate.csv", header=None)
        .squeeze()
        .to_numpy()
    )
    return year_sample, hour_sample, discount_rate


def get_paths_and_data_for_engine(
    resource_name: str,
) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray]:
    resource_path = data_path(resource_name)
    imp_path = input_path(resource_path)
    res_path = results_path(resource_path)
    year_sample, hour_sample, discount_rate = parameters(parameters_path(resource_path))
    return (imp_path, res_path, year_sample, hour_sample, discount_rate)


def get_paths_for_config(resource_name: str) -> tuple[Path, ...]:
    resource_path = data_path(resource_name)
    imp_path = input_path(resource_path)
    res_path = results_path(resource_path)
    param_path = parameters_path(resource_path)
    return imp_path, res_path, param_path


@pytest.fixture
def zefir_engine() -> Generator[ZefirEngine, None, None]:
    input_path, results_path, year_sample, hour_sample, discount_rate = (
        get_paths_and_data_for_engine("simple-data-case")
    )
    ze = ZefirEngine(
        source_path=input_path,
        result_path=results_path / "csv",
        scenario_name="scenario_1",
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        used_hourly_scale=True,
    )
    yield ze


@pytest.fixture
def zefir_engine_n_sampled() -> Generator[ZefirEngine, None, None]:
    input_path, results_path, year_sample, hour_sample, discount_rate = (
        get_paths_and_data_for_engine("simple-nsample-case")
    )
    ze = ZefirEngine(
        source_path=input_path,
        result_path=results_path / "csv",
        scenario_name="scenario_1",
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        used_hourly_scale=True,
        n_years_aggregation=2,
    )
    yield ze
