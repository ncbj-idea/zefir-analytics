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


@pytest.fixture(scope="session")
def data_path() -> Path:
    return get_resources("simple-data-case")


@pytest.fixture(scope="session")
def input_path(data_path: Path) -> Path:
    return data_path / "source_csv"


@pytest.fixture(scope="session")
def parameters_path(data_path: Path) -> Path:
    return data_path / "parameters"


@pytest.fixture(scope="session")
def results_path(data_path: Path) -> Path:
    return data_path / "results"


@pytest.fixture(scope="session")
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


@pytest.fixture
def zefir_engine(
    input_path: Path,
    results_path: Path,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Generator[ZefirEngine, None, None]:
    year_sample, hour_sample, discount_rate = parameters
    ze = ZefirEngine(
        source_path=input_path,
        result_path=results_path,
        scenario_name="scenario_1",
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        used_hourly_scale=True,
    )
    yield ze
