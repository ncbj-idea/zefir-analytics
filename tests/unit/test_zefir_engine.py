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

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pyzefir.model.network import Network
from pyzefir.postprocessing.results_handler import GeneralResultDirectory

from zefir_analytics._engine.data_queries.utils import GeneratorCapacityCostLabel
from zefir_analytics.zefir_engine import ZefirEngine


@pytest.fixture
def test_data() -> dict[str, Any]:
    return {
        "source_path": Path("/path/to/source"),
        "result_path": Path("/path/to/result"),
        "scenario_name": "TestScenario",
        "discount_rate": np.array([0.05, 0.1]),
        "year_sample": np.array([2020, 2021]),
        "hour_sample": np.array([1, 2, 3]),
        "used_hourly_scale": True,
        "generator_capacity_cost": "brutto",
        "n_years_aggregation": 1,
    }


@pytest.mark.parametrize(
    "generator_capacity_cost, expected_label",
    [
        ("brutto", GeneratorCapacityCostLabel.brutto),
        ("netto", GeneratorCapacityCostLabel.netto),
    ],
)
def test_validate_capacity_cost(
    generator_capacity_cost: str, expected_label: GeneratorCapacityCostLabel
) -> None:
    label = ZefirEngine._validate_capacity_cost(generator_capacity_cost)
    assert label == expected_label


@pytest.mark.parametrize(
    "generator_capacity_cost",
    ["invalid_cost"],
)
def test_validate_capacity_cost_invalid(generator_capacity_cost: str) -> None:
    with pytest.raises(ValueError):
        ZefirEngine._validate_capacity_cost(generator_capacity_cost)


@pytest.mark.parametrize(
    "array, dtype, attr_name, raises_exception",
    [
        pytest.param(
            np.array([1, 2, 3]), np.int64, "year_sample", False, id="int_sample"
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            np.float64,
            "discount_rate",
            False,
            id="float_sample",
        ),
        pytest.param([1, 2, 3], np.int64, "year_sample", True, id="not_numpy_array"),
        pytest.param(
            np.array([1, 2, 3]), np.float32, "year_sample", True, id="wrong_dtype"
        ),
    ],
)
def test_validate_parameter_array(
    array: np.ndarray,
    dtype: type,
    attr_name: str,
    raises_exception: bool,
) -> None:
    if raises_exception:
        with pytest.raises(TypeError):
            ZefirEngine._validate_parameter_array(array, dtype, attr_name)
    else:
        result = ZefirEngine._validate_parameter_array(array, dtype, attr_name)
        assert result is array


@patch("zefir_analytics._engine.DataLoader.load_data")
def test_load_input_data(mock_load_data: MagicMock) -> None:
    mock_network = MagicMock(spec=Network)
    mock_source_dict = {"mock_key": {"mock_subkey": MagicMock()}}
    mock_result_dict = {"mock_key": {"mock_subkey": {"mock_subsubkey": MagicMock()}}}
    mock_objective_func_value = 42.0

    mock_load_data.return_value = (
        mock_source_dict,
        mock_network,
        mock_result_dict,
        mock_objective_func_value,
    )

    source_path = Path("/mock/source")
    result_path = Path("/mock/result")
    scenario_name = "MockScenario"

    source_dict, network, result_dict, objective_func_value = (
        ZefirEngine._load_input_data(source_path, result_path, scenario_name)
    )

    assert source_dict == mock_source_dict
    assert network == mock_network
    assert result_dict == mock_result_dict
    assert objective_func_value == mock_objective_func_value
    mock_load_data.assert_called_once_with(source_path, result_path, scenario_name)


@pytest.mark.parametrize(
    "n_years_aggregation",
    [
        pytest.param(1, id="year_aggr_1"),
        pytest.param(2, id="year_aggr_2"),
        pytest.param(5, id="year_aggr_5"),
    ],
)
def test_initialization_with_aggregation(n_years_aggregation: int) -> None:
    with patch(
        "zefir_analytics.ZefirEngine._validate_capacity_cost"
    ) as MockValidateCapacityCost, patch(
        "zefir_analytics.ZefirEngine._validate_parameter_array"
    ) as MockValidateParamArray, patch(
        "zefir_analytics.ZefirEngine._load_input_data"
    ) as MockLoadInputData, patch(
        "zefir_analytics.zefir_engine.OptConfig"
    ) as MockOptConfig, patch(
        "zefir_analytics._engine.SourceParametersOverYearsQuery"
    ) as MockSourceParams, patch(
        "zefir_analytics._engine.LineParametersOverYearsQuery"
    ) as MockLineParams, patch(
        "zefir_analytics._engine.AggregatedConsumerParametersOverYearsQuery"
    ) as MockAggregatedConsumerParams, patch(
        "zefir_analytics._engine.LbsParametersOverYearsQuery"
    ) as MockLbsParams, patch(
        "zefir_analytics.zefir_engine.NetworkAggregator", autospec=True
    ) as MockNetworkAggregator:

        mock_aggregator = MagicMock()
        MockNetworkAggregator.return_value = mock_aggregator
        mock_aggregator.get_years_binding.return_value = MagicMock(
            index=pd.Series([2024, 2025, 2026])
        )

        MockValidateCapacityCost.return_value = MagicMock(value="mocked_value")
        MockValidateParamArray.side_effect = lambda x, dtype, name: np.array(
            x, dtype=dtype
        )
        MockLoadInputData.return_value = (
            {"mock_source_dict": {}},
            MagicMock(),
            {
                GeneralResultDirectory.GENERATORS_RESULTS: {},
                GeneralResultDirectory.STORAGES_RESULTS: {},
                GeneralResultDirectory.BUS_RESULTS: {},
                GeneralResultDirectory.LINES_RESULTS: {},
                GeneralResultDirectory.FRACTIONS_RESULTS: {},
            },
            1.0,
        )
        MockOptConfig.return_value = MagicMock()
        MockSourceParams.return_value = MagicMock()
        MockLineParams.return_value = MagicMock()
        MockAggregatedConsumerParams.return_value = MagicMock()
        MockLbsParams.return_value = MagicMock()

        source_path = Path("mock_source_path")
        result_path = Path("mock_result_path")
        scenario_name = "mock_scenario"
        discount_rate = np.array([0.05])
        year_sample = np.array([2024])
        hour_sample = np.array([1])
        used_hourly_scale = True

        engine = ZefirEngine(
            source_path=source_path,
            result_path=result_path,
            scenario_name=scenario_name,
            discount_rate=discount_rate,
            year_sample=year_sample,
            hour_sample=hour_sample,
            used_hourly_scale=used_hourly_scale,
            generator_capacity_cost="brutto",
            n_years_aggregation=n_years_aggregation,
        )

        assert engine._generator_capacity_cost_label.value == "mocked_value"
        np.testing.assert_array_equal(
            engine._year_sample,
            (
                np.array([2024, 2025, 2026])
                if n_years_aggregation > 1
                else np.array([2024])
            ),
        )
        np.testing.assert_array_equal(engine._hour_sample, np.array([1]))
        np.testing.assert_array_equal(engine._discount_rate, np.array([0.05]))
        assert isinstance(engine._network, MagicMock)
        assert engine._objective_func_value == 1.0

        MockValidateCapacityCost.assert_called_once_with("brutto")
        MockValidateParamArray.assert_any_call(year_sample, np.int64, "year_sample")
        MockValidateParamArray.assert_any_call(hour_sample, np.int64, "hour_sample")
        MockValidateParamArray.assert_any_call(
            discount_rate, np.float64, "discount_rate"
        )
        MockLoadInputData.assert_called_once_with(
            source_path=source_path,
            result_path=result_path,
            scenario_name=scenario_name,
        )
        MockOptConfig.assert_called_once_with(
            hours=engine._network.constants.n_hours,
            years=engine._network.constants.n_years,
            hour_sample=engine._hour_sample,
            use_hourly_scale=used_hourly_scale,
        )
        MockSourceParams.assert_called_once_with(
            network=engine.network,
            generator_results={},
            storage_results={},
            bus_results={},
            year_sample=engine._year_sample,
            discount_rate=engine._discount_rate,
            hourly_scale=engine._opt_config.hourly_scale,
            hour_sample=engine._opt_config.hour_sample,
            generator_capacity_cost_label=engine._generator_capacity_cost_label,
            years_binding=engine._years_binding,
        )
        MockLineParams.assert_called_once_with(
            network=engine.network,
            line_results={},
            hourly_scale=engine._opt_config.hourly_scale,
            hour_sample=engine._opt_config.hour_sample,
            years_binding=engine._years_binding,
        )
        MockAggregatedConsumerParams.assert_called_once_with(
            network=engine.network,
            fraction_results={},
            years_binding=engine._years_binding,
        )
        MockLbsParams.assert_called_once_with(
            network=engine.network,
            fractions_results={},
            generator_results={},
            storage_results={},
            years_binding=engine._years_binding,
        )

        if n_years_aggregation > 1:
            MockNetworkAggregator.assert_called_once_with(
                n_years=engine._network.constants.n_years,
                n_years_aggregation=n_years_aggregation,
                year_sample=np.array([2024]),
            )
            mock_aggregator.aggregate_network.assert_called_once_with(engine._network)
            mock_aggregator.get_years_binding.assert_called_once()
        else:
            MockNetworkAggregator.assert_not_called()

        assert isinstance(engine.network, MagicMock)
        assert engine.scenario_name == scenario_name
        engine.source_dict == {"mock_source_dict": {}}
        engine.result_dict == {
            GeneralResultDirectory.GENERATORS_RESULTS: {},
            GeneralResultDirectory.STORAGES_RESULTS: {},
            GeneralResultDirectory.BUS_RESULTS: {},
            GeneralResultDirectory.LINES_RESULTS: {},
            GeneralResultDirectory.FRACTIONS_RESULTS: {},
        }
        assert engine.generator_capacity_cost == "mocked_value"
        assert isinstance(engine.source_params, MagicMock)
        assert isinstance(engine.line_params, MagicMock)
        assert isinstance(engine.aggregated_consumer_params, MagicMock)
        assert isinstance(engine.lbs_params, MagicMock)
        assert engine.objective_function_value == pytest.approx(1.0)


@pytest.mark.parametrize(
    "input_format, expected_source_path, expected_result_path",
    [
        pytest.param("csv", "input_csv_path", "output_path/csv", id="csv"),
        pytest.param("xlsx", "csv_dump_path", "output_path/csv", id="xlsx"),
        pytest.param("feather", "csv_dump_path", "output_path/feather", id="feather"),
    ],
)
def test_create_from_config(
    input_format: str, expected_source_path: str, expected_result_path: str
) -> None:
    mock_config = MagicMock()
    mock_config.input_format = input_format
    mock_config.input_path = Path("input_csv_path")
    mock_config.csv_dump_path = Path("csv_dump_path")
    mock_config.output_path = Path("output_path")
    mock_config.scenario = "test_scenario"
    mock_config.discount_rate = [0.05]
    mock_config.year_sample = [2024]
    mock_config.hour_sample = [1]
    mock_config.use_hourly_scale = True
    mock_config.network_config = {"generator_capacity_cost": "brutto"}
    mock_config.n_years_aggregation = 5

    with patch("zefir_analytics.zefir_engine.ConfigLoader") as MockConfigLoader:
        MockConfigLoader.return_value.load.return_value = mock_config
        with patch("zefir_analytics.zefir_engine.ZefirEngine.__new__") as mock_new:
            mock_instance = MagicMock()
            mock_new.return_value = mock_instance
            result = ZefirEngine.create_from_config(Path("path/to/config/file"))
            MockConfigLoader.assert_called_once_with(Path("path/to/config/file"))
            MockConfigLoader.return_value.load.assert_called_once()
            mock_new.assert_called_once_with(
                ZefirEngine,
                source_path=Path(expected_source_path),
                result_path=Path(expected_result_path),
                scenario_name="test_scenario",
                discount_rate=[0.05],
                year_sample=[2024],
                hour_sample=[1],
                used_hourly_scale=True,
                generator_capacity_cost="brutto",
                n_years_aggregation=5,
            )

            assert result == mock_instance
