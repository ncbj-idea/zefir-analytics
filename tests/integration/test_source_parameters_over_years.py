import numpy as np
import pandas as pd

from zefir_analytics import ZefirEngine


def test_source_parameters_over_years(zefir_engine: ZefirEngine) -> None:
    ze = zefir_engine
    zefir_results = [
        ze.source_params.get_generation_sum(level="element"),
        ze.source_params.get_dump_energy_sum(level="element"),
        ze.source_params.get_load_sum(level="type"),
        ze.source_params.get_installed_capacity(
            level="element", filter_type="bus", filter_names=["KSE"]
        ),
        ze.source_params.get_generation_demand(level="element"),
        ze.source_params.get_fuel_usage(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        ze.source_params.get_capex_opex(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
        ze.source_params.get_emission(
            level="type", filter_type="aggr", filter_names=["MULTI_FAMILY"]
        ),
    ]
    assert len(zefir_results)
    assert all(not df.empty for df in zefir_results)


def test_cap_plus_calculation(zefir_engine: ZefirEngine) -> None:
    ze = zefir_engine
    ze.source_params._generator_results = {
        "capacity": {
            "capacity": pd.DataFrame(
                data=np.array(
                    [
                        [10, 20, 30, 40, 50],
                        [10, 10, 10, 40, 50],
                    ]
                ).transpose(),
                index=[0, 1, 2, 3, 4],
                columns=["GEN_1", "GEN_2"],
            )
        }
    }
    ze.network.generator_types["GEN_TYPE_1"] = type("GeneratorType", (object,), {"build_time": 1})  # type: ignore
    ze.network.generator_types["GEN_TYPE_2"] = type("GeneratorType", (object,), {"build_time": 3})  # type: ignore
    ze.network.generators["GEN_1"] = type("Generator", (object,), {"type": "GEN_TYPE_1"})  # type: ignore
    ze.network.generators["GEN_2"] = type("Generator", (object,), {"type": "GEN_TYPE_2"})  # type: ignore

    ze.source_params._storage_results = {
        "capacity": {
            "capacity": pd.DataFrame(
                data=np.array(
                    [
                        [10, 20, 30, 40, 50],
                        [20, 20, 20, 30, 50],
                    ]
                ).transpose(),
                index=[0, 1, 2, 3, 4],
                columns=["STOR_1", "STOR_2"],
            )
        }
    }
    ze.network.storage_types["STOR_TYPE_1"] = type("StorageType", (object,), {"build_time": 1})  # type: ignore
    ze.network.storage_types["STOR_TYPE_2"] = type("StorageType", (object,), {"build_time": 3})  # type: ignore
    ze.network.storages["STOR_1"] = type("Storage", (object,), {"build_time": 1})  # type: ignore
    ze.network.storages["STOR_2"] = type("Storage", (object,), {"build_time": 3})  # type: ignore

    ze.source_params._energy_source_type_mapping = {
        "GEN_TYPE_1": ["GEN_1"],
        "GEN_TYPE_2": ["GEN_2"],
        "STOR_TYPE_1": ["STOR_1"],
        "STOR_TYPE_2": ["STOR_2"],
    }

    gen_cap_plus, stor_cap_plus = ze.source_params._calculate_cap_plus()

    assert gen_cap_plus.equals(
        pd.DataFrame(
            data=np.array(
                [
                    [10.0, 10.0, 10.0, 10.0, 0.0],
                    [30.0, 10.0, 0.0, 0.0, 0.0],
                ]
            ).transpose(),
            index=[0, 1, 2, 3, 4],
            columns=["GEN_1", "GEN_2"],
        )
    )
    assert stor_cap_plus.equals(
        pd.DataFrame(
            data=np.array(
                [
                    [10.0, 10.0, 10.0, 10.0, 0.0],
                    [10.0, 20.0, 0.0, 0.0, 0.0],
                ]
            ).transpose(),
            index=[0, 1, 2, 3, 4],
            columns=["STOR_1", "STOR_2"],
        )
    )
