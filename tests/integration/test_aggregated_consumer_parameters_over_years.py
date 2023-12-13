from tests.integration.utils import assert_analytics_result
from zefir_analytics import ZefirEngine


def test_aggregated_consumer_params_over_years(zefir_engine: ZefirEngine) -> None:
    ze = zefir_engine
    zefir_results_dataframe_res = [
        ze.aggregated_consumer_params.get_fractions("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_n_consumers("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_yearly_energy_usage("SINGLE_FAMILY"),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage("MULTI_FAMILY"),
        ze.aggregated_consumer_params.get_fractions(),
        ze.aggregated_consumer_params.get_n_consumers(),
        ze.aggregated_consumer_params.get_yearly_energy_usage(),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage(),
        ze.aggregated_consumer_params.get_fractions(["MULTI_FAMILY"]),
        ze.aggregated_consumer_params.get_n_consumers(
            ["MULTI_FAMILY", "SINGLE_FAMILY"]
        ),
        ze.aggregated_consumer_params.get_yearly_energy_usage(["MULTI_FAMILY"]),
        ze.aggregated_consumer_params.get_total_yearly_energy_usage(
            ["MULTI_FAMILY", "SINGLE_FAMILY"]
        ),
    ]

    assert_analytics_result(
        zefir_results_dataframe_res,
    )
