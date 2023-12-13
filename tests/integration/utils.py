import pandas as pd


def assert_analytics_result(
    zefir_result: list[pd.DataFrame] | list[dict[str, pd.DataFrame]]
) -> None:
    for result in zefir_result:
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            assert not result.empty
        elif isinstance(result, dict):
            for key, value in result.items():
                assert isinstance(key, str)
                assert isinstance(value, pd.DataFrame) or isinstance(value, pd.Series)
                assert not value.empty
        else:
            assert False
