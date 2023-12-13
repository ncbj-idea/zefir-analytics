import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.integration.utils import assert_analytics_result
from zefir_analytics.zefir_engine import ZefirEngine


@pytest.fixture
def line_names() -> list[str | list[str]]:
    return [
        "DH -> MF_BASIC_H",
        "DH -> OS_BASIC_H",
        ["DH -> MF_BASIC_H", "DH -> OS_BASIC_H"],
        ["KSE -> MF_GAS_EE", "KSE -> MF_HP_EE"],
    ]


@pytest.fixture
def lines_without_tf() -> list[str]:
    return ["KSE -> SF_GAS_EE", "KSE -> SF_HP_EE"]


def test_variability_of_lines_parameters(
    zefir_engine: ZefirEngine,
    line_names: list[str | list[str]],
) -> None:
    ze = zefir_engine
    for name in line_names:
        assert_analytics_result([ze.line_params.get_flow(name)])
        assert_analytics_result([ze.line_params.get_transmission_fee(name)])


def test_cost_without_transmission_fee(
    zefir_engine: ZefirEngine, lines_without_tf: list[str]
) -> None:
    ze = zefir_engine
    results: list[pd.DataFrame] = []  # no comprehension coz of return type
    for line_name in lines_without_tf:
        results.append(ze.line_params.get_transmission_fee(line_name))

    for df in results:
        df_zero = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        df.equals(df_zero)


@pytest.mark.parametrize(
    ("df", "operation", "expected_df"),
    [
        pytest.param(
            pd.DataFrame({1: [1, 1], 3: [2, 2], 7: [3, 3]}),
            "sum",
            pd.DataFrame({"test": [2, 4, 6]}, index=pd.Index([1, 3, 7], name="Year")),
            id="df_sum_values",
        ),
        pytest.param(
            pd.DataFrame({1: [1, 3], 3: [2, 2], 7: [5, 5]}),
            "mean",
            pd.DataFrame(
                {"test": [2.0, 2.0, 5.0]}, index=pd.Index([1, 3, 7], name="Year")
            ),
            id="df_mean_values",
        ),
    ],
)
def test_get_yearly_summary(
    zefir_engine: ZefirEngine,
    df: pd.DataFrame,
    expected_df: pd.DataFrame,
    operation: str,
) -> None:
    ze = zefir_engine
    result = ze.line_params._get_yearly_summary(df, "test", operation)
    assert_frame_equal(result, expected_df)


def test_line_parameters_none(
    zefir_engine: ZefirEngine,
) -> None:
    ze = zefir_engine
    assert_analytics_result([ze.line_params.get_flow()])
    assert_analytics_result([ze.line_params.get_transmission_fee()])
