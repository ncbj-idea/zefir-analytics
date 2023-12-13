from pytest import fixture

from tests.utils import get_resources
from zefir_analytics import ZefirEngine


@fixture
def zefir_engine() -> ZefirEngine:
    data = get_resources("simple-data-case")
    ze = ZefirEngine(
        source_path=data / "source_csv",
        result_path=data / "results",
        scenario_name="scenario_1",
        parameter_path=data / "parameters",
    )
    yield ze
