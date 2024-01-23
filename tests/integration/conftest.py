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
