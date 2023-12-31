import numpy as np
import pandas as pd
from pyzefir.model.network import Network

from zefir_analytics._engine.constants import YEARS_LABEL
from zefir_analytics._engine.data_queries import utils as data_utils


class LineParametersOverYearsQuery:
    def __init__(
        self,
        network: Network,
        line_results: dict[str, dict[str, pd.DataFrame]],
    ) -> None:
        self._network = network
        self._line_results = line_results

    @property
    def line_names(self) -> list[str]:
        return list(self._line_results["flow"].keys())

    @staticmethod
    def _get_yearly_summary(
        df: pd.DataFrame, column_name: str, operation: str
    ) -> pd.DataFrame:
        df = df.agg(axis=0, func=operation).to_frame(column_name)
        df.index.name = YEARS_LABEL
        df.index = df.index.astype(np.integer)
        return df

    def _get_flow(self, line_name: str) -> pd.DataFrame:
        return self._get_yearly_summary(
            self._line_results["flow"][line_name], "Total energy volume", "sum"
        )

    def _get_transmission_fee(self, line_name: str) -> pd.DataFrame:
        df_flow = self._line_results["flow"][line_name]
        if tf_name := self._network.lines[line_name].transmission_fee:
            series_tf = self._network.transmission_fees[tf_name].fee
            df = df_flow * series_tf[df_flow.index].values[:, None]
        else:
            df = df_flow * 0.0
        return self._get_yearly_summary(df, "Transmission fee total cost", "sum")

    def get_flow(
        self, line_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return (
            data_utils.argument_condition(line_name, self._get_flow)
            if line_name is not None
            else data_utils.argument_condition(self.line_names, self._get_flow)
        )

    def get_capacity(
        self, line_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        raise NotImplementedError(
            "This method will be implemented later when basecode will be ready"
        )

    def get_transmission_fee(
        self, line_name: str | list[str] | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return (
            data_utils.argument_condition(line_name, self._get_transmission_fee)
            if line_name is not None
            else data_utils.argument_condition(
                self.line_names, self._get_transmission_fee
            )
        )
