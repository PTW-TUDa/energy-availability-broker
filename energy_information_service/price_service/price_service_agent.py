from __future__ import annotations

from osbrain import Proxy, run_agent

from energy_information_service.config import NS_ADDR, AgentNames
from energy_information_service.osbrain_modules.flex_agent import FlexAgent, retry_on_failure


class PriceServiceAgent(FlexAgent):
    name = AgentNames.PRICE_SERVICE

    def on_init(self):
        super().on_init()

    def before_loop(self):
        pass


@retry_on_failure()
@staticmethod
def start_agent(name: str = AgentNames.PRICE_SERVICE, trannsport: str = "tcp") -> Proxy:
    return run_agent(name=name, transport=trannsport, nsaddr=NS_ADDR, base=PriceServiceAgent)
