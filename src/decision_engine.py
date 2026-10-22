"""
decision_engine.py

The Cost-Benefit Decision Engine: the project's core theoretical
contribution. For a KV-cache block, given an estimated benefit and cost
for each candidate action, it selects the action with the highest
Net Utility:

    Net Utility(action) = Estimated Benefit(action) - Estimated Cost(action)

This is a greedy, real-time approximation of the constrained
optimization problem framed in the literature review (Part 3): each
action is evaluated on equal footing via a single scalar, rather than
mapping a score directly to one hard-coded action (as static-threshold
systems like MTDS do).

This module accepts plain {"benefit": x, "cost": y} dicts (the original
Zeroth-era interface, kept for backward compatibility with existing
tests/commits) as well as ActionEstimate objects from
AdaptiveImportanceEngine, so it can be used standalone or as part of
the full pipeline.
"""

from __future__ import annotations

from typing import Mapping, Tuple, Union

from src.importance_engine import ActionEstimate

ActionValue = Union[Mapping[str, float], ActionEstimate]


class DecisionEngine:
    """Selects the action with the highest Net Utility for a block."""

    def calculate_utility(self, benefit: float, cost: float) -> float:
        """Net Utility = Estimated Benefit - Estimated Cost."""
        return benefit - cost

    def decide(self, actions: Mapping[str, ActionValue]) -> Tuple[str, float]:
        """
        Choose the best action among the candidates.

        Args:
            actions: mapping of action name -> either
                {"benefit": float, "cost": float} or an ActionEstimate.

        Returns:
            (best_action, best_utility)

        Raises:
            ValueError: if `actions` is empty.
        """
        if not actions:
            raise ValueError("decide() requires at least one candidate action")

        best_action: str = ""
        best_utility = float("-inf")

        for action, values in actions.items():
            benefit, cost = self._unpack(values)
            utility = self.calculate_utility(benefit, cost)

            if utility > best_utility:
                best_utility = utility
                best_action = action

        return best_action, best_utility

    @staticmethod
    def _unpack(values: ActionValue) -> Tuple[float, float]:
        if isinstance(values, ActionEstimate):
            return values.benefit, values.cost
        return values["benefit"], values["cost"]
