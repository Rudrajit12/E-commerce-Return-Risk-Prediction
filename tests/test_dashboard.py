"""Smoke tests for dashboard data and prediction logic."""
from __future__ import annotations

import unittest

from src.dashboard.core import (
    FEATURES,
    INTERVENTION_THRESHOLD,
    batch_decisions,
    decision,
    explain_order,
    load_artifacts,
)


class DashboardCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = load_artifacts()

    def test_batch_scoring(self):
        batch = batch_decisions(self.artifacts)
        self.assertEqual(len(batch), 10000)
        self.assertTrue(batch["return_probability"].between(0, 1).all())
        self.assertEqual(
            int((batch["recommended_action"] == "INTERVENE").sum()), 4224
        )

    def test_decision_policy(self):
        high = decision(INTERVENTION_THRESHOLD)
        low = decision(INTERVENTION_THRESHOLD - 0.001)
        self.assertEqual(high["recommended_action"], "INTERVENE")
        self.assertEqual(low["recommended_action"], "MONITOR")

    def test_local_explanation(self):
        row = self.artifacts["test"].iloc[0]
        order = {feature: row[feature] for feature in FEATURES}
        explanation = explain_order(self.artifacts, order)
        self.assertFalse(explanation.empty)
        self.assertIn("shap_value", explanation.columns)
        self.assertTrue(explanation["absolute_shap"].ge(0).all())


if __name__ == "__main__":
    unittest.main()
