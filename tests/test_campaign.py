import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "campaigns" / "run_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_campaign", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_default_campaign_is_three_hours_of_paired_30_minute_runs(self):
        self.assertEqual(campaign.DEFAULT_BUDGET_SECONDS, 1800)
        self.assertEqual(len(campaign.PLAN), 6)
        self.assertEqual(campaign.SEEDS, (101, 202))
        labels = [label for label, _ in campaign.PLAN]
        self.assertEqual(len(labels), len(set(labels)))

    def test_fallback_campaign_only_uses_full_attention(self):
        self.assertEqual(campaign.BASE_ENV["OPENCLAW_FORCE_SDPA"], "1")
        self.assertEqual(campaign.BASE_ENV["WINDOW_PATTERN"], "LLLL")

    def test_reads_the_last_final_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text("\n".join([
                json.dumps({"kind": "step", "probe_train_loss": 4.1}),
                "not json",
                json.dumps({"kind": "final", "val_bpb": 2.31}),
            ]), encoding="utf-8")
            self.assertEqual(campaign.read_final(path)["val_bpb"], 2.31)

    def test_dashboard_env_reader_only_accepts_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.env"
            path.write_text(
                "DASHBOARD_INGEST_URL=https://example.invalid/api\n"
                "DASHBOARD_INGEST_TOKEN=secret\n"
                "IGNORED=unsafe\n", encoding="utf-8",
            )
            values = campaign.load_dashboard_env(path)
            self.assertEqual(set(values), {"DASHBOARD_INGEST_URL", "DASHBOARD_INGEST_TOKEN"})


if __name__ == "__main__":
    unittest.main()
