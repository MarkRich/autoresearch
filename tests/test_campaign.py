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
        self.assertEqual(campaign.DEFAULT_MAX_GPU_TEMP_C, 88)
        self.assertEqual(len(campaign.PLAN), 6)
        self.assertEqual(campaign.SEEDS, (101, 202))
        labels = [label for label, _ in campaign.PLAN]
        self.assertEqual(len(labels), len(set(labels)))
        plan = campaign.paired_plan()
        self.assertEqual([lane.label for lane in plan[0]], ["control", "control"])
        self.assertEqual([lane.seed for lane in plan[0]], [101, 202])

    def test_fallback_campaign_only_uses_full_attention(self):
        self.assertEqual(campaign.BASE_ENV["OPENCLAW_FORCE_SDPA"], "1")
        self.assertEqual(campaign.BASE_ENV["WINDOW_PATTERN"], "LLLL")
        self.assertEqual(campaign.BASE_ENV["GRAD_CLIP_NORM"], "0")
        self.assertEqual(campaign.BASE_ENV["MLP_KIND"], "relu_squared")

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

    def test_temperature_parser_rejects_implausible_values(self):
        self.assertEqual(campaign.parse_gpu_temperature("84\n"), 84)
        with self.assertRaises(ValueError):
            campaign.parse_gpu_temperature("150\n")

    def test_loads_independent_lane_plan_and_stringifies_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"rounds": [{"lanes": [
                {"gpu": 0, "label": "batch8", "env": {"DEVICE_BATCH_SIZE": 8}},
                {"gpu": 1, "label": "batch16", "env": {"DEVICE_BATCH_SIZE": "16"}},
            ]}]}), encoding="utf-8")
            plan = campaign.load_plan(path)
            self.assertEqual([lane.label for lane in plan[0]], ["batch8", "batch16"])
            self.assertEqual([lane.overrides["DEVICE_BATCH_SIZE"] for lane in plan[0]], ["8", "16"])

    def test_lane_plan_rejects_unknown_environment_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"rounds": [{"lanes": [
                {"gpu": 0, "label": "unsafe", "env": {"DASHBOARD_INGEST_TOKEN": "secret"}},
                {"gpu": 1, "label": "control", "env": {}},
            ]}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported env keys"):
                campaign.load_plan(path)


if __name__ == "__main__":
    unittest.main()
