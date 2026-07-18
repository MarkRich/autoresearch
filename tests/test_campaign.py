import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "campaigns" / "run_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_campaign", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)

SUMMARY_PATH = Path(__file__).resolve().parents[1] / "campaigns" / "summarize_campaign.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location("summarize_campaign", SUMMARY_PATH)
summary = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
sys.modules[SUMMARY_SPEC.name] = summary
SUMMARY_SPEC.loader.exec_module(summary)


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
        self.assertEqual(campaign.BASE_ENV["AMP_DTYPE"], "bf16")
        self.assertEqual(campaign.BASE_ENV["GRAD_CLIP_NORM"], "0")
        self.assertEqual(campaign.BASE_ENV["MLP_KIND"], "relu_squared")
        self.assertEqual(campaign.BASE_ENV["UNCOUNTED_WARMUP_STEPS"], "0")
        self.assertEqual(campaign.BASE_ENV["FP32_ADAM_STATE"], "0")
        self.assertEqual(campaign.BASE_ENV["ATTN_RESIDUAL_MODE"], "none")
        self.assertEqual(campaign.BASE_ENV["ATTN_OUTPUT_GATE"], "none")
        self.assertEqual(campaign.BASE_ENV["FAILFAST_REGRESSION_MIN_RISE"], "0.50")
        self.assertEqual(campaign.BASE_ENV["FAILFAST_REGRESSION_PATIENCE_EVENTS"], "3")

    def test_all_checked_in_lane_plans_are_loadable(self):
        plans = MODULE_PATH.parent / "plans"
        loaded = {path.name: campaign.load_plan(path) for path in plans.glob("*.json")}
        self.assertIn("attnres-crossover.json", loaded)
        self.assertTrue(all(rounds for rounds in loaded.values()))

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

    def test_resolved_lane_environment_includes_defaults_and_overrides(self):
        resolved = campaign.resolved_lane_env({"DEVICE_BATCH_SIZE": "16", "AMP_DTYPE": "fp16"})
        self.assertEqual(resolved["DEVICE_BATCH_SIZE"], "16")
        self.assertEqual(resolved["AMP_DTYPE"], "fp16")
        self.assertEqual(resolved["WINDOW_PATTERN"], "LLLL")

    def test_lane_plan_rejects_unknown_environment_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"rounds": [{"lanes": [
                {"gpu": 0, "label": "unsafe", "env": {"DASHBOARD_INGEST_TOKEN": "secret"}},
                {"gpu": 1, "label": "control", "env": {}},
            ]}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported env keys"):
                campaign.load_plan(path)

    def test_campaign_summary_ranks_paired_means_not_lucky_minimum(self):
        rows = summary.summarize_runs([
            {"experiment": "noisy", "status": "ok", "val_bpb": 2.0, "gpu": 0, "seed": 101},
            {"experiment": "noisy", "status": "ok", "val_bpb": 4.0, "gpu": 1, "seed": 202},
            {"experiment": "stable", "status": "ok", "val_bpb": 2.8, "gpu": 0, "seed": 101},
            {"experiment": "stable", "status": "ok", "val_bpb": 2.9, "gpu": 1, "seed": 202},
            {"experiment": "failed", "status": "failed", "val_bpb": 1.0},
            {"experiment": "smoke", "status": "ok", "val_bpb": 2.7, "gpu": 0, "seed": 101},
        ])
        by_name = {row["experiment"]: row for row in rows}
        self.assertAlmostEqual(by_name["stable"]["mean_val_bpb"], 2.85)
        self.assertAlmostEqual(by_name["noisy"]["spread"], 2.0)
        self.assertTrue(by_name["stable"]["promotion_eligible"])
        self.assertFalse(by_name["smoke"]["promotion_eligible"])

    def test_launch_round_uses_campaign_specific_work_tag(self):
        lane = campaign.LaneSpec(gpu=0, seed=101, label="control", overrides={})
        fake_process = mock.Mock(pid=1234)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(campaign.subprocess, "Popen", return_value=fake_process) as popen:
            children = campaign.launch_round(
                "trace-me", 1, [lane], 60, "python", {}, "abc123", Path(tmp),
            )
            children[0].log_handle.close()
        self.assertEqual(popen.call_args.kwargs["env"]["CODEX_WORK_TAG"], "autoresearch:trace-me")


if __name__ == "__main__":
    unittest.main()
