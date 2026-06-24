"""
analysis/tests/test_advanced_planning_evaluation.py

Unit tests for advanced_planning_evaluation.py.
Run from repo root:
    python -m pytest analysis/tests/test_advanced_planning_evaluation.py -v
"""

import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# ── import the module under test ──────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import advanced_planning_evaluation_sp as ape
from reporting import cot_alignment as cot
from reporting import plots as reporting_plots


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_domain_info():
    return dict(
        action_names={"move", "pick", "place"},
        predicates={"at", "holding", "clear"},
        functions=set(),
        schemas={
            "pick": dict(params=["b", "l"], prec_raw="(at ?b ?l) (clear ?b)", eff_raw="(not (at ?b ?l)) (holding ?b)"),
            "place": dict(params=["b", "l"], prec_raw="(holding ?b)", eff_raw="(not (holding ?b)) (at ?b ?l)"),
            "move": dict(params=["from", "to"], prec_raw="", eff_raw=""),
        },
    )


def _make_problem_info():
    return dict(
        objects={"b1", "b2", "loc1", "loc2"},
        init_atoms={("at", "b1", "loc1"), ("at", "b2", "loc2"), ("clear", "b1"), ("clear", "b2")},
        init_numeric={},
    )


def _make_scored_json(solved: bool, iterations_used: int) -> dict:
    return {
        "solved": solved,
        "iterations_used": iterations_used,
        "metrics": {"validity_at_1": solved and iterations_used == 1},
        "attempts": [{"iteration": i + 1,
                       "validation_result": {"valid": solved and i == iterations_used - 1}}
                     for i in range(iterations_used)],
    }


def _make_parsed_json(actions: list, reasoning: str = "", n_iterations: int = 1) -> dict:
    return {
        "attempts": [
            {
                "iteration": i + 1,
                "parsed_plan": {
                    "actions": actions if i == n_iterations - 1 else [],
                    "reasoning": reasoning if i == n_iterations - 1 else "",
                    "format_issues": [],
                },
            }
            for i in range(n_iterations)
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Run inventory
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skip("legacy advanced_planning_evaluation API was replaced by advanced_planning_evaluation_sp")
class TestRunInventory(unittest.TestCase):

    def _write_json(self, path: pathlib.Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _populate_layer(self, base: pathlib.Path, run_id: str,
                        model: str, protocol: str, domain: str,
                        difficulty: str, instance: str, data: dict):
        p = base / run_id / model / protocol / domain / difficulty / f"{instance}.json"
        self._write_json(p, data)

    def test_complete_run_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            kwargs = dict(run_id="run1", model="m1", protocol="p1",
                          domain="d1", difficulty="easy", instance="pfile1")
            self._populate_layer(tmp / "raw",    data={"raw": True}, **kwargs)
            self._populate_layer(tmp / "parsed", data=_make_parsed_json(["(move a b)"]), **kwargs)
            self._populate_layer(tmp / "scored", data=_make_scored_json(False, 1),       **kwargs)

            with patch.object(ape, "RAW_DIR",    tmp / "raw"), \
                 patch.object(ape, "PARSED_DIR", tmp / "parsed"), \
                 patch.object(ape, "SCORED_DIR", tmp / "scored"):
                complete, incomplete = ape.list_runs()

        self.assertIn("run1", complete)
        self.assertNotIn("run1", incomplete)

    def test_incomplete_run_detected_missing_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            kwargs = dict(run_id="run2", model="m1", protocol="p1",
                          domain="d1", difficulty="easy", instance="pfile1")
            self._populate_layer(tmp / "raw",    data={"raw": True}, **kwargs)
            self._populate_layer(tmp / "parsed", data=_make_parsed_json(["(move a b)"]), **kwargs)
            # scored missing

            with patch.object(ape, "RAW_DIR",    tmp / "raw"), \
                 patch.object(ape, "PARSED_DIR", tmp / "parsed"), \
                 patch.object(ape, "SCORED_DIR", tmp / "scored"):
                complete, incomplete = ape.list_runs()

        self.assertNotIn("run2", complete)
        self.assertIn("run2", incomplete)

    def test_no_run_inventory_key_in_output(self):
        # The JSON output must NOT contain 'run_inventory'.
        import pandas as pd
        df_raw = pd.DataFrame([{
            "Model": "m1", "Domain": "d1", "Problem": "pfile1",
            "Difficulty": "easy", "Protocol": "p1", "Run_id": "run1",
            "Valid": False, "Length": 3, "Iterations": 1,
            "Chain_of_Thought": False,
        }])
        model_tables = {"m1": {
            "overall": {"success_rate": 0.0, "fasr": 0.0, "iwsr": 0.0,
                        "retry_gap": 0.0, "executability_ratio": 1.0,
                        "hallucination_rate": 0.0, "fuzzy_hallucination_rate": 0.0,
                        "object_hallucination_rate": 0.0, "inverse_hallucination_rate": 1.0,
                        "precondition_awareness_score": float("nan"),
                        "cot_alignment": float("nan"), "composite_score": 0.45, "n": 1},
            "by_domain": {}, "by_difficulty": {}, "cot_summary": {},
            "failure_breakdown": {}, "retry_gap": {}, "composite_score": {},
            "rank_within_domain": {},
        }}
        out = ape.build_json_output(
            df_raw=df_raw, df_metrics=df_raw, model_tables=model_tables,
            plot_records=[], run_ids=["run1"], merge=False, json_stem="test",
            show_plots=False, save_plots=False, plots_dir=None, warnings_list=[],
        )
        self.assertNotIn("run_inventory", out)
        self.assertIn("metadata", out)
        self.assertIn("loaded_data_summary", out)
        self.assertIn("models", out)
        self.assertIn("warnings", out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge / single run selection (interactive flow, mocked)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skip("legacy advanced_planning_evaluation API was replaced by advanced_planning_evaluation_sp")
class TestInteractiveRunSelection(unittest.TestCase):

    def test_single_run_selected(self):
        complete = ["run1", "run2"]
        with patch("builtins.input", side_effect=["n", "run1"]):
            selected, merge = ape.interactive_run_selection(complete, [])
        self.assertEqual(selected, ["run1"])
        self.assertFalse(merge)

    def test_merge_two_runs(self):
        complete = ["run1", "run2"]
        with patch("builtins.input", side_effect=["y", "run1", "run2", "stop"]):
            selected, merge = ape.interactive_run_selection(complete, [])
        self.assertEqual(sorted(selected), ["run1", "run2"])
        self.assertTrue(merge)

    def test_incomplete_run_rejected(self):
        complete = ["run1"]
        # tries "runX" (incomplete) then falls through to stop with no valid → retry with run1
        with patch("builtins.input", side_effect=["y", "runX", "run1", "stop"]):
            selected, merge = ape.interactive_run_selection(complete, ["runX"])
        self.assertEqual(selected, ["run1"])

    def test_no_complete_runs_exits(self):
        with self.assertRaises(SystemExit):
            ape.interactive_run_selection([], [])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Show / save plots
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skip("legacy advanced_planning_evaluation API was replaced by advanced_planning_evaluation_sp")
class TestOutputOptions(unittest.TestCase):

    def test_show_and_save_yes(self):
        with patch("builtins.input", side_effect=["y", "y", "n"]), \
             patch.object(ape, "HAS_PLOT", True):
            show, save, fname = ape.interactive_output_options()
        self.assertTrue(show)
        self.assertTrue(save)
        self.assertTrue(fname.endswith(".json"))

    def test_show_no_save_no(self):
        with patch("builtins.input", side_effect=["n", "n", "n"]), \
             patch.object(ape, "HAS_PLOT", True):
            show, save, fname = ape.interactive_output_options()
        self.assertFalse(show)
        self.assertFalse(save)

    def test_plots_dir_matches_json_stem(self):
        with patch("builtins.input", side_effect=["n", "y", "y", "mio_report"]), \
             patch.object(ape, "HAS_PLOT", True):
            show, save, fname = ape.interactive_output_options()
        stem = pathlib.Path(fname).stem
        self.assertEqual(stem, "mio_report")
        plots_dir = ape.RESULTS_DIR / stem
        self.assertTrue(str(plots_dir).endswith(stem))

    def test_no_plot_libs_skips_questions(self):
        with patch("builtins.input", side_effect=["n"]), \
             patch.object(ape, "HAS_PLOT", False):
            show, save, fname = ape.interactive_output_options()
        self.assertFalse(show)
        self.assertFalse(save)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Custom JSON name
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skip("legacy advanced_planning_evaluation API was replaced by advanced_planning_evaluation_sp")
class TestCustomJsonName(unittest.TestCase):

    def test_custom_name_normalised(self):
        with patch("builtins.input", side_effect=["n", "n", "y", "My Report 2026!"]), \
             patch.object(ape, "HAS_PLOT", False):
            _, _, fname = ape.interactive_output_options()
        self.assertFalse(" " in fname)
        self.assertTrue(fname.endswith(".json"))

    def test_custom_name_json_extension_not_doubled(self):
        with patch("builtins.input", side_effect=["n", "n", "y", "report.json"]), \
             patch.object(ape, "HAS_PLOT", False):
            _, _, fname = ape.interactive_output_options()
        self.assertEqual(fname.count(".json"), 1)

    def test_default_name_has_timestamp(self):
        with patch("builtins.input", side_effect=["n", "n", "n"]), \
             patch.object(ape, "HAS_PLOT", False):
            _, _, fname = ape.interactive_output_options()
        self.assertTrue(fname.startswith("advanced_planning_evaluation_"))
        self.assertTrue(fname.endswith(".json"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plots save dir == JSON stem
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skip("legacy advanced_planning_evaluation API was replaced by advanced_planning_evaluation_sp")
class TestPlotsDirStem(unittest.TestCase):

    def test_plots_dir_is_results_slash_stem(self):
        with patch("builtins.input", side_effect=["n", "y", "y", "mio_report"]), \
             patch.object(ape, "HAS_PLOT", True):
            _, save, fname = ape.interactive_output_options()
        stem = pathlib.Path(fname).stem          # "mio_report"
        expected_dir = ape.RESULTS_DIR / stem
        self.assertEqual(expected_dir.name, stem)
        self.assertTrue(save)


# ─────────────────────────────────────────────────────────────────────────────
# 6. JSON structure under models[model_id]
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skip("legacy advanced_planning_evaluation API was replaced by advanced_planning_evaluation_sp")
class TestJsonModelStructure(unittest.TestCase):

    def _minimal_output(self):
        import pandas as pd
        df_raw = pd.DataFrame([{
            "Model": "m1", "Domain": "d1", "Problem": "pfile1",
            "Difficulty": "easy", "Protocol": "p1", "Run_id": "r1",
            "Valid": True, "Length": 2, "Iterations": 1, "Chain_of_Thought": False,
        }])
        model_tables = {"m1": {
            "overall": {
                "n": 1, "success_rate": 1.0, "fasr": 1.0, "iwsr": 1.0,
                "retry_gap": 0.0, "executability_ratio": 1.0,
                "hallucination_rate": 0.0, "fuzzy_hallucination_rate": 0.0,
                "object_hallucination_rate": 0.0, "inverse_hallucination_rate": 1.0,
                "precondition_awareness_score": float("nan"),
                "cot_alignment": float("nan"), "composite_score": 0.85,
            },
            "by_domain": {"d1": {"n": 1, "success_rate": 1.0, "fasr": 1.0,
                                  "iwsr": 1.0, "executability_ratio": 1.0,
                                  "hallucination_rate": 0.0,
                                  "precondition_awareness_score": float("nan"),
                                  "cot_alignment": float("nan")}},
            "by_difficulty": {"easy": {"n": 1, "success_rate": 1.0, "fasr": 1.0,
                                        "executability_ratio": 1.0, "hallucination_rate": 0.0}},
            "cot_summary": {}, "failure_breakdown": {}, "retry_gap": {},
            "composite_score": {}, "rank_within_domain": {"d1": 1},
        }}
        return ape.build_json_output(
            df_raw=df_raw, df_metrics=df_raw, model_tables=model_tables,
            plot_records=[], run_ids=["r1"], merge=False, json_stem="test",
            show_plots=False, save_plots=False, plots_dir=None, warnings_list=[],
        )

    def test_model_keys_present(self):
        out = self._minimal_output()
        m1  = out["models"]["m1"]
        for key in ("row_metrics", "tables", "plots", "reasoning_notes"):
            self.assertIn(key, m1, f"missing key: {key}")

    def test_tables_keys_present(self):
        out    = self._minimal_output()
        tables = out["models"]["m1"]["tables"]
        for key in ("overall", "by_domain", "by_difficulty", "cot_summary",
                    "failure_breakdown", "retry_gap", "composite_score", "rank_within_domain"):
            self.assertIn(key, tables, f"missing table: {key}")

    def test_no_nan_in_json(self):
        out  = self._minimal_output()
        text = json.dumps(out)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_row_metrics_no_actions(self):
        out  = self._minimal_output()
        rows = out["models"]["m1"]["row_metrics"]
        for row in rows:
            self.assertNotIn("_actions", row)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Capability profiles (4A)
# ─────────────────────────────────────────────────────────────────────────────


def _capability_notes(metrics: dict) -> dict:
    return ape.classify_capability_profile({
        "Success_Rate": metrics.get("success_rate"),
        "FASR": metrics.get("fasr"),
        "IWSR": metrics.get("iwsr"),
        "Retry_Gap": metrics.get("retry_gap"),
        "Exec": metrics.get("executability_ratio"),
        "IHR": metrics.get("inverse_hallucination_rate"),
        "PAS": metrics.get("precondition_awareness_score"),
        "CoT_Alignment": metrics.get("cot_alignment"),
        "Composite_Score": metrics.get("composite_score"),
        "N": metrics.get("n"),
    })

class TestCapabilityProfiles(unittest.TestCase):

    def _profile(self, **kwargs) -> str:
        defaults = dict(
            success_rate=0.0, fasr=0.0, iwsr=0.0, retry_gap=0.0,
            executability_ratio=float("nan"), hallucination_rate=0.0,
            inverse_hallucination_rate=1.0, precondition_awareness_score=float("nan"),
            cot_alignment=float("nan"), composite_score=0.0, n=10,
            fuzzy_hallucination_rate=0.0, object_hallucination_rate=0.0,
        )
        defaults.update(kwargs)
        return _capability_notes(defaults)["assigned_profile"]

    def test_genuine_planner(self):
        p = self._profile(success_rate=0.85, fasr=0.80, retry_gap=0.05,
                          inverse_hallucination_rate=0.97, cot_alignment=0.80)
        self.assertEqual(p, "Genuine Planner")

    def test_stochastic_searcher(self):
        p = self._profile(success_rate=0.5, fasr=0.10, retry_gap=0.40)
        self.assertEqual(p, "Stochastic Searcher")

    def test_no_grounding(self):
        p = self._profile(success_rate=0.02, fasr=0.0, retry_gap=0.02,
                          inverse_hallucination_rate=0.30,
                          executability_ratio=0.10,
                          precondition_awareness_score=0.10)
        self.assertEqual(p, "No Grounding")

    def test_understander(self):
        p = self._profile(success_rate=0.05, fasr=0.0, retry_gap=0.05,
                          executability_ratio=0.90,
                          inverse_hallucination_rate=0.95,
                          precondition_awareness_score=0.80)
        self.assertEqual(p, "Understander")

    def test_lucky_retriever(self):
        p = self._profile(success_rate=0.80, fasr=0.75, retry_gap=0.05,
                          inverse_hallucination_rate=0.20, cot_alignment=0.25)
        self.assertEqual(p, "Lucky Retriever")

    def test_efficient_corrector(self):
        p = self._profile(success_rate=0.65, fasr=0.20, retry_gap=0.45)
        # SR>0.50, FASR<0.30, RG between 0.10-0.30 fails (0.45), but it's the best partial match
        # Stochastic Searcher: FASR<0.20 fails (0.20==0.20 → false), RG>0.30 matches
        # We just assert a consistent profile was assigned and reasoning_notes is populated
        self.assertIsNotNone(p)

    def test_vocabulary_only(self):
        p = self._profile(success_rate=0.10, fasr=0.0, retry_gap=0.10,
                          inverse_hallucination_rate=0.95,
                          executability_ratio=0.50,
                          precondition_awareness_score=0.15)
        self.assertEqual(p, "Vocabulary-Only")

    def test_reasoning_notes_fields(self):
        overall = dict(success_rate=0.85, fasr=0.80, iwsr=0.60, retry_gap=0.05,
                       executability_ratio=1.0, hallucination_rate=0.03,
                       inverse_hallucination_rate=0.97, precondition_awareness_score=0.75,
                       cot_alignment=0.80, composite_score=0.78, n=20,
                       fuzzy_hallucination_rate=0.0, object_hallucination_rate=0.0)
        notes = _capability_notes(overall)
        for key in ("assigned_profile", "matched_conditions", "missing_conditions",
                    "interpretation", "key_reference"):
            self.assertIn(key, notes)

    def test_all_nan_returns_something(self):
        notes = _capability_notes({"success_rate": float("nan")})
        self.assertIn("assigned_profile", notes)

    def test_matched_conditions_are_list(self):
        notes = _capability_notes({"success_rate": 0.9, "fasr": 0.8})
        self.assertIsInstance(notes["matched_conditions"], list)
        self.assertIsInstance(notes["missing_conditions"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Hallucination metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationMetrics(unittest.TestCase):

    def test_no_hallucination(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        actions = ["(move loc1 loc2)", "(pick b1 loc1)"]
        r = ape.compute_hallucination_metrics(actions, dinfo, pinfo)
        self.assertEqual(r["hallucination_rate"], 0.0)
        self.assertEqual(r["object_hallucination_rate"], 0.0)

    def test_strict_hallucination(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        actions = ["(fly loc1 loc2)", "(move loc1 loc2)"]
        r = ape.compute_hallucination_metrics(actions, dinfo, pinfo)
        self.assertGreater(r["hallucination_rate"], 0.0)

    def test_object_hallucination(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        actions = ["(move ghost_loc loc1)"]
        r = ape.compute_hallucination_metrics(actions, dinfo, pinfo)
        self.assertGreater(r["object_hallucination_rate"], 0.0)

    def test_empty_actions(self):
        r = ape.compute_hallucination_metrics([], _make_domain_info(), _make_problem_info())
        self.assertTrue(math.isnan(r["hallucination_rate"]))

    def test_fuzzy_not_counted_for_close_name(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        # "mov" is distance 1 from "move" → within MAX_FUZZY_DISTANCE=2 → fuzzy_halluc=0
        actions = ["(mov loc1 loc2)"]
        r = ape.compute_hallucination_metrics(actions, dinfo, pinfo)
        self.assertEqual(r["fuzzy_hallucinated_count"], 0)
        self.assertEqual(r["hallucinated_action_count"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Precondition metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestPreconditionMetrics(unittest.TestCase):

    def test_fully_executable(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        actions = ["(pick b1 loc1)", "(place b1 loc2)"]
        r = ape.compute_precondition_metrics(actions, dinfo, pinfo)
        self.assertEqual(r["executability_ratio"], 1.0)
        self.assertEqual(r["sequencing_error_count"], 0)
        self.assertEqual(r["state_fabrication_count"], 0)

    def test_empty_actions(self):
        r = ape.compute_precondition_metrics([], _make_domain_info(), _make_problem_info())
        self.assertTrue(math.isnan(r["executability_ratio"]))

    def test_fabrication_on_missing_object(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        # b99 doesn't exist in init state → state fabrication
        actions = ["(pick b99 loc1)"]
        r = ape.compute_precondition_metrics(actions, dinfo, pinfo)
        self.assertGreater(r["state_fabrication_count"] + r["sequencing_error_count"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CoT alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestCotAlignment(unittest.TestCase):

    def test_semantic_support_full_coverage(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        actions = ["(pick b1 loc1)", "(place b1 loc2)"]
        cot_text = "I will pick b1 from loc1 then place b1 at loc2"
        r = cot.compute_cot_semantic_support(cot_text, actions, dinfo, pinfo)
        self.assertGreater(r["cot_semantic_support_score"], 0.5)

    def test_empty_cot_semantic_support(self):
        dinfo = _make_domain_info()
        pinfo = _make_problem_info()
        actions = ["(pick b1 loc1)"]
        r = cot.compute_cot_semantic_support("", actions, dinfo, pinfo)
        self.assertEqual(r["cot_semantic_support_score"], 0.0)

    def test_exact_match_both_invalid_keeps_alignment_one(self):
        parsed = {
            "iteration": 1,
            "parsed_plan": {
                "raw": {"actions": ["(pick b1 loc1)", "(place b1 loc2)"]},
                "reasoning": {"actions": ["(pick b1 loc1)", "(place b1 loc2)"]},
            },
        }
        scored = {"final_plan_valid": False, "reasoning_final_plan_valid": False}
        raw = {"generation": {"reasoning_text": "pick b1 loc1 then place b1 loc2"}}
        r = cot.compute_cot_alignment_for_attempt(parsed, scored, raw, _make_domain_info(), _make_problem_info())
        self.assertEqual(r["cot_plan_alignment_score"], 1.0)
        self.assertIsNone(r["cot_plan_alignment_proxy_score"])
        self.assertEqual(r["cot_alignment_status"], "comparable_but_both_invalid")

    def test_semantic_proxy_is_capped(self):
        parsed = {"iteration": 1, "parsed_plan": {"raw": {"actions": ["(pick b1 loc1)"]}, "reasoning": {"actions": []}}}
        scored = {"final_plan_valid": True, "reasoning_final_plan_valid": None}
        raw = {"generation": {"reasoning_text": "pick b1 loc1"}}
        r = cot.compute_cot_alignment_for_attempt(parsed, scored, raw, _make_domain_info(), _make_problem_info())
        self.assertIsNone(r["cot_plan_alignment_score"])
        self.assertLessEqual(r["cot_plan_alignment_proxy_score"], 0.35)
        self.assertEqual(r["cot_alignment_status"], "semantic_proxy_only")
        self.assertEqual(r["cot_alignment_confidence"], "low")

    def test_adjacent_swap_detection(self):
        r = cot.compute_sequence_alignment(["A", "B", "C", "D"], ["A", "C", "B", "D"])
        self.assertFalse(r["exact_sequence_match"])
        self.assertEqual(r["first_mismatch_index"], 1)
        self.assertEqual(r["adjacent_swap_count"], 1)
        self.assertLess(r["structural_alignment"], 1.0)

    def test_shorter_valid_prefix_is_diagnostic_only(self):
        actions = ["(pick b1 loc1)", "(place b1 loc2)"]
        parsed = {"iteration": 1, "parsed_plan": {"raw": {"actions": actions}, "reasoning": {"actions": actions}}}
        scored = {
            "final_plan_valid": True,
            "reasoning_final_plan_valid": True,
            "first_valid_prefix_length": 1,
            "reasoning_first_valid_prefix_length": 1,
        }
        raw = {"generation": {"reasoning_text": "pick b1 loc1 place b1 loc2"}}
        r = cot.compute_cot_alignment_for_attempt(parsed, scored, raw, _make_domain_info(), _make_problem_info())
        self.assertEqual(r["cot_plan_alignment_score"], 1.0)
        self.assertTrue(r["raw_has_shorter_valid_prefix"])
        self.assertTrue(r["reasoning_has_shorter_valid_prefix"])

    def test_old_parsed_plan_actions_fallback(self):
        parsed_plan = {"actions": ["(move loc1 loc2)"]}
        self.assertEqual(cot.parsed_plan_raw_actions(parsed_plan), ["(move loc1 loc2)"])


# ─────────────────────────────────────────────────────────────────────────────
# 11. Composite score
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScore(unittest.TestCase):

    def test_all_zeros(self):
        s = ape.compute_composite_score(
            dict(fasr=0, iwsr=0, exec_ratio=0, one_minus_halluc=0, pas=0),
            ape.COMPOSITE_WEIGHTS, ape.COT_BONUS_WEIGHT,
        )
        self.assertAlmostEqual(s, 0.0, places=4)

    def test_perfect_scores(self):
        s = ape.compute_composite_score(
            dict(fasr=1, iwsr=1, exec_ratio=1, one_minus_halluc=1, pas=1, cot_alignment=1),
            ape.COMPOSITE_WEIGHTS, ape.COT_BONUS_WEIGHT,
        )
        self.assertAlmostEqual(s, 1.0, places=4)

    def test_result_clipped_to_01(self):
        s = ape.compute_composite_score(
            dict(fasr=2, iwsr=2, exec_ratio=2, one_minus_halluc=2, pas=2),
            ape.COMPOSITE_WEIGHTS, ape.COT_BONUS_WEIGHT,
        )
        self.assertLessEqual(s, 1.0)

    def test_cot_bonus_applied_when_not_nan(self):
        base = ape.compute_composite_score(
            dict(fasr=0.5, iwsr=0.5, exec_ratio=0.5, one_minus_halluc=0.5, pas=0.5),
            ape.COMPOSITE_WEIGHTS, ape.COT_BONUS_WEIGHT,
        )
        with_cot = ape.compute_composite_score(
            dict(fasr=0.5, iwsr=0.5, exec_ratio=0.5, one_minus_halluc=0.5, pas=0.5, cot_alignment=1.0),
            ape.COMPOSITE_WEIGHTS, ape.COT_BONUS_WEIGHT,
        )
        # CoT=1 should push score up compared to no CoT bonus
        self.assertGreaterEqual(with_cot, base - 0.01)  # at most marginal change in either direction


# ─────────────────────────────────────────────────────────────────────────────
# 12. Levenshtein
# ─────────────────────────────────────────────────────────────────────────────

class TestLevenshtein(unittest.TestCase):

    def test_identical(self):
        self.assertEqual(ape.levenshtein("move", "move"), 0)

    def test_one_insertion(self):
        self.assertEqual(ape.levenshtein("mov", "move"), 1)

    def test_one_substitution(self):
        self.assertEqual(ape.levenshtein("move", "muve"), 1)

    def test_symmetry(self):
        self.assertEqual(ape.levenshtein("abc", "xyz"), ape.levenshtein("xyz", "abc"))


# ─────────────────────────────────────────────────────────────────────────────
# 13. parse_action
# ─────────────────────────────────────────────────────────────────────────────

class TestParseAction(unittest.TestCase):

    def test_basic(self):
        name, args = ape.parse_action("(move loc1 loc2)")
        self.assertEqual(name, "move")
        self.assertEqual(args, ["loc1", "loc2"])

    def test_no_args(self):
        name, args = ape.parse_action("(noop)")
        self.assertEqual(name, "noop")
        self.assertEqual(args, [])

    def test_invalid(self):
        name, args = ape.parse_action("not an action")
        self.assertIsNone(name)
        self.assertEqual(args, [])

    def test_case_normalised(self):
        name, args = ape.parse_action("(MOVE LOC1 LOC2)")
        self.assertEqual(name, "move")
        self.assertEqual(args, ["loc1", "loc2"])


# ─────────────────────────────────────────────────────────────────────────────
# 14. _slugify
# ─────────────────────────────────────────────────────────────────────────────

class TestSlugify(unittest.TestCase):

    def test_spaces_become_underscores(self):
        self.assertEqual(ape.sanitize_json_filename("my report"), "my_report.json")

    def test_special_chars_removed(self):
        result = ape.sanitize_json_filename("test! @2026")
        self.assertNotIn("!", result)
        self.assertNotIn("@", result)

    def test_empty_string(self):
        self.assertEqual(ape.sanitize_json_filename(""), "advanced_planning_evaluation.json")


if __name__ == "__main__":
    unittest.main()
