import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .local_experiment import (PROFILE, make_plan, prune_codex_trust_sections, published_summary,
                               metrics, run_environment, summarize, doctor)


class LocalExperimentTests(unittest.TestCase):
    def test_exact_paired_panel_and_generalization_mix(self):
        plan = make_plan()
        self.assertEqual(plan['episode_count'], 100)
        self.assertEqual(len({r['case']['case_id'] for r in plan['runs']}), 50)
        for left, right in zip(plan['runs'][::2], plan['runs'][1::2]):
            self.assertEqual(left['identity'], right['identity'])
            self.assertNotEqual(left['method'], right['method'])
        for task in ('fold_clothes', 'pack_objects_into_box', 'arrange_largest_number'):
            cases = make_plan('gpt_only', task)['runs']
            self.assertEqual([r['case']['variant'] for r in cases],
                             ['standard', 'standard', 'random', 'random', 'random'])

    def test_unknown_or_excluded_case_rejected(self):
        with self.assertRaises(ValueError):
            make_plan(case_id='fold_clothes__standard__g0__l2')
        with self.assertRaises(ValueError):
            make_plan(task='typo')

    def test_reference_denominators(self):
        result = published_summary()['methods']
        self.assertEqual(result['gpt_only']['scored_cases'], 48)
        self.assertEqual(result['gpt_only']['success_rate'], .26)
        self.assertAlmostEqual(result['gpt_only']['mean_score_100'], 37.8125)
        self.assertEqual(result['pi05_plus_gpt']['success_rate'], .48)
        self.assertAlmostEqual(result['pi05_plus_gpt']['mean_score_100'], 62.6)

    def test_unknown_outcome_is_not_a_failure_or_zero_score(self):
        result = metrics([dict(success=True, score=.5), dict(success=None, score=None)])
        self.assertIsNone(result['success_rate'])
        self.assertEqual(result['mean_score_100'], 50)
        self.assertEqual(result['scored_cases'], 1)

    def test_runtime_task_seed_and_unlimited_decisions(self):
        run = make_plan('gpt_only', case_id='fold_clothes__random__g0__l2')['runs'][0]
        args = SimpleNamespace(work_dir=Path('/tmp/experiment'), source=Path('/tmp/source'),
            sim_python=Path('/tmp/python'), codex='/tmp/codex', graphics_mode='system',
            sim_gpu='0', policy_gpu='0', sim_port=19113, policy_port=18830)
        with patch.dict('os.environ', {'ROLLOUT_EVAL_SEED': '99', 'ROLLOUT_MAX_SECONDS': '1'}):
            env = run_environment(args, run, '/tmp/case.json', Path('/tmp/private/config.toml'), '/tmp/auth', 7)
        self.assertEqual(env['TASK'], 'fold_clothes_random')
        self.assertEqual(env['SEED'], '2')
        self.assertEqual(env['EVAL_SEED'], '0')
        self.assertEqual(env['MAX_DECISIONS'], '0')
        self.assertEqual(env['ROLLOUT_ACCOUNT_LOCK_FD'], '7')
        self.assertNotIn('ROLLOUT_MAX_SECONDS', env)

    def test_missing_archives_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'plan.json').write_text(json.dumps(make_plan('gpt_only', task='build_tower')))
            result = summarize(path)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['methods']['gpt_only']['success_rate'])
        self.assertEqual(result['methods']['gpt_only']['cases'], 5)

    def test_mismatched_archive_cannot_count_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            plan = make_plan('gpt_only', case_id='build_tower__standard__g0__l0')
            (path/'plan.json').write_text(json.dumps(plan))
            run = plan['runs'][0]
            archive = path/'results'/('gpt_only__'+run['case']['case_id'])
            (archive/'sim').mkdir(parents=True)
            (archive/'artifact_manifest.json').write_text(json.dumps(dict(status='verified',
                checks=dict(paired_evaluation_identity_matches=True))))
            outcome = dict(complete=True, native_success=True, native_score=1,
                           evaluation_case={'case_id': 'wrong'})
            target = archive/'sim/evaluation_outcome.json'
            target.write_text(json.dumps(outcome))
            self.assertFalse(summarize(path)['complete'])
            outcome['evaluation_case'] = run['identity']
            target.write_text(json.dumps(outcome))
            result = summarize(path)
            self.assertTrue(result['complete'])
            self.assertEqual(result['methods']['gpt_only']['success_rate'], 1)

    def test_direct_preflight_does_not_require_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(codex='codex', work_dir=Path(tmp), source=None,
                                   sim_python=None, sim_gpu='0', policy_gpu='0')
            with patch('hybrid_rollout.robodojo.local_experiment.probe', return_value=(False, 'unavailable')):
                result = doctor(args, make_plan('gpt_only', task='build_tower'))
        self.assertFalse(result['ready'])
        self.assertFalse(any(c['name'].startswith(('openpi', 'checkpoint')) for c in result['checks']))

    def test_codex_auto_trust_sections_are_pruned_for_relaunch(self):
        from .codex_backend.profiles import config_text, initialize
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            initialize(PROFILE, path)
            config = path/'private/auth_profiles'/PROFILE/'codex_home/config.toml'
            config.write_text(config_text(PROFILE)
                              + '\n[projects."/tmp/repro"]\ntrust_level = "trusted"\n')
            with self.assertRaises(ValueError):
                initialize(PROFILE, path)
            prune_codex_trust_sections(path)
            initialize(PROFILE, path)
            self.assertEqual(config.read_text(), config_text(PROFILE))

    def test_unexpected_config_changes_are_never_pruned(self):
        from .codex_backend.profiles import config_text, initialize
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            initialize(PROFILE, path)
            config = path/'private/auth_profiles'/PROFILE/'codex_home/config.toml'
            config.write_text(config_text(PROFILE).replace('xhigh', 'high'))
            prune_codex_trust_sections(path)
            with self.assertRaises(ValueError):
                initialize(PROFILE, path)


if __name__ == '__main__':
    unittest.main()
