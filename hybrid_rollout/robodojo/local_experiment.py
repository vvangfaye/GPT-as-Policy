"""Local, serial reproduction of the report's exact 50 paired RoboDojo cases.

Planning and published-result verification use only the Python standard library.
Execution reuses the released simulator, controller and artifact verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess

from .evaluation import case_identity, read_panel, verify_assets
from .io import write_json

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / 'hybrid_rollout/robodojo/eval_panels/robodojo_panel60_v1.json'
RESULTS = ROOT / 'public_results/evaluation_cases.json'
METHODS = {'gpt_only': 'gpt', 'pi05_plus_gpt': 'mix'}
PROFILE = 'codex_a'


def report_cases():
    """Validate paired identities before selecting any runs; ignore outcomes."""
    panel = read_panel(PANEL)
    published = json.loads(RESULTS.read_text())['cases']
    groups = {m: [r for r in published if r['method'] == m] for m in METHODS.values()}
    left, right = groups['mix'], groups['gpt']
    for rows in groups.values():
        if len(rows) != 50 or len({r['case_id'] for r in rows}) != 50:
            raise ValueError('Expected 50 distinct published cases per method')
        if set(Counter(r['task'] for r in rows).values()) != {5}:
            raise ValueError('Expected five cases per task')
    if {r['case_id']: r['seeds'] for r in left} != {r['case_id']: r['seeds'] for r in right}:
        raise ValueError('Published method seeds differ')
    definitions = {c['case_id']: c for c in panel['cases']}
    selected = []
    for row in left:
        case = definitions[row['case_id']]
        if row['task'] != case['task'] or row['variant'] != case['variant'] or any(
                case[k] != v for k, v in row['seeds'].items()):
            raise ValueError('Published case differs from frozen panel')
        selected.append(case)
    return panel, selected


def make_plan(method='both', task=None, case_id=None):
    panel, cases = report_cases()
    cases = [c for c in cases if (task is None or c['task'] == task)
             and (case_id is None or c['case_id'] == case_id)]
    if not cases:
        raise ValueError('No published cases match the selection')
    methods = list(METHODS) if method == 'both' else [method]
    return dict(schema='robodojo.local_plan.v1', model='gpt-6-astra', effort='xhigh',
                panel_sha256=panel['panel_sha256'], episode_count=len(cases)*len(methods),
                runs=[dict(method=m, case=c, identity=case_identity(panel, c))
                      for c in cases for m in methods])


def metrics(rows):
    scores = [r['score'] for r in rows if type(r.get('score')) in (int, float)
              and math.isfinite(r['score'])]
    known = [r for r in rows if type(r.get('success')) is bool]
    return dict(cases=len(rows), success_known=len(known),
                successes=sum(r['success'] for r in known),
                success_rate=sum(r['success'] for r in known)/len(rows)
                if rows and len(known) == len(rows) else None,
                scored_cases=len(scores), mean_score_100=100*sum(scores)/len(scores) if scores else None)


def published_summary():
    report_cases()
    rows = json.loads(RESULTS.read_text())['cases']
    return dict(source='published_reference_not_new_experiments',
                methods={m: metrics([r for r in rows if r['method'] == label])
                         for m, label in METHODS.items()})


def probe(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=ROOT)
        return p.returncode == 0, (p.stdout + p.stderr).strip()[-2000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def doctor(args, plan):
    checks = []

    def check(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))

    ok, detail = probe([args.codex, '--version'])
    check('codex', ok, detail)
    ok, detail = probe(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'])
    check('nvidia_driver', ok, detail)
    if ok:
        indices = {line.split(',')[0].strip() for line in detail.splitlines()}
        check('sim_gpu', args.sim_gpu in indices, args.sim_gpu)
        if any(r['method'] == 'pi05_plus_gpt' for r in plan['runs']):
            check('policy_gpu', args.policy_gpu in indices, args.policy_gpu)
    check('robodojo_source', args.source is not None and (args.source/'Assets').is_dir(),
          args.source or 'Set --source to the RoboDojo checkout with Assets')
    if args.source and (args.source/'Assets').is_dir():
        try:
            panel = read_panel(PANEL)
            verify_assets(panel, args.source, [r['case'] for r in plan['runs']])
            check('frozen_assets', True, panel['native_source_commit'])
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            check('frozen_assets', False, exc)
    if args.sim_python:
        ok, detail = probe([str(args.sim_python), '-c',
            'import importlib.metadata as m; v=m.version("isaacsim"); '
            'assert v.startswith("5.1."), v; import numpy, scipy, PIL, yaml; '
            'from hybrid_rollout.robodojo.codex_backend import validate; print(v)'])
        check('isaacsim_5_1_python', ok, detail)
    else:
        check('isaacsim_5_1_python', False, 'Set --sim-python')
    if any(r['method'] == 'pi05_plus_gpt' for r in plan['runs']):
        for name, path in [('openpi_source', args.openpi_source/'src/openpi' if args.openpi_source else None),
                           ('checkpoint_params', args.checkpoint/'params' if args.checkpoint else None),
                           ('normalizer', args.checkpoint/'assets/arx_x5_sim/norm_stats.json' if args.checkpoint else None)]:
            check(name, path is not None and path.exists(), path or 'Not configured')
        if args.openpi_python:
            ok, detail = probe([str(args.openpi_python), '-c', 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'])
            check('openpi_jax_gpu', ok, detail)
        else:
            check('openpi_jax_gpu', False, 'Set --openpi-python')
    from .codex_backend.profiles import validate_credential
    try:
        validate_credential(PROFILE, args.work_dir)
        check('isolated_codex_login', True, 'Credential format checked; model access not tested')
    except ValueError as exc:
        check('isolated_codex_login', False, str(exc) + '; run the login subcommand')
    return dict(ready=all(c['ok'] for c in checks), checks=checks,
                note='No simulation or paid model inference performed; model access and rendering need a smoke run.')


def run_environment(args, run, case_file, config, home, lock_fd):
    env = dict(os.environ)
    # Do not inherit another rollout's case identity or limits.
    for key in list(env):
        if key.startswith('ROLLOUT_'):
            env.pop(key)
    case = run['case']
    run_id = run['method'] + '__' + case['case_id']
    values = dict(CODE_ROOT=ROOT, RUNTIME_ROOT=args.work_dir,
        RESULTS_ROOT=args.work_dir/'results', RUN_ID=run_id,
        ROBODOJO_SOURCE=args.source, ROBODOJO_PYTHON=args.sim_python,
        CODEX_BIN=args.codex, ROLLOUT_SHARED_ROOT=args.work_dir,
        ROLLOUT_AUTH_PROFILE=PROFILE, ROLLOUT_ACCOUNT_LOCK_FD=lock_fd,
        ROLLOUT_CODEX_HOME_DIR=home, ROLLOUT_CODEX_CONFIG=config,
        ROLLOUT_CODEX_STATE_DIR=config.parent, ROLLOUT_CREDENTIAL_SOURCE='chatgpt_managed_file',
        ROLLOUT_GRAPHICS_MODE=args.graphics_mode,
        ROLLOUT_EVALUATION_METHOD=run['method'], ROLLOUT_EVAL_MANIFEST=PANEL,
        ROLLOUT_EVAL_MANIFEST_SHA256=run['identity']['panel_sha256'],
        ROLLOUT_CASE_FILE=case_file, TASK=case['runtime_task'], SEED=case['layout_id'],
        EVAL_SEED=case['eval_seed'], MAX_DECISIONS=0,
        SIM_GPU=args.sim_gpu, POLICY_GPU=args.policy_gpu,
        SIM_PORT=args.sim_port, POLICY_PORT=args.policy_port,
        TASK_ROOT=config.parent, ROBODOJO_RUN_ID=run_id)
    if run['method'] == 'pi05_plus_gpt':
        values.update(OPENPI_SOURCE=args.openpi_source, OPENPI_PYTHON=args.openpi_python,
                      CHECKPOINT=args.checkpoint, OPENPI_DATA_HOME=args.work_dir/'openpi_cache')
    env.update({k: str(v) for k, v in values.items()})
    return env


def prune_codex_trust_sections(work_dir):
    """Codex CLI appends [projects."..."] trust blocks to the profile config on each
    run; strip them so re-running in the same work-dir still passes initialize()."""
    config = work_dir/'private/auth_profiles'/PROFILE/'codex_home/config.toml'
    if not config.exists():
        return
    from .codex_backend.profiles import config_text
    kept, in_projects = [], False
    for line in config.read_text().splitlines(keepends=True):
        if line.lstrip().startswith('['):
            in_projects = line.lstrip().startswith('[projects.')
        if not in_projects:
            kept.append(line)
    if ''.join(kept).rstrip('\n') == config_text(PROFILE).rstrip('\n'):
        fd = os.open(config, os.O_WRONLY | os.O_TRUNC)
        with os.fdopen(fd, 'w') as stream:
            stream.write(config_text(PROFILE))


def execute(args, plan):
    from .codex_backend.profiles import account_lock, config_text, initialize
    status = doctor(args, plan)
    if not status['ready']:
        print(json.dumps(status, indent=2))
        raise ValueError('Preflight failed; no experiment launched')
    prune_codex_trust_sections(args.work_dir)
    home = initialize(PROFILE, args.work_dir)
    with account_lock(PROFILE, args.work_dir) as fd:
        for run in plan['runs']:
            name = run['method'] + '__' + run['case']['case_id']
            if (args.work_dir/'results'/name).exists():
                raise ValueError(f'Archive already exists: {name}; use a new --work-dir, never overwrite outcomes')
        plan_path = args.work_dir/'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('Work directory belongs to a different plan; use a new --work-dir')
        write_json(plan_path, plan)
        for run in plan['runs']:
            name = run['method'] + '__' + run['case']['case_id']
            private = args.work_dir/'private_runtime'/name
            private.mkdir(parents=True, mode=0o700, exist_ok=True)
            config = private/'config.toml'
            config.write_text(config_text(PROFILE))
            case_file = private/'case.json'
            write_json(case_file, dict(case=run['case'], identity=run['identity']))
            env = run_environment(args, run, case_file, config, home, fd)
            print(f'Launching {name}', flush=True)
            process = subprocess.Popen(['bash', str(ROOT/'hybrid_rollout/robodojo/run_local.sh')],
                                       env=env, pass_fds=(fd,))
            def terminate(signum, frame):
                process.send_signal(signum)
            previous = {s: signal.signal(s, terminate) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                code = process.wait()
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
            if code:
                raise ValueError(f'{name} exited {code}; stopped without replacing/retrying this case')


def summarize(work_dir):
    plan = json.loads((work_dir/'plan.json').read_text())
    rows = []
    for run in plan['runs']:
        archive = work_dir/'results'/(run['method']+'__'+run['case']['case_id'])
        outcome_file = archive/'sim/evaluation_outcome.json'
        artifact_file = archive/'artifact_manifest.json'
        outcome = json.loads(outcome_file.read_text()) if outcome_file.exists() else {}
        artifact = json.loads(artifact_file.read_text()) if artifact_file.exists() else {}
        complete = (outcome.get('complete') is True and artifact.get('status') == 'verified'
                    and outcome.get('evaluation_case') == run['identity']
                    and artifact.get('checks', {}).get('paired_evaluation_identity_matches') is True)
        rows.append(dict(method=run['method'], case_id=run['case']['case_id'],
                         complete=complete, success=outcome.get('native_success') if complete else None,
                         score=outcome.get('native_score') if complete else None))
    return dict(source='local_experiments', complete=all(r['complete'] for r in rows),
                methods={m: metrics([r for r in rows if r['method'] == m])
                         for m in METHODS if any(r['method'] == m for r in rows)}, cases=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'doctor', 'login', 'run', 'summarize', 'published'])
    p.add_argument('--method', choices=['both', *METHODS], default='both')
    p.add_argument('--task')
    p.add_argument('--case-id')
    p.add_argument('--work-dir', type=Path, default=ROOT/'runtime/local_experiment')
    p.add_argument('--source', type=Path)
    p.add_argument('--sim-python', type=Path)
    p.add_argument('--openpi-source', type=Path)
    p.add_argument('--openpi-python', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--codex', default=shutil.which('codex') or 'codex')
    p.add_argument('--sim-gpu', default='0')
    p.add_argument('--policy-gpu', default='0')
    p.add_argument('--sim-port', type=int, default=19113)
    p.add_argument('--policy-port', type=int, default=18830)
    p.add_argument('--graphics-mode', choices=['system', 'bundled'], default='system')
    args = p.parse_args()
    for name in ('work_dir', 'source', 'sim_python', 'openpi_source', 'openpi_python', 'checkpoint'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().absolute())
    try:
        if args.command == 'published':
            result = published_summary()
        elif args.command == 'summarize':
            result = summarize(args.work_dir)
        elif args.command == 'login':
            from .codex_backend.profiles import login_session
            login_session(PROFILE, args.codex, args.work_dir)
            return
        else:
            plan = make_plan(args.method, args.task, args.case_id)
            if args.command == 'run':
                execute(args, plan)
                result = summarize(args.work_dir)
            elif args.command == 'doctor':
                result = doctor(args, plan)
            else:
                result = plan
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == 'doctor' and not result['ready']:
            raise SystemExit(2)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
