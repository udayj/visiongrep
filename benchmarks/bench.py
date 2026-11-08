#!/usr/bin/env python3
"""Fresh, paired comparisons of two VisionGrep commits."""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

if __package__:
    from . import aws, quality
else:
    import aws
    import quality

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = Path.home() / '.cache/visiongrep-bench'
SCENARIOS = ('index', 'novel_text', 'cached_text', 'external_image', 'changed_1pct', 'server_novel_text')
TEXT = 'a bicycle near water'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs():
    images = read(HERE / 'corpora/coco-500.json')['images']
    checksums = read(HERE / 'corpora/image-checksums.json')
    return [row | checksums[row['file_name']] for row in images]


def fetch(url, path, checksum):
    if path.is_file() and digest(path) == checksum:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, tmp)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        if digest(temp) != checksum:
            raise ValueError(f'checksum mismatch: {url}')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def prepare():
    for row in read(HERE / 'artifacts.json') + inputs():
        fetch(row['url'], CACHE / 'objects' / row['sha256'], row['sha256'])


def verify(rows):
    for row in read(HERE / 'artifacts.json') + rows:
        path = CACHE / 'objects' / row['sha256']
        if not path.is_file() or digest(path) != row['sha256']:
            raise ValueError(f'missing or corrupt {path}; run prepare first')


def resolve(ref):
    return subprocess.check_output(['git', 'rev-parse', '--verify', f'{ref}^{{commit}}'], cwd=ROOT, text=True).strip()


def build(sha):
    binary = CACHE / 'binaries' / sha / 'visiongrep'
    if binary.is_file():
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='visiongrep-build-') as tmp:
        tree = Path(tmp) / 'source'
        subprocess.run(['git', 'worktree', 'add', '--detach', str(tree), sha], cwd=ROOT, check=True)
        try:
            subprocess.run(['cargo', 'build', '--release', '--locked'], cwd=tree, check=True)
            with tempfile.NamedTemporaryFile(dir=binary.parent, delete=False) as output:
                temporary = Path(output.name)
            try:
                shutil.copy2(tree / 'target/release/visiongrep', temporary)
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            subprocess.run(['git', 'worktree', 'remove', '--force', str(tree)], cwd=ROOT, check=True)
    return binary


def change_interval(reference, candidate):
    def change(a, b):
        return 1 - statistics.median(b) / statistics.median(a)
    rng = random.Random(91017)
    draws = []
    for _ in range(2000):
        indices = rng.choices(range(len(reference)), k=len(reference))
        draws.append(change([reference[i] for i in indices], [candidate[i] for i in indices]))
    draws.sort()
    return change(reference, candidate), draws[49], draws[1949]


def output_difference(left, right):
    if len(left) != len(right):
        return f'result count differs: {len(left)} vs {len(right)}'
    for i, (a, b) in enumerate(zip(left, right)):
        if (a['path'] != b['path'] or not math.isfinite(a['score'])
                or not math.isfinite(b['score']) or abs(a['score'] - b['score']) > 1e-4):
            return f'result {i + 1} differs: {a} vs {b}'
    return None


def sample_order(index):
    return (0, 1) if index % 2 == 0 else (1, 0)


def rss_bytes(value, platform_name):
    return value if platform_name.lower() == 'darwin' else value * 1024


def measured(command, env):
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.perf_counter()
        child = subprocess.Popen(command, stdout=out, stderr=err, env=env)
        _, status, usage = os.wait4(child.pid, 0)
        elapsed = time.perf_counter() - start
        child.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        if child.returncode not in (0, 1):
            raise RuntimeError(f'command failed: {command}\n{err.read().decode(errors="replace")[-3000:]}')
        return {'seconds': elapsed, 'rss_bytes': rss_bytes(usage.ru_maxrss, sys.platform), 'results': json.load(out)}


def stage(root, binary, rows):
    images = root / 'images'
    images.mkdir(parents=True)
    for row in rows:
        shutil.copyfile(CACHE / 'objects' / row['sha256'], images / row['file_name'])
    models = root / 'cache/visiongrep/models'
    models.mkdir(parents=True)
    for row in read(HERE / 'artifacts.json'):
        (models / row['name']).symlink_to(CACHE / 'objects' / row['sha256'])
    external = root / 'external.jpg'
    shutil.copyfile(images / rows[0]['file_name'], external)
    env = os.environ.copy()
    env['XDG_CACHE_HOME'] = str(root / 'cache')
    return {'root': root, 'images': images, 'index': root / 'index.db', 'external': external, 'env': env, 'binary': binary, 'rows': rows}


def command(state, query=TEXT, image=None, top=10):
    query_args = ['--image', str(image)] if image else [query]
    return [str(state['binary']), *query_args, str(state['images']),
            '--top', str(top), '--threshold', '-1', '--json', '--quiet',
            '--index-path', str(state['index'])]


def run(state, query=TEXT, image=None, top=10):
    result = measured(command(state, query, image, top), state['env'])
    for item in result['results']:
        item['path'] = str(Path(item['path']).relative_to(state['images']))
    return result


def checkpoint(state):
    with sqlite3.connect(state['index']) as db:
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')


def reset(state, scenario):
    for suffix in ('', '-wal', '-shm', '-journal'):
        Path(str(state['index']) + suffix).unlink(missing_ok=True)
    if scenario != 'index':
        shutil.copyfile(state['root'] / 'seed.db', state['index'])
    if scenario == 'changed_1pct':
        for row in state['rows'][:max(1, math.ceil(len(state['rows']) * .01))]:
            path = state['images'] / row['file_name']
            shutil.copyfile(CACHE / 'objects' / state['rows'][-1]['sha256'], path)
            stamp = state['stamps'][row['file_name']] + 1_000_000_000
            os.utime(path, ns=(stamp, stamp))


def restore_images(state):
    for row in state['rows'][:max(1, math.ceil(len(state['rows']) * .01))]:
        path = state['images'] / row['file_name']
        shutil.copyfile(CACHE / 'objects' / row['sha256'], path)
        stamp = state['stamps'][row['file_name']]
        os.utime(path, ns=(stamp, stamp))


def start_server(state):
    cmd = [str(state['binary']), 'serve', '--socket', '--index-path', str(state['index']), str(state['images'])]
    log = (state['root'] / 'server.log').open('wb')
    child = subprocess.Popen(cmd, env=state['env'], stdout=log, stderr=log)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if child.poll() is not None:
            log.close()
            raise RuntimeError(f'server exited: {(state["root"] / "server.log").read_text()[-3000:]}')
        for path in (state['root'] / 'cache/visiongrep/run').glob('*.sock'):
            try:
                with socket.socket(socket.AF_UNIX) as s:
                    s.connect(str(path))
                return child, log
            except OSError:
                pass
        time.sleep(.1)
    child.kill()
    child.wait()
    log.close()
    raise RuntimeError('server did not become ready')


def ram_bytes():
    if sys.platform == 'darwin':
        return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            return int(line.split()[1]) * 1024
    raise ValueError('cannot determine total RAM')


def print_report(report):
    print(
        f'{"scenario":20} {"ref ms":>10} {"cand ms":>10} '
        f'{"change (95% interval)":>28} {"ref MiB":>9} {"cand MiB":>9}'
    )
    for name, samples in report['scenarios'].items():
        a, b = samples['reference'], samples['candidate']
        change, low, high = change_interval(
            [x['seconds'] for x in a], [x['seconds'] for x in b]
        )
        ref_ms = statistics.median(x['seconds'] for x in a) * 1000
        cand_ms = statistics.median(x['seconds'] for x in b) * 1000
        ref_mib = statistics.median(x['rss_bytes'] for x in a) / 1048576
        cand_mib = statistics.median(x['rss_bytes'] for x in b) / 1048576
        print(
            f'{name:20} {ref_ms:10.1f} {cand_ms:10.1f} '
            f'{change*100:7.1f}% [{low*100:6.1f}%, {high*100:6.1f}%] '
            f'{ref_mib:9.1f} {cand_mib:9.1f}'
        )


def compare(args):
    if args.samples < 1:
        raise ValueError('samples must be positive')
    scenarios = list(SCENARIOS) if args.scenarios == 'all' else args.scenarios.split(',')
    if not scenarios or any(s not in SCENARIOS for s in scenarios) or len(set(scenarios)) != len(scenarios):
        raise ValueError('invalid or duplicate scenario')
    shas = [resolve(args.ref), resolve(args.cand)]
    rows = inputs()
    verify(rows)
    ram = ram_bytes()
    if ram < 8 * 1024**3:
        print('warning: less than 8 GiB RAM may affect benchmark results', file=sys.stderr)
    binaries = [build(sha) for sha in shas]
    report = {
        'reference': shas[0],
        'candidate': shas[1],
        'machine': {
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
            'ram_bytes': ram,
        },
        'flags': {
            'samples': args.samples,
            'corpus': '500',
            'scenarios': scenarios,
            'allow_output_changes': args.allow_output_changes,
        },
        'scenarios': {},
    }
    differences = []
    with tempfile.TemporaryDirectory(prefix='vg-', dir='/tmp') as tmp:
        states = [stage(Path(tmp) / str(i), binaries[i], rows) for i in range(2)]
        for state in states:
            run(state)
            checkpoint(state)
            shutil.copyfile(state['index'], state['root'] / 'seed.db')
            state['stamps'] = {r['file_name']: (state['images'] / r['file_name']).stat().st_mtime_ns for r in rows}
        for scenario in scenarios:
            if scenario == 'server_novel_text':
                help_results = [
                    subprocess.run([str(binary), 'serve', '--help'], capture_output=True)
                    for binary in dict.fromkeys(binaries)
                ]
                supported = all(
                    result.returncode == 0 and b'--socket' in result.stdout
                    for result in help_results
                )
                if not supported:
                    print('server_novel_text skipped: serve --socket is unavailable', file=sys.stderr)
                    continue
            print(f'running {scenario}...', file=sys.stderr, flush=True)
            servers = []
            try:
                for state in states:
                    reset(state, scenario)
                    if scenario == 'server_novel_text':
                        servers.append(start_server(state))
                    run(state, query='warmup ' + TEXT, image=state['external'] if scenario == 'external_image' else None)
                samples = [[], []]
                for n in range(args.samples):
                    for i in sample_order(n):
                        state = states[i]
                        if scenario != 'server_novel_text':
                            reset(state, scenario)
                        query = f'{TEXT} novel {n}' if scenario in ('novel_text', 'server_novel_text') else TEXT
                        if servers and servers[i][0].poll() is not None:
                            raise RuntimeError('benchmark server exited before the sample')
                        result = run(state, query=query, image=state['external'] if scenario == 'external_image' else None)
                        if servers and servers[i][0].poll() is not None:
                            raise RuntimeError('benchmark server exited during the sample')
                        samples[i].append(result)
                        if scenario == 'changed_1pct':
                            restore_images(state)
                    difference = output_difference(samples[0][-1]['results'], samples[1][-1]['results'])
                    if difference:
                        differences.append(f'{scenario} sample {n + 1}: {difference}')
            finally:
                for child, log in servers:
                    child.terminate()
                    child.wait()
                    log.close()
            report['scenarios'][scenario] = {'reference': samples[0], 'candidate': samples[1]}
    CACHE.joinpath('runs').mkdir(parents=True, exist_ok=True)
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'-{shas[0][:12]}-{shas[1][:12]}.json'
    path = CACHE / 'runs' / name
    path.write_text(json.dumps(report, indent=2) + '\n')
    print_report(report)
    print(f'raw samples: {path}')
    if differences:
        print(f'output difference: {differences[0]}', file=sys.stderr)
        if not args.allow_output_changes:
            raise ValueError('outputs differ; use --allow-output-changes to accept')


def quality_command(args):
    shas = [resolve(ref) for ref in (args.ref, args.cand) if ref]
    rows = inputs()
    verify(rows)
    specification = read(HERE / 'corpora/quality-500.json')
    all_runs = []
    with tempfile.TemporaryDirectory(prefix='vg-quality-', dir='/tmp') as tmp:
        for i, sha in enumerate(shas):
            state = stage(Path(tmp) / str(i), build(sha), rows)
            run(state)
            runs = []
            for n, query in enumerate(specification['queries']):
                print(f'quality {sha[:12]}: {n + 1}/{len(specification["queries"])}', file=sys.stderr, end='\r', flush=True)
                result = run(state, query=query['query'], top=len(rows))
                if len(result['results']) != len(rows):
                    raise ValueError('quality requires a score for every image')
                runs.append(query | {'results': result['results']})
            all_runs.append(runs)
    print(file=sys.stderr)
    metrics = [quality.evaluate(runs, specification['threshold']) for runs in all_runs]
    print(f'{"metric":30} {shas[0][:12]:>12}' + (f' {shas[1][:12]:>12} {"difference":>12}' if len(shas) == 2 else ''))
    for name, value in metrics[0].items():
        print(f'{name:30} {value:12.4f}' + (f' {metrics[1][name]:12.4f} {metrics[1][name]-value:12.4f}' if len(metrics) == 2 else ''))
    if len(shas) == 2:
        print(f'top-10 identical: {quality.top10_agreement(*all_runs):.4f}')


def add_compare_flags(parser):
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--scenarios', default=','.join(SCENARIOS[1:]))
    parser.add_argument('--allow-output-changes', action='store_true')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('compare')
    p.add_argument('ref')
    p.add_argument('cand')
    add_compare_flags(p)
    p = sub.add_parser('quality')
    p.add_argument('ref')
    p.add_argument('cand', nargs='?')
    p = sub.add_parser('aws-run')
    p.add_argument('ref')
    p.add_argument('cand')
    p.add_argument('--settings', type=Path, required=True)
    add_compare_flags(p)
    p.add_argument('--instance-type', default='c7a.xlarge')
    p.add_argument('--max-minutes', type=int, default=60)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('aws-script')
    p.add_argument('ref')
    p.add_argument('script')
    p.add_argument('--settings', type=Path, required=True)
    p.add_argument('--instance-type', default='c7a.xlarge')
    p.add_argument('--max-minutes', type=int, default=60)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('aws-fetch')
    p.add_argument('run_id')
    p.add_argument('--settings', type=Path, required=True)
    p = sub.add_parser('aws-list')
    p.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare()
        elif args.action == 'compare':
            compare(args)
        elif args.action == 'quality':
            quality_command(args)
        elif args.action in ('aws-run', 'aws-script'):
            if args.action == 'aws-run':
                aws.validate_compare_options(args.samples, args.scenarios)
            settings = aws.settings_from(args.settings)
            if args.action == 'aws-run':
                ref, checkout = aws.resolve_remote(args.ref), aws.resolve_remote(args.cand)
                flags = ['--samples', str(args.samples), '--scenarios', args.scenarios]
                if args.allow_output_changes:
                    flags.append('--allow-output-changes')
                command_line = ['python3', 'benchmarks/bench.py', 'compare', ref, checkout, *flags]
            else:
                checkout = aws.resolve_remote(args.ref)
                command_line = ['env', f"BUCKET={settings['bucket']}", 'bash', '--', args.script]
            aws.run(args, settings, checkout, command_line)
        elif args.action == 'aws-fetch':
            aws.fetch(args, print_report)
        else:
            aws.list_instances(args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
