"""Generate pinned responses in atomic shards and resume committed response IDs.

Run with --help for backend, subset and context settings.
"""
import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

from jsonschema import Draft202012Validator

from data_contracts import (CONTRACTS, ROOT, VERSION, canonical, digest, manifest_identity,
                            response_id, validate_bundle, verify_artifacts)
from resolve_models import sha256, template_text, tokenizer_for, SLOW_TOKENIZERS


def atomic_write(path, value, lines=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        if lines:
            for row in value:
                stream.write(canonical(row) + '\n')
        else:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def directory_lock(path):
    with (path / '.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Shards:
    def __init__(self, path, kind, key):
        self.path, self.kind, self.key = Path(path), kind, key
        self.path.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.path / f'{kind}.checksums.json'
        self.journal = read_json(self.journal_path) if self.journal_path.exists() else {}
        self.rows = {}
        validator = Draft202012Validator(CONTRACTS[kind])
        files = sorted(self.path.glob(f'{kind}-*.jsonl'))
        if set(self.journal) - {p.name for p in files}:
            raise ValueError('A committed shard is missing')
        for file in files:
            checksum = file_hash(file)
            if file.name in self.journal and checksum != self.journal[file.name]:
                raise ValueError('Committed shard checksum changed')
            for line in file.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                validator.validate(row)
                if row[key] in self.rows:
                    raise ValueError('Duplicate identity in committed shards')
                self.rows[row[key]] = row
            self.journal[file.name] = checksum
        if files:
            atomic_write(self.journal_path, self.journal)
        self.next_index = max([int(p.stem.rsplit('-', 1)[1]) for p in files], default=-1) + 1

    def append(self, rows):
        validator = Draft202012Validator(CONTRACTS[self.kind])
        ids = [row[self.key] for row in rows]
        if len(ids) != len(set(ids)) or any(key in self.rows for key in ids):
            raise ValueError('Refusing duplicate shard identities')
        for row in rows:
            validator.validate(row)
        path = self.path / f'{self.kind}-{self.next_index:06d}.jsonl'
        if path.exists():
            raise ValueError('Refusing to overwrite a committed shard')
        atomic_write(path, rows, lines=True)
        self.journal[path.name] = file_hash(path)
        atomic_write(self.journal_path, self.journal)
        self.rows.update({row[self.key]: row for row in rows})
        self.next_index += 1


def hardware_info():
    import torch
    devices = [{'name': torch.cuda.get_device_name(i),
                'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
    return canonical({'platform': platform.system(), 'devices': devices or ['CPU']})


class GPUMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.baseline = []
        self.peak = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                command = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
                result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                values = [int(line.strip()) for line in result.stdout.splitlines()]
                if not self.baseline:
                    self.baseline = values
                self.peak = [max(a, b) for a, b in zip(self.peak or values, values)]
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)


def safe_error(exc):
    # Exceptions from Hub downloads can contain signed URLs.
    import re
    text = re.sub(r'https?://\S+', '[URL]', str(exc))
    text = re.sub(r'hf_[A-Za-z0-9]+', '[TOKEN]', text)
    return type(exc).__name__ + ': ' + text[:800]


class TransformersBackend:
    def __init__(self, model, max_model_len=4096, device='cuda', cpu_threads=2):
        import torch
        if cpu_threads < 1:
            raise ValueError('CPU threads must be positive')
        torch.set_num_threads(cpu_threads)
        self.model_entry = model
        self.tokenizer = tokenizer_for(model)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; use --device cpu explicitly')
        if device == 'cpu' and model['precision'] != 'float32':
            raise ValueError('CPU runs require a separate float32 model manifest')
        self.engine = 'transformers'
        self.engine_version = version('transformers')
        self.hardware = hardware_info()
        self.configuration = {'max_model_len': max_model_len, 'device': device, 'cpu_threads': cpu_threads,
                              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                              'precision': model['precision'], 'tokenizer_class': type(self.tokenizer).__name__}
        self.configuration['tokenizer_packages'] = {
            name: version(name) for name in ['transformers', 'tokenizers', 'huggingface_hub', 'sentencepiece']}

    def generate(self, requests, settings):
        import torch
        from transformers import AutoModelForCausalLM
        if self.model is None:
            dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_entry['model_id'], revision=self.model_entry['revision'],
                torch_dtype=dtype[self.model_entry['precision']], trust_remote_code=False)
            self.model.to(self.device).eval()
            torch.manual_seed(settings['seed'])
        inputs = self.tokenizer.pad({'input_ids': [r['input_token_ids'] for r in requests]},
                                    padding=True, return_tensors='pt').to(self.device)
        width = inputs['input_ids'].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False,
                                            max_new_tokens=settings['max_new_tokens'],
                                            pad_token_id=self.tokenizer.pad_token_id)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        outputs = []
        for row in generated[:, width:].tolist():
            stop = next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
            ids = row[:stop]
            stopped = bool(ids and ids[-1] in eos)
            outputs.append({'text': self.tokenizer.decode(ids, skip_special_tokens=True),
                            'output_tokens': len(ids), 'finish_reason': 'stop' if stopped else 'length',
                            'output_truncated': not stopped and len(ids) >= settings['max_new_tokens']})
        return outputs


class VLLMBackend:
    def __init__(self, model, max_model_len=4096, tensor_parallel=2, seed=42):
        import torch
        self.model_entry = model
        self.tokenizer = tokenizer_for(model)
        self.engine = 'vllm'
        self.engine_version = version('vllm')
        self.hardware = hardware_info()
        self.configuration = {'max_model_len': max_model_len, 'tensor_parallel_size': tensor_parallel,
                              'precision': model['precision'], 'gpu_memory_utilization': .85,
                              'enforce_eager': True, 'seed': seed, 'generation_config': 'auto',
                              'attention_backend': os.environ.get('VLLM_ATTENTION_BACKEND', 'auto'),
                              'vllm_v1': os.environ.get('VLLM_USE_V1', 'auto'),
                              'transformers_version': version('transformers'),
                              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                              'tokenizer_packages': {name: version(name) for name in
                                  ['transformers', 'tokenizers', 'huggingface_hub', 'sentencepiece']}}
        self.llm = None

    def generate(self, requests, settings):
        from vllm import LLM, SamplingParams
        if self.llm is None:
            self.llm = LLM(model=self.model_entry['model_id'], revision=self.model_entry['revision'],
                           tokenizer_revision=self.model_entry['tokenizer_revision'],
                           tokenizer_mode='slow' if self.model_entry['model_id'] in SLOW_TOKENIZERS else 'auto',
                           dtype=self.model_entry['precision'], trust_remote_code=False,
                           max_model_len=self.configuration['max_model_len'],
                           tensor_parallel_size=self.configuration['tensor_parallel_size'],
                           gpu_memory_utilization=.85, enforce_eager=True, seed=settings['seed'],
                           generation_config='auto')
        params = SamplingParams(temperature=0, top_p=1, max_tokens=settings['max_new_tokens'],
                                seed=settings['seed'])
        outputs = self.llm.generate([{'prompt_token_ids': r['input_token_ids']} for r in requests],
                                    params, use_tqdm=False)
        result = []
        for output in outputs:
            choice = output.outputs[0]
            result.append({'text': choice.text, 'output_tokens': len(choice.token_ids),
                           'finish_reason': choice.finish_reason or 'unknown',
                           'output_truncated': choice.finish_reason == 'length'})
        return result


def requests_for(prompts, tokenizer, limit):
    requests = []
    for prompt in prompts:
        try:
            rendered = tokenizer.apply_chat_template(prompt['messages'], tokenize=False,
                                                      add_generation_prompt=True)
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            requests.append({'prompt_id': prompt['prompt_id'], 'rendered_prompt': rendered,
                             'raw_input_tokens': len(tokens), 'input_token_ids': tokens[-limit:],
                             'input_truncated': len(tokens) > limit, 'error': None})
        except Exception as exc:
            requests.append({'prompt_id': prompt['prompt_id'], 'rendered_prompt': None,
                             'raw_input_tokens': None, 'input_token_ids': [],
                             'input_truncated': False, 'error': safe_error(exc)})
    return requests


def generate_isolated(backend, requests, settings):
    if not requests:
        return []
    try:
        result = backend.generate(requests, settings)
        if len(result) != len(requests):
            raise ValueError('Backend returned the wrong response count')
        return result
    except Exception:
        result = []
        for request in requests:
            try:
                result.append(backend.generate([request], settings)[0])
            except Exception as exc:
                result.append({'error': safe_error(exc)})
        return result


def run_generation(prompt_bundle, model_bundle, model_id, backend, settings, output_root,
                   splits=('development',), limit=None, batch_size=4, after_commit=None):
    if settings['do_sample'] or settings['repetitions'] != 1:
        raise ValueError('This protocol supports greedy decoding and one repetition')
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError('Batch size and limit must be positive')
    context = {**prompt_bundle, **model_bundle}
    validate_bundle(context)
    panel = model_bundle['model_manifests'][0]
    if panel['status'] != 'frozen':
        raise ValueError('Inference requires a frozen model manifest')
    model = next(m for m in panel['models'] if m['model_id'] == model_id)
    if model['role'] != 'primary_assistant':
        raise ValueError('Base diagnostics need their own continuation protocol')
    if sha256(template_text(backend.tokenizer)) != model['chat_template_sha256']:
        raise ValueError('Loaded chat template differs from the pinned manifest')
    prompt_manifest = prompt_bundle['prompt_manifests'][0]
    selected = sorted([p for p in prompt_bundle['prompts'] if
                       prompt_manifest['group_splits'][p['prompt_group_id']] in splits],
                      key=lambda p: p['prompt_id'])
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError('Prompt subset is empty')
    token_limit = min(settings['max_input_tokens'],
                      backend.configuration['max_model_len'] - settings['max_new_tokens'])
    if token_limit < 1:
        raise ValueError('Context must leave space for input and generated tokens')
    code_files = [Path(__file__), Path(__file__).with_name('data_contracts.py'),
                  Path(__file__).with_name('resolve_models.py')]
    code_checksums = {path.name: file_hash(path) for path in code_files}
    configuration = {**backend.configuration, 'batch_size': batch_size, 'model_id': model_id,
                     'effective_input_token_limit': token_limit,
                     'selection': {'splits': sorted(splits), 'limit': limit,
                                   'prompt_ids': [p['prompt_id'] for p in selected]},
                     'code_sha256': code_checksums}
    run = {'schema_version': VERSION, 'status': 'frozen', 'prompt_manifest_id': prompt_manifest['manifest_id'],
           'model_manifest_id': panel['manifest_id'], 'generation': settings,
           'runtime': {'engine': backend.engine, 'engine_version': backend.engine_version,
                       'hardware': backend.hardware, 'configuration': configuration},
           'output_path': 'data/processed/responses'}
    run['run_id'] = manifest_identity('run_manifests', run)
    context = {**context, 'run_manifests': [run]}
    validate_bundle(context)
    folder = Path(output_root) / run['run_id'][4:20] / model_id.replace('/', '__')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with directory_lock(folder):
        bundle_path = folder / 'run.bundle.json'
        if bundle_path.exists() and read_json(bundle_path) != context:
            raise ValueError('Run context differs from the saved bundle')
        if not bundle_path.exists():
            atomic_write(bundle_path, context)
        (folder / 'code').mkdir(exist_ok=True)
        for path in code_files:
            target = folder / 'code' / path.name
            if not target.exists():
                target.write_bytes(path.read_bytes())
            if file_hash(target) != code_checksums[path.name]:
                raise ValueError('Saved inference source differs from the run checksum')
        store = Shards(folder, 'responses', 'response_id')
        validate_bundle({**context, 'responses': list(store.rows.values())})
        selected_ids = {p['prompt_id'] for p in selected}
        if any(row['prompt_id'] not in selected_ids or row['model_id'] != model_id
               for row in store.rows.values()):
            raise ValueError('Committed responses fall outside this run subset')
        pending = [p for p in selected if response_id(run['run_id'], p['prompt_id'], model_id,
                   model['revision'], 0) not in store.rows]
        existing = len(store.rows)
        previous_stats = read_json(folder / 'stats.json') if (folder / 'stats.json').exists() else {}
        attempts = previous_stats.get('attempts', [])
        generated_tokens = 0
        generation_seconds = 0.0
        completed = False
        monitor = GPUMonitor()
        monitor.start()
        try:
            for offset in range(0, len(pending), batch_size):
                prompts = pending[offset:offset + batch_size]
                requests = requests_for(prompts, backend.tokenizer, token_limit)
                valid = [request for request in requests if request['error'] is None]
                before = time.perf_counter()
                outputs = generate_isolated(backend, valid, settings)
                generation_seconds += time.perf_counter() - before
                outputs = dict(zip([r['prompt_id'] for r in valid], outputs))
                rows = []
                for request in requests:
                    output = outputs.get(request['prompt_id'], {'error': request['error']})
                    error = output.get('error')
                    row = {'schema_version': VERSION, 'response_id': response_id(run['run_id'],
                           request['prompt_id'], model_id, model['revision'], 0), 'run_id': run['run_id'],
                           'prompt_id': request['prompt_id'], 'model_id': model_id,
                           'model_revision': model['revision'], 'repetition': 0,
                           'status': 'failed' if error else 'success', 'error': error,
                           'text': None if error else output['text'],
                           'input_tokens': len(request['input_token_ids']),
                           'output_tokens': None if error else output['output_tokens'],
                           'input_truncated': request['input_truncated'],
                           'output_truncated': False if error else output['output_truncated'],
                           'finish_reason': None if error else output['finish_reason'],
                           **{key: request[key] for key in ['rendered_prompt', 'raw_input_tokens',
                                                          'input_token_ids']}}
                    rows.append(row)
                store.append(rows)
                generated_tokens += sum(row['output_tokens'] or 0 for row in rows)
                if after_commit:
                    after_commit(len(store.rows))
                print(f'{model_id}: {len(store.rows)}/{len(selected)} committed', flush=True)
            completed = True
        finally:
            monitor.finish()
            elapsed = time.perf_counter() - started
            attempts.append({'wall_seconds': elapsed, 'generation_seconds': generation_seconds,
                             'new_responses': len(store.rows) - existing, 'new_output_tokens': generated_tokens,
                             'resumed_responses': existing, 'completed': completed})
            memory = []
            try:
                import torch
                memory = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
            except ImportError:
                pass
            rows = list(store.rows.values())
            seconds = sum(item['wall_seconds'] for item in attempts)
            old_peaks = previous_stats.get('peak_gpu_allocated_bytes', [])
            if len(memory) == len(old_peaks):
                memory = [max(a, b) for a, b in zip(memory, old_peaks)]
            device_peaks = monitor.peak
            old_device_peaks = previous_stats.get('peak_gpu_device_used_mib', [])
            if len(device_peaks) == len(old_device_peaks):
                device_peaks = [max(a, b) for a, b in zip(device_peaks, old_device_peaks)]
            stats = {'run_id': run['run_id'], 'model_id': model_id, 'completed': completed,
                     'responses': len(rows), 'failures': sum(r['status'] == 'failed' for r in rows),
                     'input_truncated': sum(r['input_truncated'] for r in rows),
                     'output_truncated': sum(r['output_truncated'] for r in rows),
                     'input_tokens': sum(r['input_tokens'] or 0 for r in rows),
                     'output_tokens': sum(r['output_tokens'] or 0 for r in rows),
                     'wall_seconds': seconds,
                     'output_tokens_per_wall_second': sum(r['output_tokens'] or 0 for r in rows) / seconds,
                     'peak_gpu_allocated_bytes': memory, 'hardware': backend.hardware,
                     'peak_memory_scope': 'calling process; vLLM worker allocations are not included',
                     'peak_gpu_device_used_mib': device_peaks,
                     'baseline_gpu_device_used_mib': previous_stats.get(
                         'baseline_gpu_device_used_mib', monitor.baseline),
                     'device_memory_scope': '1-second nvidia-smi samples, all processes including workers',
                     'selection': configuration['selection'], 'attempts': attempts}
            atomic_write(folder / 'stats.json', stats)
        validation = {**context, 'responses': list(store.rows.values())}
        validate_bundle(validation)
        atomic_write(folder / 'validation.bundle.json', validation)
    return folder, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['transformers', 'vllm'], required=True)
    parser.add_argument('--split', nargs='+', choices=['development', 'validation', 'test'],
                        default=['development'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--tensor-parallel', type=int, default=2)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/processed/responses')
    args = parser.parse_args()
    prompts, models = read_json(args.prompts), read_json(args.models)
    validate_bundle({**prompts, **models})
    verify_artifacts(prompts)
    model = next(m for m in models['model_manifests'][0]['models'] if m['model_id'] == args.model)
    backend = TransformersBackend(model, args.max_model_len, args.device, args.cpu_threads) \
        if args.backend == 'transformers' \
        else VLLMBackend(model, args.max_model_len, args.tensor_parallel, args.seed)
    settings = {'seed': args.seed, 'do_sample': False, 'temperature': 0, 'top_p': 1,
                'max_input_tokens': args.max_input_tokens, 'max_new_tokens': args.max_new_tokens,
                'repetitions': 1}
    folder, stats = run_generation(prompts, models, args.model, backend, settings, args.output_root,
                                   args.split, args.limit, args.batch_size)
    print('Run folder:', folder)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
