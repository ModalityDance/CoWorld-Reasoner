"""Local SDK replay journal owned by the active run.

Provider responses and tool returns are replayed verbatim after an interruption.
The SDK still builds the history, preserving call IDs and order. Pickle stores
locally produced SDK objects.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

from agents.models.interface import Model


def _journal_path(path):
    """Allow long checkpoint paths on Windows using the active runtime."""
    path = Path(path)
    if os.name != 'nt':
        return path
    value = os.path.abspath(path)
    if not value.startswith('\\\\?\\'):
        value = ('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\')
                 else '\\\\?\\' + value)
    return Path(value)


def atomic_bytes(path, content):
    path = _journal_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode())


def digest(value):
    from runtime.agent_runtime import _jsonable
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def save_object(path, value):
    atomic_bytes(path, gzip.compress(pickle.dumps(value), compresslevel=1))


def load_object(path):
    return pickle.loads(gzip.decompress(_journal_path(path).read_bytes()))


def context_state(context):
        # Persist execution state and dynamic tool fields; runtime clients,
        # credentials, closures and backend configuration stay process-local.
    excluded = {'output_dir', 'observed_image_paths', 'observed_image_roles',
                'started_at', 'backend_config', 'events', 'provider_errors'}
    result = {}
    for key, value in vars(context).items():
        if key not in excluded and isinstance(value, (str, int, float, bool, list, dict, type(None))):
            result[key] = copy.deepcopy(value)
    return result


class ReplayModel(Model):
    def __init__(self, inner, root, context):
        self.inner, self.root, self.context = inner, _journal_path(root), context
        self.index = 0

    async def get_response(self, *args, **kwargs):
        index = self.index
        self.index += 1
        path = self.root / f'model_{index:03d}.pkl.gz'
        # Tracing instances and callback objects remain process metadata.
        key = digest({'args': args, **{k: v for k, v in kwargs.items()
                     if k in {'system_instructions', 'input', 'model_settings',
                              'previous_response_id', 'conversation_id'}}})
        if path.exists():
            saved = load_object(path)
            if saved['key'] != key:
                raise ValueError(f'replay_model_input_changed:{index}')
            self.context.emit('checkpoint_replay', {'kind': 'model', 'index': index})
            return saved['response']
        atomic_json(self.root / f'model_{index:03d}.started.json', {'key': key})
        response = await self.inner.get_response(*args, **kwargs)
        save_object(path, {'key': key, 'response': response})
        return response

    async def stream_response(self, *args, **kwargs):
        raise RuntimeError('durable_agent_requires_nonstreaming_runner')
        yield

    async def close(self):
        await self.inner.close()


def journal_tools(tools, root):
    result = []
    for tool in tools:
        original = tool.on_invoke_tool
        wrapped = copy.copy(tool)

        async def invoke(wrapper, raw, original=original, name=tool.name):
            call_id = str(wrapper.tool_call_id)
            key = digest({'name': name, 'call_id': call_id, 'arguments': raw})
            path = _journal_path(Path(root) / ('tool_' + key + '.pkl.gz'))
            context = wrapper.context
            if path.exists():
                saved = load_object(path)
                for field, value in saved['state'].items():
                    setattr(context, field, value)
                context.events.extend(saved['events'])
                context.emit('checkpoint_replay', {'kind': 'tool', 'name': name, 'call_id': call_id})
                return saved['result']
            atomic_json(path.with_suffix('.started.json'), {'name': name, 'call_id': call_id, 'key': key})
            start = len(context.events)
            output = await original(wrapper, raw)
            # Transport failures must be recovered against the same cached model
            # tool call, hence the same backend request ID and FAL receipt.
            generations = getattr(context, 'generation_records', [])
            if generations and any(not r.get('generation', {}).get('technically_valid')
                                   for r in generations):
                failures = [str(reason) for record in generations
                            for reason in (record.get('generation', {}).get('tool_return', {})
                                           .get('failure_modes') or [])]
                detail = '; '.join(failures)[:2000]
                raise RuntimeError('generation_delivery_incomplete_resume_same_request'
                                   + (':' + detail if detail else ''))
            reviews = getattr(context, 'verifier_records', [])
            from runtime.failures import verifier_transport_failure
            failure = verifier_transport_failure(reviews)
            if failure:
                raise RuntimeError(failure + ':resume_same_call')
            save_object(path, {'result': output, 'state': context_state(context),
                               'events': context.events[start:]})
            return output

        wrapped.on_invoke_tool = invoke
        result.append(wrapped)
    return result
