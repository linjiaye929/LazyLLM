import asyncio
import copy
import uuid

import pytest
import lazyllm
from lazyllm.module.module import ModuleExecutionError
from lazyllm.tools import ReactAgent
from lazyllm.tools.agent import ToolExecutionError


pytestmark = pytest.mark.usefixtures('model_call_session')


def final_summary_observation():
    '''Return a completed observation for final-summary tests.'''
    return 'completed observation'


class ScriptedModel:
    def __init__(self, outputs=None):
        self._module_id = uuid.uuid4().hex
        self.calls = []
        self.outputs = iter(outputs) if outputs is not None else None

    def share(self, **kwargs):
        return copy.copy(self)

    def used_by(self, module_id):
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if self.outputs is not None:
            return copy.deepcopy(next(self.outputs))
        return {'content': '', 'tool_calls': [{
            'id': 'final-summary-tool', 'type': 'function',
            'function': {'name': 'final_summary_observation', 'arguments': '{}'},
        }]}


@pytest.mark.parametrize('cancelled', [False, True])
def test_final_summary_callback_uses_the_public_constructor_and_preserves_llm(tmp_path, cancelled):
    llm = ScriptedModel()
    prompts = []
    agent = None

    def summarize(prompt):
        assert agent._llm is llm
        prompts.append(prompt)
        if cancelled:
            raise asyncio.CancelledError('summary cancelled')
        return 'final answer from the confirmed model'

    agent = ReactAgent(llm, [final_summary_observation], max_retries=1, force_summarize=True,
                       workspace=str(tmp_path), enable_builtin_tools=False, sandbox=None,
                       final_summary_call=summarize)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            agent('task')
    else:
        assert agent('task') == 'final answer from the confirmed model'
    assert len(llm.calls) == 2
    assert len(prompts) == 1 and 'completed observation' in prompts[0]
    assert agent._llm is llm


@pytest.mark.parametrize('use_callback', [False, True])
def test_final_summary_includes_last_results_without_advancing_history(tmp_path, use_callback):
    max_retries = 1
    executed = []
    prompts = []
    snapshots = []
    calls = {'compact': 0, 'context': 0, 'prepare': 0}

    def observe(index: int) -> str:
        '''Return a distinct observation for each tool round.'''
        executed.append(index)
        return f'completed-observation-{index}'

    outputs = [{'content': f'Inspect round {index}.', 'tool_calls': [{
        'id': f'call-{index}', 'type': 'function',
        'function': {'name': 'observe', 'arguments': {'index': index}},
    }]} for index in range(max_retries + 1)]
    llm = ScriptedModel(outputs + ['final answer'])

    def summarize(prompt):
        prompts.append(prompt)
        return 'final answer'

    def compact(history, _keep, current_round_messages=None, **kwargs):
        calls['compact'] += 1
        return list(history), list(current_round_messages or [])

    def provide_context():
        calls['context'] += 1
        return 'Runtime context.'

    def prepare():
        calls['prepare'] += 1

    def observe_runtime(event, **kwargs):
        if event == 'turn_end':
            workspace = lazyllm.locals['_lazyllm_agent']['workspace']
            snapshots.append((workspace, copy.deepcopy(workspace), dict(calls)))

    histories = lazyllm.locals['chat_history']
    histories['other-model'] = [{'role': 'user', 'content': 'unrelated'}]
    agent = ReactAgent(llm, [observe], max_retries=max_retries, force_summarize=True,
                       workspace=str(tmp_path), enable_builtin_tools=False, sandbox=None,
                       final_summary_call=summarize if use_callback else None,
                       history_compactor=compact, model_context_provider=provide_context,
                       before_model_request=prepare, runtime_observer=observe_runtime)

    assert agent('task') == 'final answer'
    assert executed == list(range(max_retries + 1))
    assert len(llm.calls) == max_retries + 1 + (not use_callback)
    assert len(prompts) == int(use_callback)
    prompt = prompts[0] if use_callback else llm.calls[-1][0]
    for index in executed:
        assert prompt.count(f'completed-observation-{index}') == 1
        assert prompt.index(f'Inspect round {index}.') < prompt.index(f'completed-observation-{index}')
    workspace, snapshot, call_snapshot = snapshots[-1]
    assert lazyllm.locals['_lazyllm_agent']['workspace'] is workspace
    assert workspace == snapshot
    assert calls == call_snapshot == {name: max_retries + 1 for name in calls}
    assert histories == {'other-model': [{'role': 'user', 'content': 'unrelated'}]}


def test_final_summary_includes_multiple_results_without_failure_metadata(tmp_path):
    executed = []
    prompts = []

    def observe() -> dict:
        '''Return a structured observation.'''
        executed.append('observe')
        return {'status': 'final-structured-result'}

    def require_approval() -> str:
        '''Return an approval failure without changing anything.'''
        executed.append('require_approval')
        raise ToolExecutionError.approval_required('final-approval-failure')

    first_round = {'content': '', 'tool_calls': [{
        'id': 'first-round', 'type': 'function',
        'function': {'name': 'final_summary_observation', 'arguments': '{}'},
    }]}
    llm = ScriptedModel([first_round, {'content': 'Inspect both results.', 'tool_calls': [{
        'id': f'call-{name}', 'type': 'function',
        'function': {'name': name, 'arguments': '{}'},
    } for name in ('observe', 'require_approval')]}])

    def summarize(prompt):
        prompts.append(prompt)
        return 'final answer'

    agent = ReactAgent(llm, [final_summary_observation, observe, require_approval], max_retries=1, force_summarize=True,
                       workspace=str(tmp_path), enable_builtin_tools=False, sandbox=None,
                       final_summary_call=summarize)
    assert agent('task') == 'final answer'
    assert sorted(executed) == ['observe', 'require_approval']
    assert len(llm.calls) == 2
    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt.count('final-structured-result') == 1
    assert prompt.count('final-approval-failure') == 1
    assert prompt.index('final-structured-result') < prompt.index('final-approval-failure')
    assert 'needs_approval' not in prompt
    assert 'tool_call_result' not in prompt


@pytest.mark.parametrize('outcome', ['', RuntimeError('summary failed')])
def test_unsuccessful_final_summary_preserves_cleanup(tmp_path, outcome):
    llm = ScriptedModel()
    prompts = []
    state = lazyllm.locals['_lazyllm_agent']
    state.update(completed=['previous result'], history=['previous history'])
    histories = lazyllm.locals['chat_history']
    histories['other-model'] = ['unrelated']

    def summarize(prompt):
        prompts.append(prompt)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    agent = ReactAgent(llm, [final_summary_observation], max_retries=1, force_summarize=True,
                       workspace=str(tmp_path), enable_builtin_tools=False, sandbox=None,
                       final_summary_call=summarize)
    with pytest.raises(ModuleExecutionError, match='After retrying 1 times') as error:
        agent('task')
    assert isinstance(error.value.__context__, ValueError)
    assert len(llm.calls) == 2
    assert len(prompts) == 1
    assert state == {'completed': ['previous result'], 'history': ['previous history']}
    assert histories == {'other-model': ['unrelated']}
