"""Fail-closed admission identity tests; temporary fixtures, no native stages."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from orchestration import admission


class AdmissionBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rssr_admission_binding_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controller = self.root / 'controllers/v1'
        self.controller.mkdir(parents=True)
        for name in admission.CONTROLLER_FILES:
            text = "VERSION = 'fixture.controller.v1'\n" if name == 'short_refine_agent.py' else '# controller fixture\n'
            (self.controller / name).write_text(text)
        self.implementation = self.root / 'implementation'
        self.implementation.mkdir()
        self.source_native = self.root / 'source_native'
        self.source_native.mkdir()
        self.source_evidence = self.source_native / 'evidence.json'
        self.source_evidence.write_text('{"source": true}\n')
        self.source = dict(source_id='physical_00__stack_cups_ep000001', case_id='stack_cups_ep000001',
                           physical_index=0, source_order=1, source_native=str(self.source_native),
                           source_pins={str(self.source_evidence): admission.sha(self.source_evidence)})
        self.strategy = self.root / 'strategy.json'
        self.write_strategy()
        self.native = self.root / 'attempt/native'
        self.write_native(self.native)
        # Calling a physical/audit child from any of these fixtures is a failure.
        self.stage = patch.object(admission.common, 'run_stage', side_effect=AssertionError('No native stage may run in identity tests'))
        self.stage_mock = self.stage.start()
        self.addCleanup(self.stage.stop)

    def write_strategy(self, pins=None):
        if pins is None:
            pins = {str((self.controller / name).resolve()): admission.sha(self.controller / name)
                    for name in admission.CONTROLLER_FILES}
        self.strategy.write_text(json.dumps({'implementation_sha256': pins}) + '\n')

    def write_native(self, path, agent_version='fixture.controller.v1'):
        path.mkdir(parents=True)
        (path / 'receipt.json').write_text(json.dumps({
            'accepted': False, 'failure_reason': 'fixture native task failure',
            'case_id': self.source['case_id'], 'agent_version': agent_version,
            'strategy_sha256': admission.sha(self.strategy)}) + '\n')

    def call(self, **updates):
        args = dict(native=self.native, strategy_path=self.strategy, controller=self.controller,
                    source_row=self.source, implementation=self.implementation,
                    python_executable=sys.executable)
        args.update(updates)
        return admission.admit(**args)

    @property
    def destination(self):
        return self.native.parent / 'admission_receipt.json'

    def test_rejected_native_receipt_stays_rejected_and_matching_resume_is_allowed(self):
        first = self.call()
        before = self.destination.read_bytes()
        self.assertFalse(first['accepted'])
        self.assertEqual(first['request_binding']['source_id'], self.source['source_id'])
        self.assertEqual(self.call(), first)
        self.assertEqual(self.destination.read_bytes(), before)
        self.stage_mock.assert_not_called()

    def test_owned_receipt_cannot_be_reused_for_another_native_sibling(self):
        self.call()
        other = self.native.with_name('another_native')
        self.write_native(other)
        before = self.destination.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'native_case differs'):
            self.call(native=other)
        self.assertEqual(self.destination.read_bytes(), before)

    def test_owned_receipt_cannot_be_reused_for_another_strategy_path(self):
        self.call()
        other = self.strategy.with_name('another_strategy.json')
        other.write_bytes(self.strategy.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'strategy differs'):
            self.call(strategy_path=other)

    def test_owned_receipt_cannot_be_reused_for_another_source_id(self):
        self.call()
        source = {**self.source, 'source_id': 'physical_01__stack_cups_ep000001'}
        with self.assertRaisesRegex(RuntimeError, 'source_id differs'):
            self.call(source_row=source)

    def test_owned_receipt_binds_source_evidence_and_order(self):
        self.call()
        source = {**self.source, 'source_order': 99}
        with self.assertRaisesRegex(RuntimeError, 'request binding differs'):
            self.call(source_row=source)
        extra = self.source_native / 'another.json'
        extra.write_text('{}\n')
        source = {**self.source, 'source_pins': {**self.source['source_pins'], str(extra): admission.sha(extra)}}
        with self.assertRaisesRegex(RuntimeError, 'request binding differs'):
            self.call(source_row=source)

    def test_owned_receipt_binds_explicit_runtime_and_safety_mode(self):
        self.call()
        other = self.root / 'another_implementation'
        other.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'request binding differs'):
            self.call(implementation=other)
        with self.assertRaisesRegex(RuntimeError, 'request binding differs'):
            self.call(safety_owned_externally=True)

    def test_unowned_receipt_is_not_adopted(self):
        first = self.call()
        first['input_pins'].pop(str(Path(admission.__file__).resolve()))
        self.destination.write_text(json.dumps(first) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'not owned'):
            self.call()

    def test_owned_receipt_must_cover_all_current_frozen_inputs(self):
        first = self.call()
        first['input_pins'].pop(str(self.source_evidence))
        self.destination.write_text(json.dumps(first) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'every current frozen input'):
            self.call()

    def test_all_four_controller_files_must_be_pinned(self):
        pins = json.loads(self.strategy.read_text())['implementation_sha256']
        pins.pop(str(self.controller / 'replay_load_audit.py'))
        self.write_strategy(pins)
        (self.native / 'receipt.json').unlink()
        self.native.rmdir()
        self.write_native(self.native)
        with self.assertRaisesRegex(ValueError, 'Controller file is not pinned'):
            self.call()
        self.assertFalse(self.destination.exists())

    def test_controller_digest_mismatch_rejects_before_any_stage(self):
        pins = json.loads(self.strategy.read_text())['implementation_sha256']
        pins[str(self.controller / 'short_native_audit.py')] = '0' * 64
        self.write_strategy(pins)
        with self.assertRaisesRegex(ValueError, 'Pinned controller file changed'):
            self.call()
        self.assertFalse(self.destination.exists())

    def test_controller_edit_after_pinning_rejects_even_when_receipt_exists(self):
        self.call()
        (self.controller / 'run_case.py').write_text('# replaced controller\n')
        with self.assertRaisesRegex(ValueError, 'Pinned controller file changed'):
            self.call()

    def test_missing_controller_file_rejects(self):
        (self.controller / 'short_native_audit.py').unlink()
        with self.assertRaises(FileNotFoundError):
            self.call()
        self.assertFalse(self.destination.exists())

    def test_complete_but_wrong_controller_version_rejects(self):
        (self.controller / 'short_refine_agent.py').write_text("VERSION = 'different.controller'\n")
        self.write_strategy()
        (self.native / 'receipt.json').unlink()
        self.native.rmdir()
        self.write_native(self.native)
        with self.assertRaisesRegex(ValueError, 'controller version does not match'):
            self.call()
        self.assertFalse(self.destination.exists())

    def test_native_receipt_case_must_match_requested_source(self):
        with self.assertRaisesRegex(ValueError, 'Native receipt case does not match'):
            self.call(source_row={**self.source, 'case_id': 'stack_cups_ep000099'})
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
