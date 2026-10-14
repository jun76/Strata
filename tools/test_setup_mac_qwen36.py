"""Qwen3.6 Metal setup guardrails and opt-in dispatch; no GPU or downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'metal'))
import setup_mac as M

class Qwen36Mac(unittest.TestCase):
    def test_context_boundaries(self):
        self.assertEqual(M.q36_args(Path('/model.gguf'),4096),['--native','/model.gguf','--max-context','4096'])
        for n in (0,-1,4097,32768):
            with self.assertRaises(ValueError): M.q36_args(Path('/model.gguf'),n)

    def test_explicit_family_is_not_skipped_when_another_model_is_installed(self):
        info={'chip':'Apple M1 Pro','ram_gb':16,'macos':'27.0.1','arm64':True,'clt':'/tools','perf_cores':6}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(M,'mac_configs',return_value=[Path('/existing.json')]), \
             mock.patch.object(M,'mac_info',return_value=info), \
             mock.patch.object(M.S,'data_folder',return_value=(Path(td),[])), \
             mock.patch.object(M.S,'free_gb',return_value=100), \
             mock.patch.object(M,'setup_q36',return_value=0) as install, \
             mock.patch.object(M,'start') as start:
            self.assertEqual(M.main(['--family','qwen36','--yes','--no-start']),0)
            install.assert_called_once();start.assert_not_called()

    def test_invalid_context_fails_before_download_or_build(self):
        from argparse import Namespace
        args=Namespace(model=None,context=32768,vision='no')
        def fail(message,*a):raise ValueError(message)
        with mock.patch.object(M.S,'fail',side_effect=fail),mock.patch.object(M.S,'download') as download,mock.patch.object(M,'build_q36_engine') as build:
            with self.assertRaisesRegex(ValueError,'1..4096'):M.setup_q36(args,{'ram_gb':16},Path('/data'),Path('/models'))
            download.assert_not_called();build.assert_not_called()

if __name__=='__main__':unittest.main()
