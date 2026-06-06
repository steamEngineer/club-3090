# EVAL-ONLY workaround for vLLM #44494 (gemma4_unified compressed-tensors).
# Gemma4UnifiedVisionEmbedder builds patch_dense as ColumnParallelLinear(quant_config=...)
# with NO prefix=, so compressed-tensors can't match the checkpoint's `ignore` list and
# force-quantizes the vision embedder — which the QAT-w4a16 checkpoint stores as BF16 ->
# "no parameter vision_embedder.patch_dense.weight" at load. Force the embedder unquantized.
# Auto-imported by Python at startup when this dir is on PYTHONPATH (incl. spawn workers).
# Proper upstream fix = plumb prefix= so the ignore matches (see vLLM #44494). NOT for prod.
import importlib.abc
import importlib.machinery
import sys

_TARGET = "vllm.model_executor.models.gemma4_unified"


class _UnquantizeVisionEmbedder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)  # one-shot
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        _exec = spec.loader.exec_module

        def exec_module(module):
            _exec(module)
            cls = getattr(module, "Gemma4UnifiedVisionEmbedder", None)
            if cls is None:
                return
            _orig = cls.__init__

            def __init__(self, config, quant_config=None):
                # drop quant_config: the vision embedder is stored BF16, keep it that way
                _orig(self, config, quant_config=None)

            cls.__init__ = __init__
            print("[g4-unified-patch] Gemma4UnifiedVisionEmbedder forced unquantized (vLLM #44494)", flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _UnquantizeVisionEmbedder())
