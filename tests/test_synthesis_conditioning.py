"""CPU regressions for visual-token conditioning, without loading SDXL weights."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "PyTorch is needed for CPU conditioning checks")
class SynthesisConditioningTests(unittest.TestCase):
    def setUp(self):
        # Compile the actual helper in isolation: no optional diffusion imports,
        # downloads, model construction, or GPU allocation are needed.
        tree = ast.parse((ROOT / "diffusion/synthesize_from_reference.py").read_text(encoding="utf-8"))
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_prepare_sdxl_conditioning")
        self.globals = {"torch": torch, "Tuple": Tuple, "StableDiffusionXLPipeline": object}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "synthesis_conditioning", "exec"), self.globals)
        self.condition = self.globals["_prepare_sdxl_conditioning"]
        self.positive = torch.ones(1, 3, 4)
        self.negative = torch.zeros(1, 3, 4)
        self.pooled = torch.ones(1, 2)
        self.negative_pooled = torch.zeros(1, 2)
        self.tokens = torch.full((1, 2, 4), 7.0)
        self.pipe = SimpleNamespace(
            text_encoder_2=SimpleNamespace(config=SimpleNamespace(projection_dim=2)),
            _get_add_time_ids=lambda **kwargs: torch.zeros(1, 6),
            encode_prompt=lambda **kwargs: (
                self.positive,
                self.negative if kwargs["do_classifier_free_guidance"] else None,
                self.pooled,
                self.negative_pooled if kwargs["do_classifier_free_guidance"] else None,
            ),
        )

    def call(self, guidance):
        return self.condition(self.pipe, "", guidance, device="cpu", dtype=torch.float32,
                              crop_coords=(0, 0), concept_tokens=self.tokens)

    def test_non_cfg_inversion_uses_positive_pooled_embedding_and_tokens(self):
        embeddings, pooled, added, cfg = self.call(0.0)
        self.assertFalse(cfg)
        self.assertTrue(torch.equal(pooled, self.pooled))
        self.assertEqual(tuple(embeddings.shape), (1, 5, 4))
        self.assertTrue(torch.equal(embeddings[:, 3:], self.tokens))
        self.assertEqual(tuple(added["time_ids"].shape), (1, 6))

    def test_cfg_generation_preserves_both_branches_and_visual_tokens(self):
        embeddings, pooled, added, cfg = self.call(7.5)
        self.assertTrue(cfg)
        self.assertEqual(tuple(embeddings.shape), (2, 5, 4))
        self.assertTrue(torch.equal(embeddings[0:1, :3], self.negative))
        self.assertTrue(torch.equal(embeddings[1:2, :3], self.positive))
        self.assertTrue(torch.equal(embeddings[:, 3:], self.tokens.repeat(2, 1, 1)))
        self.assertTrue(torch.equal(pooled, torch.cat([self.negative_pooled, self.pooled])))
        self.assertEqual(tuple(added["time_ids"].shape), (2, 6))

    def test_manual_prompt_fallback_retains_visual_tokens(self):
        self.pipe.encode_prompt = lambda **kwargs: (self.positive,)
        self.globals["_encode_prompt_manual"] = lambda *args: (
            self.positive, self.pooled, {"time_ids": torch.zeros(1, 6)}, False
        )
        embeddings, _, _, _ = self.call(0.0)
        self.assertEqual(tuple(embeddings.shape), (1, 5, 4))
        self.assertTrue(torch.equal(embeddings[:, 3:], self.tokens))


if __name__ == "__main__":
    unittest.main()
