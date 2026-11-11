"""Regression coverage for fork cancellation across upstream inference paths."""

import unittest
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_generation_cancellation import tiny_model

from irodori_tts import length_check
from irodori_tts.gradio_generation import GenerationCancelled
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest


class Tokenizer:
    def batch_encode(self, texts, max_length):
        return torch.tensor([[1, 2]] * len(texts)), torch.ones((len(texts), 2), dtype=torch.bool)


class Codec:
    sample_rate = 8
    model = SimpleNamespace(hop_length=2)

    def __init__(self):
        self.decoded = []
        self.cancel_after_decode = None

    def decode_latent(self, latent):
        self.decoded.append(tuple(latent.shape))
        if self.cancel_after_decode is not None:
            self.cancel_after_decode.set()
        return latent.transpose(1, 2)[:, :1].repeat_interleave(2, dim=2)


def runtime_for(flow):
    model = tiny_model(flow)
    codec = Codec()
    # Watermark initialization can load external weights; test the inference paths locally.
    with patch(
        "irodori_tts.inference_runtime.SilentCipherWatermarker",
        return_value=SimpleNamespace(ready=False),
    ):
        return InferenceRuntime(
            key=RuntimeKey(checkpoint="test", model_device="cpu"),
            model_cfg=model.cfg,
            train_cfg=None,
            model=model,
            tokenizer=Tokenizer(),
            caption_tokenizer=None,
            codec=codec,
            default_text_max_len=4,
            default_caption_max_len=4,
        )


def check_event(event):
    if event.is_set():
        raise GenerationCancelled()


class RuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cancel_during_sampling_releases_runtime_for_single_and_batch_calls(self):
        for flow in ("rf_velocity", "meanflow"):
            for batched in (False, True):
                with self.subTest(flow=flow, batched=batched):
                    self._check_runtime_cancellation(flow, batched)

    def _check_runtime_cancellation(self, flow, batched):
        runtime = runtime_for(flow)
        request = SamplingRequest(
            text="hello", no_ref=True, seconds=0.5, num_steps=4, seed=7, trim_tail=False
        )
        requests = [request, replace(request, seconds=1.0, seed=8)]
        cancelled = Event()
        forward = runtime.model.forward_with_encoded_conditions

        def run_forward(**kwargs):
            result = forward(**kwargs)
            cancelled.set()
            return result

        def synthesize(**kwargs):
            if batched:
                return runtime.synthesize_batch(requests, **kwargs)
            return [runtime.synthesize(request, **kwargs)]

        with patch.object(runtime.model, "forward_with_encoded_conditions", run_forward):
            with self.assertRaises(GenerationCancelled):
                synthesize(cancel_check=lambda: check_event(cancelled))
        self.assertEqual(runtime.codec.decoded, [])
        self.assertFalse(runtime._infer_lock.locked())
        results = synthesize()
        self.assertEqual([r.audio.shape[-1] for r in results], [4, 8] if batched else [4])
        self.assertTrue(all(torch.isfinite(r.audio).all() for r in results))

    def test_stop_during_decode_does_not_decode_the_next_candidate(self):
        runtime = runtime_for("rf_velocity")
        cancelled = Event()
        runtime.codec.cancel_after_decode = cancelled
        request = SamplingRequest(
            text="hello",
            no_ref=True,
            seconds=0.5,
            num_steps=1,
            num_candidates=3,
            seed=7,
            trim_tail=False,
        )
        with self.assertRaises(GenerationCancelled):
            runtime.synthesize(request, cancel_check=lambda: check_event(cancelled))
        self.assertEqual(len(runtime.codec.decoded), 1)
        self.assertFalse(runtime._infer_lock.locked())

    def test_pre_cancelled_batch_does_not_start_model_or_decode(self):
        runtime = runtime_for("meanflow")
        cancelled = Event()
        cancelled.set()
        with self.assertRaises(GenerationCancelled):
            runtime.synthesize_batch(
                [SamplingRequest(text="hello", no_ref=True)],
                cancel_check=lambda: check_event(cancelled),
            )
        self.assertEqual(runtime.codec.decoded, [])

    def test_batched_results_keep_lengths_and_seeds(self):
        for flow in ("rf_velocity", "meanflow"):
            with self.subTest(flow=flow):
                runtime = runtime_for(flow)
                requests = [
                    SamplingRequest(
                        text="hello",
                        no_ref=True,
                        seconds=seconds,
                        num_steps=2,
                        seed=seed,
                        trim_tail=False,
                    )
                    for seconds, seed in ((0.5, 7), (1.0, 8))
                ]
                results = runtime.synthesize_batch(requests)
                self.assertEqual([r.audio.shape[-1] for r in results], [4, 8])
                self.assertEqual([r.used_seed for r in results], [7, 8])


class LengthCheckCancellationTests(unittest.TestCase):
    def test_cancellation_escapes_every_pass_and_removes_attention_hooks(self):
        for cancel_at in (1, 2, 3, 4):
            with self.subTest(cancel_at=cancel_at):
                self._check_length_cancellation(cancel_at)

    def _check_length_cancellation(self, cancel_at):
        model = tiny_model("rf_velocity")
        attention = model.blocks[0].attention
        original_hooks = len(attention._forward_pre_hooks)
        cancelled = Event()

        class Runtime:
            model_cfg = SimpleNamespace(latent_patch_size=1)
            codec = Codec()

            def __init__(self):
                self.model = model
                self.calls = 0

            def synthesize(self, request, *, log_fn=None, cancel_check=None):
                self.calls += 1
                if self.calls == cancel_at:
                    cancelled.set()
                if cancel_check is not None:
                    cancel_check()
                # Simulate an overlong first pass and two too-short retries.
                if self.calls == 1:
                    raise length_check._Stop(4, 100)
                if self.calls in (2, 3):
                    raise length_check._Stop()
                return object()

        runtime = Runtime()
        with patch.object(length_check, "supported", return_value=True):
            with self.assertRaises(GenerationCancelled):
                length_check.synthesize(
                    runtime,
                    SamplingRequest(text="hello", num_steps=1, seed=7),
                    cancel_check=lambda: check_event(cancelled),
                )
        self.assertEqual(runtime.calls, cancel_at)
        self.assertEqual(len(attention._forward_pre_hooks), original_hooks)


if __name__ == "__main__":
    unittest.main()
