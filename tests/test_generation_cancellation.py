import asyncio
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
import torch

from irodori_tts.config import ModelConfig
from irodori_tts.gradio_generation import GenerationControl, stop_generation
from irodori_tts.inference_runtime import SamplingResult
from irodori_tts.meanflow import sample_euler_meanflow
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import sample_euler_rf_cfg


def tiny_model(flow):
    return TextToLatentRFDiT(
        ModelConfig(
            flow_parameterization=flow,
            latent_dim=4,
            model_dim=16,
            num_layers=1,
            num_heads=2,
            text_vocab_size=16,
            text_dim=16,
            text_layers=1,
            text_heads=2,
            speaker_dim=16,
            speaker_layers=1,
            speaker_heads=2,
            use_speaker_condition=False,
            timestep_embed_dim=16,
            adaln_rank=8,
        )
    ).eval()


class SamplerCancellationTests(unittest.TestCase):
    def test_stop_after_first_step_skips_remaining_model_calls(self):
        for flow, sampler in (
            ("rf_velocity", sample_euler_rf_cfg),
            ("meanflow", sample_euler_meanflow),
        ):
            with self.subTest(flow=flow):
                model = tiny_model(flow)
                calls = []
                forward = model.forward_with_encoded_conditions

                def run_forward(**kwargs):
                    calls.append(True)
                    return forward(**kwargs)

                def check_cancelled():
                    if calls:
                        raise InterruptedError("stop requested")

                inputs = dict(
                    model=model,
                    text_input_ids=torch.tensor([[1, 2]]),
                    text_mask=torch.ones((1, 2), dtype=torch.bool),
                    ref_latent=None,
                    ref_mask=None,
                    sequence_length=2,
                    num_steps=4,
                )
                try:
                    with patch.object(model, "forward_with_encoded_conditions", run_forward):
                        with self.assertRaises(InterruptedError):
                            sampler(**inputs, cancel_check=check_cancelled)
                except TypeError as exc:
                    self.fail(f"Sampler does not support stopping: {exc}")
                self.assertEqual(len(calls), 1)
                # Cancellation must leave the loaded model reusable.
                result = sampler(**inputs)
                self.assertEqual(tuple(result.shape), (1, 2, 4))
                self.assertTrue(torch.isfinite(result).all())


class GradioCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_button_is_available_in_both_uis(self):
        for module_name in ("gradio_app", "gradio_app_voicedesign"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                demo = module.build_ui(
                    default_checkpoint="Aratako/Irodori-TTS-v4-Large",
                    default_model_device="cpu",
                    default_model_precision="fp32",
                    default_codec_device="cpu",
                    default_codec_precision="fp32",
                )
                try:
                    stop = [b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Stop"]
                    self.assertEqual(len(stop), 1)
                    listeners = [f for f in demo.fns.values() if any(target[0] == stop[0]._id for target in f.targets)]
                    self.assertTrue(listeners)
                    self.assertTrue(all(not f.queue for f in listeners))
                finally:
                    demo.close()

    async def test_stop_preserves_outputs_and_allows_next_generation_in_both_uis(self):
        for module_name, length_check_enabled in (
            ("gradio_app", False),
            ("gradio_app", True),
            ("gradio_app_voicedesign", False),
        ):
            with self.subTest(module=module_name, length_check=length_check_enabled), TemporaryDirectory() as output_dir:
                module = importlib.import_module(module_name)
                demo = module.build_ui(
                    default_checkpoint="Aratako/Irodori-TTS-v4-Large",
                    default_model_device="cpu",
                    default_model_precision="fp32",
                    default_codec_device="cpu",
                    default_codec_precision="fp32",
                )
                functions = {f.fn.__name__: f for f in demo.fns.values() if f.fn}
                generate = functions["_run_generation"]
                state = demo.state_holder["cancel-test"]
                other_state = demo.state_holder["other-browser"]
                state_input = next(b for b in generate.inputs if isinstance(b, gr.State))
                other_control = other_state[state_input._id]
                other_control.begin()
                started = Event()
                release = Event()
                audio = torch.zeros((1, 256))

                class Runtime:
                    model_cfg = SimpleNamespace(
                        use_caption_condition=True, use_speaker_condition_resolved=False,
                        flow_parameterization="meanflow",
                    )

                    def synthesize(self, req, *, log_fn, cancel_check):
                        started.set()
                        while not release.wait(0.01):
                            cancel_check()
                        cancel_check()
                        return SamplingResult(audio, [audio], 24000, [], 0.1, 1, [])

                values = [b.value if not isinstance(b, gr.State) else None for b in generate.inputs]
                for i, block in enumerate(generate.inputs):
                    if block.label == "Text":
                        values[i] = "hello"
                    if block.label == "Length Check (two-pass)":
                        values[i] = length_check_enabled

                async def call(name, inputs, session=state):
                    return await demo.process_api(functions[name], inputs, state=session)

                task = None
                try:
                    with (
                        patch.object(module, "_build_runtime_key", return_value=object()),
                        patch.object(module, "get_cached_runtime", return_value=(Runtime(), False)),
                        patch.object(module, "Path", side_effect=lambda p: Path(output_dir) / p),
                    ):
                        begin = await call("begin_generation", [None])
                        self.assertFalse(begin["data"][0]["interactive"])
                        self.assertTrue(begin["data"][1]["interactive"])
                        task = asyncio.create_task(call("_run_generation", values))
                        self.assertTrue(await asyncio.to_thread(started.wait, 5))
                        stopped = await call("stop_generation", [None])
                        self.assertIn("Stop requested", stopped["data"][0])
                        cancelled = await asyncio.wait_for(asyncio.shield(task), 5)
                        self.assertEqual(cancelled["data"][-2], "Generation cancelled.")
                        self.assertTrue(all("value" not in update for update in cancelled["data"][:-2]))
                        self.assertEqual(list(Path(output_dir).rglob("*.wav")), [])
                        other_control.check()  # Stop must not affect a different browser session.
                        self.assertFalse(state[state_input._id].cancel())
                        ended = await call("end_generation", [])
                        self.assertTrue(ended["data"][0]["interactive"])
                        self.assertFalse(ended["data"][1]["interactive"])

                        # Use the same session and loaded runtime again after cancellation.
                        release.set()
                        await call("begin_generation", [None])
                        completed = await call("_run_generation", values)
                        self.assertIn("saved[1]", completed["data"][-2])
                        self.assertEqual(len(list(Path(output_dir).rglob("*.wav"))), 1)
                finally:
                    release.set()
                    if task and not task.done():
                        await asyncio.wait_for(task, 5)
                    demo.close()

    async def test_stop_before_worker_starts_does_not_load_a_model(self):
        from irodori_tts.gradio_generation import begin_generation, cancellable_generation

        control = GenerationControl()
        begin_generation(control)
        stop_generation(control)

        @cancellable_generation(audio_outputs=1)
        def generate(generation_control):
            self.fail("Cancelled queued generation should not start")

        result = generate(control)
        self.assertEqual(result[1], "Generation cancelled.")
        # A second request must not retain the previous stop signal.
        begin_generation(control)
        control.check()

    async def test_exception_releases_session_for_next_generation(self):
        from irodori_tts.gradio_generation import begin_generation, cancellable_generation

        control = GenerationControl()

        @cancellable_generation(audio_outputs=1)
        def generate(generation_control):
            raise ValueError("invalid request")

        begin_generation(control)
        with self.assertRaisesRegex(ValueError, "invalid request"):
            generate(control)
        begin_generation(control)
        control.check()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
