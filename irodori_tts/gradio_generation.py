"""Session-local, cooperative cancellation for the Gradio generation callbacks."""

from collections.abc import Callable
from functools import wraps
from inspect import signature
from threading import Event, Lock

import gradio as gr


class GenerationCancelled(Exception):
    """Raised at a safe boundary when the user requests cancellation."""


class GenerationControl:
    def __init__(self):
        self._lock = Lock()
        self._cancelled = Event()
        self._active = False

    def __deepcopy__(self, memo):
        # Gradio copies the initial State for each browser session.
        return type(self)()

    def begin(self, *, if_idle: bool = False) -> None:
        with self._lock:
            if self._active:
                if if_idle:
                    return
                raise gr.Error("A generation is already in progress.")
            self._cancelled.clear()
            self._active = True

    def finish(self) -> None:
        with self._lock:
            self._active = False

    def cancel(self) -> bool:
        with self._lock:
            if not self._active:
                return False
            self._cancelled.set()
            return True

    def check(self) -> None:
        if self._cancelled.is_set():
            raise GenerationCancelled()


def begin_generation(generation_control: GenerationControl):
    generation_control.begin()
    return gr.update(interactive=False), gr.update(interactive=True), "Generating..."


def end_generation():
    return gr.update(interactive=True), gr.update(interactive=False)


def stop_generation(generation_control: GenerationControl):
    if generation_control.cancel():
        return "Stop requested. Waiting for the current operation to finish..."
    return gr.skip()


def cancellable_generation(audio_outputs: int) -> Callable:
    def decorate(fn):
        fn_signature = signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            control = fn_signature.bind(*args, **kwargs).arguments["generation_control"]
            # Also support direct API calls that did not run begin_generation.
            control.begin(if_idle=True)
            try:
                control.check()
                return fn(*args, **kwargs)
            except GenerationCancelled:
                return (*[gr.skip() for _ in range(audio_outputs)], "Generation cancelled.", gr.skip())
            finally:
                control.finish()

        return wrapped

    return decorate
