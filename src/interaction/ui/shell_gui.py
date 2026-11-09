"""Universal workspace GUI demo for the multimodal shell runtime."""

from __future__ import annotations

from dataclasses import dataclass

from interaction.contracts import EnvironmentSnapshot, SessionMode
from interaction.feedback import VoiceFeedbackEvent, VoiceLoopPhase
from interaction.runtime import VoiceCommandLoop

from .overlay import ConsoleOverlayRenderer, OverlayController
from .shell import (
    ShellSurface,
    ShellViewState,
    TerminalShellRenderer,
    build_shell_view_state,
    infer_shell_surface,
)


@dataclass(frozen=True)
class ShellDemoRunResult:
    shell_state: ShellViewState
    shell_rendered: str
    overlay_rendered: str
    event_log: tuple[str, ...]


class ShellDemoController:
    """Drive a universal shell demo through the existing voice runtime."""

    def __init__(
        self,
        *,
        loop: VoiceCommandLoop,
        environment: EnvironmentSnapshot,
        mode: SessionMode,
        fallback_to_dictation: bool,
        surface_override: str | None = None,
        source: str = "voice",
    ) -> None:
        self.loop = loop
        self.environment = environment
        self.mode = mode
        self.fallback_to_dictation = fallback_to_dictation
        self.surface_override = surface_override
        self.source = source
        self.pointer_mode = "semantic"
        self.selection_label: str | None = None
        self.events: list[object] = []
        self.overlay = OverlayController()
        self.overlay_renderer = ConsoleOverlayRenderer()
        self.shell_renderer = TerminalShellRenderer()
        self._sync_surface_defaults(preserve_selection=False)

    def current_result(self) -> ShellDemoRunResult:
        shell_state = build_shell_view_state(
            source=self.source,
            overlay_state=self.overlay.state,
            environment=self.environment,
            mode=self.mode,
            active_scope=self._active_scope(),
            surface_override=self.surface_override,
            pointer_mode=self.pointer_mode,
            selection_label=self.selection_label,
        )
        return ShellDemoRunResult(
            shell_state=shell_state,
            shell_rendered=self.shell_renderer.render(shell_state),
            overlay_rendered=self.overlay_renderer.render(self.overlay.state),
            event_log=tuple(self._event_log_lines()),
        )

    def run_utterance(self, utterance: str) -> ShellDemoRunResult:
        normalized = " ".join(utterance.strip().split())
        if not normalized:
            return self.current_result()

        local_events = self._handle_local_command(normalized)
        if local_events is not None:
            self.events.extend(local_events)
            for event in local_events:
                self.overlay.apply_event(self.source, event)
            self._sync_surface_defaults()
            return self.current_result()

        events = self.loop.run_text_turn(
            normalized,
            self.environment,
            mode=self.mode,
            fallback_to_dictation=self.fallback_to_dictation,
        )
        self.events.extend(events)
        for event in events:
            self.overlay.apply_event(self.source, event)
        self._transition_environment(normalized)
        self._sync_surface_defaults()
        return self.current_result()

    def _active_scope(self) -> str | None:
        snapshot = self.loop.state_snapshot()
        active_scope = snapshot.get("active_scope")
        return active_scope if isinstance(active_scope, str) else None

    def _event_log_lines(self) -> list[str]:
        lines: list[str] = []
        for event in self.events[-12:]:
            phase = getattr(event, "phase", None)
            phase_value = getattr(phase, "value", phase)
            message = str(getattr(event, "message", "") or "")
            transcript = str(getattr(event, "transcript", "") or "")
            if transcript:
                lines.append(f"[{phase_value}] {message} :: {transcript}")
            else:
                lines.append(f"[{phase_value}] {message}")
        return lines

    def _handle_local_command(self, utterance: str) -> list[object] | None:
        lowered = utterance.casefold()
        if lowered in {"workspace", "open workspace", "go workspace", "go home", "home"}:
            self._set_surface(ShellSurface.WORKSPACE)
            return self._shell_local_events(utterance, "Workspace surface is active.")
        if lowered in {"browser", "open browser", "go browser"}:
            self._set_surface(ShellSurface.BROWSER)
            return self._shell_local_events(utterance, "Browser surface is active.")
        if lowered in {"notes", "open notes lane", "go notes"}:
            self._set_surface(ShellSurface.NOTES)
            return self._shell_local_events(utterance, "Notes surface is active.")
        if lowered in {"precision mode", "exact mode"}:
            self.pointer_mode = "precision"
            self.selection_label = None
            return self._shell_local_events(
                utterance,
                "Precision layout preview only. Live cursor control is available through gaze-live, not this demo.",
            )
        if lowered in {"semantic mode", "target mode"}:
            self.pointer_mode = "semantic"
            self.selection_label = None
            return self._shell_local_events(
                utterance,
                "Semantic mode is active. Large landmarks are primary again.",
            )

        focus_label = _selection_for_command(lowered)
        if focus_label is not None:
            self.selection_label = focus_label
            if lowered in {"focus cursor", "cursor here", "caret line start", "caret line end"}:
                self.pointer_mode = "precision"
            return self._shell_local_events(utterance, f'Selected demo landmark "{focus_label}"; no desktop pointer action was performed.')
        return None

    def _shell_local_events(self, utterance: str, message: str) -> list[object]:
        return [
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.INTERPRETING,
                transcript=utterance,
                message=message,
            ),
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.IDLE,
                message="Shell workspace is idle.",
            ),
        ]

    def _transition_environment(self, utterance: str) -> None:
        lowered = utterance.casefold()
        if lowered in {"open notes", "switch notes", "switch to notes", "go to notes"}:
            self._set_surface(ShellSurface.NOTES)
            return
        if lowered in {
            "open safari",
            "switch safari",
            "switch to safari",
            "go to safari",
            "open browser",
            "switch browser",
        }:
            self._set_surface(ShellSurface.BROWSER)

    def _set_surface(self, surface: ShellSurface) -> None:
        self.surface_override = surface.value
        if surface == ShellSurface.NOTES:
            self.environment = EnvironmentSnapshot(
                active_app="Notes",
                active_bundle_id="com.apple.Notes",
                active_window_title="Shell Notes",
                active_profile_id="notes_family",
            )
            return
        if surface == ShellSurface.BROWSER:
            self.environment = EnvironmentSnapshot(
                active_app="Safari",
                active_bundle_id="com.apple.Safari",
                active_window_title="Shell Browser",
                active_profile_id="safari_browser_family",
            )
            return
        self.environment = EnvironmentSnapshot(
            active_app="Finder",
            active_bundle_id="com.apple.finder",
            active_window_title="Shell Workspace",
        )

    def _sync_surface_defaults(self, *, preserve_selection: bool = True) -> None:
        if self.surface_override is None:
            inferred_surface = infer_shell_surface(self.environment)
            self.surface_override = inferred_surface.value
        shell_state = build_shell_view_state(
            source=self.source,
            overlay_state=self.overlay.state,
            environment=self.environment,
            mode=self.mode,
            active_scope=self._active_scope(),
            surface_override=self.surface_override,
            pointer_mode=self.pointer_mode,
            selection_label=self.selection_label,
        )
        valid_labels = {landmark.label for landmark in shell_state.landmarks}
        if not preserve_selection or self.selection_label not in valid_labels:
            self.selection_label = next(
                (landmark.label for landmark in shell_state.landmarks if landmark.active),
                shell_state.landmarks[0].label if shell_state.landmarks else None,
            )


def run_shell_gui_demo(
    controller: ShellDemoController,
    *,
    title: str = "Interaction Universal Workspace",
) -> ShellDemoRunResult:
    import tkinter as tk

    result = controller.current_result()
    root = tk.Tk()
    root.title(title)
    root.geometry("1440x920")
    root.configure(bg="#f3ede2")

    header_var = tk.StringVar()
    subheader_var = tk.StringVar()
    status_var = tk.StringVar()
    transcript_var = tk.StringVar()
    target_var = tk.StringVar()
    input_var = tk.StringVar()

    header = tk.Label(
        root,
        textvariable=header_var,
        font=("Helvetica", 24, "bold"),
        bg="#f3ede2",
        fg="#14213d",
        anchor="w",
    )
    header.pack(fill="x", padx=24, pady=(20, 4))

    subheader = tk.Label(
        root,
        textvariable=subheader_var,
        font=("Helvetica", 11),
        bg="#f3ede2",
        fg="#49566c",
        anchor="w",
        justify="left",
    )
    subheader.pack(fill="x", padx=24)

    tab_frame = tk.Frame(root, bg="#f3ede2")
    tab_frame.pack(fill="x", padx=24, pady=(18, 8))

    body = tk.Frame(root, bg="#f3ede2")
    body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

    left = tk.Frame(body, bg="#f3ede2")
    left.pack(side="left", fill="both", expand=True)
    center = tk.Frame(body, bg="#f3ede2")
    center.pack(side="left", fill="both", expand=True, padx=(18, 0))
    right = tk.Frame(body, bg="#f3ede2")
    right.pack(side="left", fill="both", expand=True, padx=(18, 0))

    pane_frame = tk.Frame(left, bg="#f3ede2")
    pane_frame.pack(fill="both", expand=True)

    landmark_card = tk.LabelFrame(
        center,
        text="Landmark Grid",
        font=("Helvetica", 13, "bold"),
        bg="#102542",
        fg="#f6f2e8",
        padx=12,
        pady=12,
    )
    landmark_card.pack(fill="both", expand=True)
    landmark_grid = tk.Frame(landmark_card, bg="#102542")
    landmark_grid.pack(fill="both", expand=True)

    status_card = tk.LabelFrame(
        right,
        text="Workspace Status",
        font=("Helvetica", 13, "bold"),
        bg="#fff9f0",
        fg="#14213d",
        padx=12,
        pady=12,
    )
    status_card.pack(fill="x", pady=(0, 12))
    tk.Label(status_card, textvariable=status_var, font=("Helvetica", 12, "bold"), bg="#fff9f0", fg="#14213d", anchor="w", justify="left", wraplength=360).pack(fill="x")
    tk.Label(status_card, textvariable=transcript_var, font=("Helvetica", 11), bg="#fff9f0", fg="#49566c", anchor="w", justify="left", wraplength=360).pack(fill="x", pady=(8, 0))
    tk.Label(status_card, textvariable=target_var, font=("Helvetica", 11), bg="#fff9f0", fg="#49566c", anchor="w", justify="left", wraplength=360).pack(fill="x", pady=(6, 0))

    shell_card = tk.LabelFrame(
        right,
        text="Shell State",
        font=("Helvetica", 13, "bold"),
        bg="#14213d",
        fg="#f6f2e8",
        padx=10,
        pady=10,
    )
    shell_card.pack(fill="both", expand=True, pady=(0, 12))
    shell_text = tk.Text(
        shell_card,
        height=18,
        bg="#14213d",
        fg="#f0f4f8",
        insertbackground="#f0f4f8",
        font=("Menlo", 11),
        wrap="word",
        relief="flat",
        padx=8,
        pady=8,
    )
    shell_text.pack(fill="both", expand=True)

    log_card = tk.LabelFrame(
        right,
        text="Recent Events",
        font=("Helvetica", 13, "bold"),
        bg="#fff9f0",
        fg="#14213d",
        padx=10,
        pady=10,
    )
    log_card.pack(fill="both", expand=True)
    log_text = tk.Text(
        log_card,
        height=12,
        bg="#fff9f0",
        fg="#14213d",
        font=("Menlo", 10),
        wrap="word",
        relief="flat",
        padx=8,
        pady=8,
    )
    log_text.pack(fill="both", expand=True)

    footer = tk.Frame(root, bg="#f3ede2")
    footer.pack(fill="x", padx=24, pady=(0, 20))
    entry = tk.Entry(
        footer,
        textvariable=input_var,
        font=("Helvetica", 14),
        bg="#fff9f0",
        fg="#14213d",
        relief="flat",
        highlightthickness=1,
        highlightbackground="#a7b7cc",
        highlightcolor="#eb5e28",
    )
    entry.pack(side="left", fill="x", expand=True, padx=(0, 12), ipady=8)

    def _update_view(run_result: ShellDemoRunResult) -> None:
        shell_state = run_result.shell_state
        header_var.set(f"{shell_state.workspace_title} · {shell_state.title}")
        subheader_var.set(
            "Layout demo with typed commands. Microphone, gaze input, and precision landmarks are not connected here.\n"
            f"App: {shell_state.active_app or '-'}    Profile: {shell_state.active_profile_id or '-'}    "
            f"Mode: {shell_state.mode or '-'}    Scope: {shell_state.active_scope or '-'}    Pointer: {shell_state.pointer_mode}"
        )
        status_var.set(shell_state.message or "No activity yet.")
        transcript_var.set(f"Transcript: {shell_state.transcript or '-'}")
        target_var.set(
            f"Selected landmark: {shell_state.selection_label or '-'}    "
            f"Grounded target: {shell_state.target_label or '-'}    "
            f"Pending confirmation: {'yes' if shell_state.pending_confirmation else 'no'}"
        )

        shell_text.configure(state="normal")
        shell_text.delete("1.0", "end")
        shell_text.insert("1.0", run_result.shell_rendered)
        shell_text.configure(state="disabled")

        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.insert("1.0", "\n".join(run_result.event_log) or "No commands run yet.")
        log_text.configure(state="disabled")

        for child in tab_frame.winfo_children():
            child.destroy()
        for tab in shell_state.tabs:
            button = tk.Button(
                tab_frame,
                text=tab.title,
                font=("Helvetica", 12, "bold"),
                bg="#eb5e28" if tab.active else "#e1d8c8",
                fg="#ffffff" if tab.active else "#14213d",
                activebackground="#c94b1c" if tab.active else "#d0c3af",
                activeforeground="#ffffff" if tab.active else "#14213d",
                relief="flat",
                padx=18,
                pady=10,
                command=lambda utterance=tab.utterance: _run_command(utterance),
            )
            button.pack(side="left", padx=(0, 10))

        for child in pane_frame.winfo_children():
            child.destroy()
        for pane in shell_state.panes:
            pane_card = tk.LabelFrame(
                pane_frame,
                text=pane.title,
                font=("Helvetica", 12, "bold"),
                bg="#fff9f0" if pane.active else "#f7f1e5",
                fg="#14213d",
                padx=12,
                pady=12,
            )
            pane_card.pack(fill="x", pady=(0, 12))
            tk.Label(
                pane_card,
                text=f"{pane.role.upper()} · {pane.summary}",
                font=("Helvetica", 10),
                bg="#fff9f0" if pane.active else "#f7f1e5",
                fg="#49566c",
                wraplength=360,
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=(0, 10))
            for command in pane.commands:
                button = tk.Button(
                    pane_card,
                    text=f"{command.utterance}\n{command.label}",
                    font=("Helvetica", 11, "bold"),
                    bg="#2f6fed" if command.hot_path else "#6b7a90",
                    fg="#ffffff",
                    activebackground="#214fb6" if command.hot_path else "#526072",
                    activeforeground="#ffffff",
                    relief="flat",
                    padx=10,
                    pady=10,
                    wraplength=320,
                    justify="center",
                    command=lambda utterance=command.utterance: _run_command(utterance),
                )
                button.pack(fill="x", pady=(0, 8))

        for child in landmark_grid.winfo_children():
            child.destroy()
        for index, landmark in enumerate(shell_state.landmarks):
            row = index // 2
            column = index % 2
            card = tk.Button(
                landmark_grid,
                text=(
                    f"{landmark.label}\n{landmark.detail}\n"
                    f"Command: {landmark.utterance}"
                ),
                font=("Helvetica", 11, "bold"),
                bg="#f4a261" if landmark.active else ("#335c67" if landmark.exact else "#1f3b73"),
                fg="#ffffff",
                activebackground="#eb5e28" if landmark.active else ("#27444d" if landmark.exact else "#17305e"),
                activeforeground="#ffffff",
                relief="flat",
                padx=14,
                pady=16,
                wraplength=250,
                justify="left",
                anchor="w",
                command=lambda utterance=landmark.utterance: _run_command(utterance),
            )
            card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        for column in range(2):
            landmark_grid.grid_columnconfigure(column, weight=1)

    def _run_command(utterance: str | None = None) -> None:
        command = utterance or input_var.get().strip()
        if not command:
            return
        input_var.set("")
        _update_view(controller.run_utterance(command))

    run_button = tk.Button(
        footer,
        text="Run Command",
        font=("Helvetica", 12, "bold"),
        bg="#14213d",
        fg="#ffffff",
        activebackground="#0d172a",
        activeforeground="#ffffff",
        relief="flat",
        padx=18,
        pady=10,
        command=_run_command,
    )
    run_button.pack(side="left")
    entry.bind("<Return>", lambda _event: _run_command())

    _update_view(result)
    entry.focus_set()
    root.mainloop()
    return controller.current_result()


def _selection_for_command(command: str) -> str | None:
    selection_map = {
        "focus browser lane": "Browser Lane",
        "focus notes lane": "Notes Lane",
        "focus command rail": "Command Rail",
        "focus precision rail": "Precision Rail",
        "focus tab strip": "Tab Strip",
        "focus search bar": "Search Bar",
        "focus address bar": "Search Bar",
        "focus document": "Primary Document",
        "focus note list": "Notes List",
        "focus editor block": "Editor Block",
        "focus cursor": "Cursor Anchor",
        "cursor here": "Cursor Anchor",
        "caret line start": "Line Start",
        "caret line end": "Line End",
    }
    return selection_map.get(command)
