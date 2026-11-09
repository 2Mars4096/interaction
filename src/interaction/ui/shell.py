"""Structured shell view helpers for multimodal runtime state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

from interaction.contracts import EnvironmentSnapshot, SessionMode

from .overlay import OverlayState


class ShellSurface(StrEnum):
    WORKSPACE = "workspace"
    BROWSER = "browser"
    NOTES = "notes"


@dataclass(frozen=True)
class ShellCommandHint:
    utterance: str
    label: str
    hot_path: bool = True


@dataclass(frozen=True)
class ShellPanel:
    title: str
    subtitle: str | None = None
    commands: tuple[ShellCommandHint, ...] = ()


@dataclass(frozen=True)
class ShellWorkspaceTab:
    surface: str
    title: str
    utterance: str
    active: bool = False


@dataclass(frozen=True)
class ShellWorkspacePane:
    title: str
    role: str
    summary: str
    active: bool = False
    commands: tuple[ShellCommandHint, ...] = ()


@dataclass(frozen=True)
class ShellLandmark:
    label: str
    utterance: str
    detail: str
    exact: bool = False
    active: bool = False


@dataclass(frozen=True)
class ShellViewState:
    source: str
    surface: str
    title: str
    workspace_title: str = "Universal Workspace"
    active_app: str | None = None
    active_window_title: str | None = None
    active_profile_id: str | None = None
    mode: str | None = None
    active_scope: str | None = None
    pointer_mode: str = "semantic"
    selection_label: str | None = None
    exact_pointing_available: bool = False
    message: str | None = None
    transcript: str | None = None
    target_label: str | None = None
    pending_confirmation: bool = False
    tabs: tuple[ShellWorkspaceTab, ...] = field(default_factory=tuple)
    panes: tuple[ShellWorkspacePane, ...] = field(default_factory=tuple)
    landmarks: tuple[ShellLandmark, ...] = field(default_factory=tuple)
    panels: tuple[ShellPanel, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TerminalShellRenderer:
    """Render a compact shell-style status view for terminal workflows."""

    def render(self, state: ShellViewState) -> str:
        lines = [
            "== Interaction Shell ==",
            f"workspace: {state.workspace_title} | pointer: {state.pointer_mode}",
            f"surface: {state.title} | source: {state.source}",
            f"app: {state.active_app or '-'} | window: {state.active_window_title or '-'}",
            f"profile: {state.active_profile_id or '-'} | mode: {state.mode or '-'} | scope: {state.active_scope or '-'}",
            f"selection: {state.selection_label or '-'} | exact_pointing: {'yes' if state.exact_pointing_available else 'no'}",
            f"message: {state.message or '-'}",
            f"transcript: {state.transcript or '-'}",
            f"target: {state.target_label or '-'} | pending_confirmation: {'yes' if state.pending_confirmation else 'no'}",
            "",
            "[Tabs]",
        ]
        for tab in state.tabs:
            marker = ">" if tab.active else "-"
            lines.append(f"{marker} {tab.title} -> {tab.utterance}")
        lines.append("")
        lines.append("[Panes]")
        for pane in state.panes:
            marker = ">" if pane.active else "-"
            lines.append(f"{marker} {pane.title} ({pane.role})")
            lines.append(f"  {pane.summary}")
            for command in pane.commands:
                command_marker = "*" if command.hot_path else "-"
                lines.append(f"  {command_marker} {command.utterance} -> {command.label}")
        lines.append("")
        lines.append("[Landmarks]")
        for landmark in state.landmarks:
            marker = ">" if landmark.active else "-"
            precision = " exact" if landmark.exact else ""
            lines.append(f"{marker} {landmark.label}{precision} -> {landmark.utterance}")
            lines.append(f"  {landmark.detail}")
        for panel in state.panels:
            lines.append("")
            lines.append(f"[{panel.title}]")
            if panel.subtitle:
                lines.append(panel.subtitle)
            for command in panel.commands:
                marker = "*" if command.hot_path else "-"
                lines.append(f"{marker} {command.utterance} -> {command.label}")
        return "\n".join(lines)


def build_shell_view_state(
    *,
    source: str,
    overlay_state: OverlayState,
    environment: EnvironmentSnapshot,
    mode: SessionMode | None = None,
    active_scope: str | None = None,
    surface_override: str | None = None,
    pointer_mode: str = "semantic",
    selection_label: str | None = None,
) -> ShellViewState:
    surface = infer_shell_surface(environment, override=surface_override)
    landmarks = _surface_landmarks(surface, pointer_mode=pointer_mode, selection_label=selection_label)
    resolved_selection = selection_label or next((landmark.label for landmark in landmarks if landmark.active), None)
    return ShellViewState(
        source=source,
        surface=surface.value,
        title=_surface_title(surface),
        active_app=environment.active_app,
        active_window_title=environment.active_window_title,
        active_profile_id=environment.active_profile_id,
        mode=mode.value if mode is not None else None,
        active_scope=active_scope,
        pointer_mode=pointer_mode,
        selection_label=resolved_selection,
        message=overlay_state.message,
        transcript=overlay_state.transcript,
        target_label=overlay_state.target_label,
        pending_confirmation=overlay_state.pending_confirmation,
        tabs=_workspace_tabs(surface),
        panes=_workspace_panes(surface, pointer_mode=pointer_mode),
        landmarks=landmarks,
        panels=_surface_panels(surface, pointer_mode=pointer_mode),
    )


def infer_shell_surface(
    environment: EnvironmentSnapshot,
    *,
    override: str | None = None,
) -> ShellSurface:
    normalized_override = _normalize_surface_override(override)
    if normalized_override is not None:
        return normalized_override

    active_profile = _normalize(environment.active_profile_id)
    if active_profile in {"chrome_chromium_family", "safari_browser_family"}:
        return ShellSurface.BROWSER

    active_app = _normalize(environment.active_app)
    active_bundle = _normalize(environment.active_bundle_id)
    if active_app in {"safari", "google chrome", "chromium", "brave browser", "arc"}:
        return ShellSurface.BROWSER
    if active_bundle in {
        "com.apple.safari",
        "com.google.chrome",
        "org.chromium.chromium",
        "com.brave.browser",
        "company.thebrowser.browser",
    }:
        return ShellSurface.BROWSER
    if active_app == "notes" or active_bundle == "com.apple.notes":
        return ShellSurface.NOTES
    return ShellSurface.WORKSPACE


def _surface_title(surface: ShellSurface) -> str:
    if surface == ShellSurface.BROWSER:
        return "Browser"
    if surface == ShellSurface.NOTES:
        return "Notes"
    return "Workspace"


def _workspace_tabs(surface: ShellSurface) -> tuple[ShellWorkspaceTab, ...]:
    return (
        ShellWorkspaceTab(
            surface=ShellSurface.WORKSPACE.value,
            title="Workspace",
            utterance="workspace",
            active=surface == ShellSurface.WORKSPACE,
        ),
        ShellWorkspaceTab(
            surface=ShellSurface.BROWSER.value,
            title="Browser",
            utterance="browser",
            active=surface == ShellSurface.BROWSER,
        ),
        ShellWorkspaceTab(
            surface=ShellSurface.NOTES.value,
            title="Notes",
            utterance="notes",
            active=surface == ShellSurface.NOTES,
        ),
        ShellWorkspaceTab(
            surface="precision",
            title="Precision",
            utterance="precision mode",
            active=False,
        ),
    )


def _workspace_panes(
    surface: ShellSurface,
    *,
    pointer_mode: str,
) -> tuple[ShellWorkspacePane, ...]:
    if surface == ShellSurface.BROWSER:
        return (
            ShellWorkspacePane(
                title="Browser Pane",
                role="primary",
                summary="Large browser landmarks replace pixel hunting for tabs, search, and document flow.",
                active=True,
                commands=(
                    ShellCommandHint("tab next", "Move to the next browser tab"),
                    ShellCommandHint("tab back", "Move to the previous browser tab"),
                    ShellCommandHint("window next", "Cycle browser windows"),
                ),
            ),
            ShellWorkspacePane(
                title="Command Rail",
                role="secondary",
                summary="Bounded high-frequency verbs stay short and reversible.",
                commands=(
                    ShellCommandHint("scroll down", "Scroll the current page"),
                    ShellCommandHint("press escape", "Dismiss transient UI"),
                    ShellCommandHint("open Notes", "Jump into capture"),
                ),
            ),
            ShellWorkspacePane(
                title="Precision Rail",
                role="precision",
                summary=_precision_summary(pointer_mode),
                active=pointer_mode == "precision",
                commands=(
                    ShellCommandHint("precision mode", "Prioritize cursor anchors", hot_path=False),
                    ShellCommandHint("semantic mode", "Return to landmark targets", hot_path=False),
                    ShellCommandHint("caret line end", "Jump to the end of the current line", hot_path=False),
                ),
            ),
        )
    if surface == ShellSurface.NOTES:
        return (
            ShellWorkspacePane(
                title="Editor Pane",
                role="primary",
                summary="Structured note capture with semantic anchors before exact caret placement.",
                active=True,
                commands=(
                    ShellCommandHint("type hello there", "Insert dictated text", hot_path=False),
                    ShellCommandHint("type summary period", "Append a short summary", hot_path=False),
                    ShellCommandHint("open Safari", "Return to browser context"),
                ),
            ),
            ShellWorkspacePane(
                title="Capture Rail",
                role="secondary",
                summary="Text editing, revision, and translation stay available from the same shell.",
                commands=(
                    ShellCommandHint("window next", "Cycle note windows"),
                    ShellCommandHint("app back", "Return to the previous app"),
                ),
            ),
            ShellWorkspacePane(
                title="Precision Rail",
                role="precision",
                summary=_precision_summary(pointer_mode),
                active=pointer_mode == "precision",
                commands=(
                    ShellCommandHint("precision mode", "Prioritize caret anchors", hot_path=False),
                    ShellCommandHint("semantic mode", "Return to semantic anchors", hot_path=False),
                    ShellCommandHint("caret line start", "Jump to the start of the current line", hot_path=False),
                ),
            ),
        )
    return (
        ShellWorkspacePane(
            title="Browser Lane",
            role="primary",
            summary="Search, tab, and document commands route through browser landmarks instead of raw screen points.",
            active=True,
            commands=(
                ShellCommandHint("browser", "Open the browser workspace"),
                ShellCommandHint("tab next", "Move to the next browser tab"),
                ShellCommandHint("scroll down", "Scroll the current browser pane"),
            ),
        ),
        ShellWorkspacePane(
            title="Notes Lane",
            role="secondary",
            summary="Capture and rewrite without leaving the structured workspace.",
            commands=(
                ShellCommandHint("notes", "Open the notes workspace"),
                ShellCommandHint("type hello there", "Insert dictated text", hot_path=False),
            ),
        ),
        ShellWorkspacePane(
            title="Command Rail",
            role="utility",
            summary="Global navigation stays short and low-latency.",
            commands=(
                ShellCommandHint("app next", "Cycle applications"),
                ShellCommandHint("window next", "Cycle windows"),
                ShellCommandHint("press escape", "Dismiss transient UI"),
            ),
        ),
        ShellWorkspacePane(
            title="Precision Rail",
            role="precision",
            summary=_precision_summary(pointer_mode),
            active=pointer_mode == "precision",
            commands=(
                ShellCommandHint("precision mode", "Enable exact anchors", hot_path=False),
                ShellCommandHint("semantic mode", "Return to landmark navigation", hot_path=False),
                ShellCommandHint("cursor here", "Move to the current cursor anchor", hot_path=False),
            ),
        ),
    )


def _surface_landmarks(
    surface: ShellSurface,
    *,
    pointer_mode: str,
    selection_label: str | None,
) -> tuple[ShellLandmark, ...]:
    if surface == ShellSurface.BROWSER:
        landmarks = (
            ShellLandmark("Tab Strip", "focus tab strip", "Bounded tab targets replace precise tab-bar aiming."),
            ShellLandmark("Search Bar", "focus search bar", "Jump to the primary search/address landmark."),
            ShellLandmark("Primary Document", "focus document", "Scroll, read, and act on the main content area."),
            ShellLandmark("Current Action", "click this", "Use gaze grounding on the strongest visible browser target."),
            ShellLandmark("Notes Capture", "open Notes", "Move the current idea into the notes lane."),
            ShellLandmark("Cursor Anchor", "cursor here", "Fallback to exact targeting for fine placement.", exact=True),
            ShellLandmark("Line End", "caret line end", "Use an exact caret anchor inside an editable field.", exact=True),
        )
    elif surface == ShellSurface.NOTES:
        landmarks = (
            ShellLandmark("Notes List", "focus note list", "Select among recent notes with a countable target set."),
            ShellLandmark("Editor Block", "focus editor block", "Primary writing surface for dictation and edits."),
            ShellLandmark("Summary Block", "type summary period", "A fast semantic target for short capture updates."),
            ShellLandmark("Browser Return", "open Safari", "Jump back to the browser lane."),
            ShellLandmark("Cursor Anchor", "cursor here", "Fallback to exact caret positioning when needed.", exact=True),
            ShellLandmark("Line Start", "caret line start", "Exact anchor for the beginning of the current line.", exact=True),
            ShellLandmark("Line End", "caret line end", "Exact anchor for the end of the current line.", exact=True),
        )
    else:
        landmarks = (
            ShellLandmark("Browser Lane", "browser", "Semantic browser controls with large tab and document targets."),
            ShellLandmark("Notes Lane", "notes", "Semantic note capture before exact caret placement."),
            ShellLandmark("Command Rail", "focus command rail", "Global structural navigation verbs stay visible."),
            ShellLandmark("Precision Rail", "precision mode", "Exact cursor and caret anchors for fine placement.", exact=True),
            ShellLandmark("Cursor Anchor", "cursor here", "Use exact pointing only when semantic targets are not enough.", exact=True),
        )

    desired_selection = selection_label
    if desired_selection is None:
        desired_selection = next(
            (
                landmark.label
                for landmark in landmarks
                if landmark.exact == (pointer_mode == "precision")
            ),
            landmarks[0].label,
        )

    return tuple(
        ShellLandmark(
            label=landmark.label,
            utterance=landmark.utterance,
            detail=landmark.detail,
            exact=landmark.exact,
            active=landmark.label == desired_selection,
        )
        for landmark in landmarks
    )


def _surface_panels(
    surface: ShellSurface,
    *,
    pointer_mode: str,
) -> tuple[ShellPanel, ...]:
    if surface == ShellSurface.BROWSER:
        return (
            ShellPanel(
                title="Hot Path",
                subtitle="Commands that should feel instant on a browser-heavy workflow.",
                commands=(
                    ShellCommandHint("tab 1", "Switch to tab 1"),
                    ShellCommandHint("tab next", "Move to the next tab"),
                    ShellCommandHint("tab back", "Move to the previous tab"),
                    ShellCommandHint("window next", "Cycle browser windows"),
                ),
            ),
            ShellPanel(
                title="Semantic Anchors",
                subtitle="Large landmarks reduce gaze burden before falling back to exact pointing.",
                commands=(
                    ShellCommandHint("focus tab strip", "Select the tab strip landmark", hot_path=False),
                    ShellCommandHint("focus search bar", "Select the search bar landmark", hot_path=False),
                    ShellCommandHint("click this", "Act on the current grounded target", hot_path=False),
                    ShellCommandHint("open Notes", "Switch to notes for capture"),
                ),
            ),
            ShellPanel(
                title="Precision",
                subtitle=_precision_summary(pointer_mode),
                commands=(
                    ShellCommandHint("precision mode", "Enable exact anchors", hot_path=False),
                    ShellCommandHint("semantic mode", "Return to semantic landmarks", hot_path=False),
                    ShellCommandHint("cursor here", "Move to the selected cursor anchor", hot_path=False),
                ),
            ),
        )
    if surface == ShellSurface.NOTES:
        return (
            ShellPanel(
                title="Capture",
                subtitle="Text-first note capture on the existing command surface.",
                commands=(
                    ShellCommandHint("type hello there", "Insert dictated text", hot_path=False),
                    ShellCommandHint("window next", "Cycle note windows"),
                    ShellCommandHint("app back", "Return to the prior app"),
                ),
            ),
            ShellPanel(
                title="Anchors",
                subtitle="Semantic targets stay visible even when you need exact caret work.",
                commands=(
                    ShellCommandHint("focus editor block", "Select the main editor block", hot_path=False),
                    ShellCommandHint("focus note list", "Select the note list landmark", hot_path=False),
                    ShellCommandHint("precision mode", "Enable exact caret anchors", hot_path=False),
                    ShellCommandHint("open Safari", "Switch back to browser"),
                ),
            ),
        )
    return (
        ShellPanel(
            title="Surfaces",
            subtitle="A universal shell keeps the target set countable before voice or gaze does fine-grained work.",
            commands=(
                ShellCommandHint("browser", "Open the browser lane"),
                ShellCommandHint("notes", "Open the notes lane"),
                ShellCommandHint("app next", "Cycle applications"),
                ShellCommandHint("window next", "Cycle windows"),
            ),
        ),
        ShellPanel(
            title="Modes",
            subtitle=_precision_summary(pointer_mode),
            commands=(
                ShellCommandHint("semantic mode", "Prefer large landmarks", hot_path=False),
                ShellCommandHint("precision mode", "Prefer exact cursor anchors", hot_path=False),
                ShellCommandHint("focus command rail", "Select the global command rail", hot_path=False),
                ShellCommandHint("cursor here", "Move to the active cursor anchor", hot_path=False),
            ),
        ),
    )


def _precision_summary(pointer_mode: str) -> str:
    if _normalize(pointer_mode) == "precision":
        return "Precision mode is active. Exact cursor and caret anchors are currently prioritized."
    return "Semantic mode is active. Large landmarks stay primary until exact placement is needed."


def _normalize_surface_override(value: str | None) -> ShellSurface | None:
    normalized = _normalize(value)
    if not normalized:
        return None
    if normalized in {"home", "workspace"}:
        return ShellSurface.WORKSPACE
    if normalized in {"browser", "safari"}:
        return ShellSurface.BROWSER
    if normalized == "notes":
        return ShellSurface.NOTES
    try:
        return ShellSurface(normalized)
    except ValueError:
        return None


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.casefold().split())
