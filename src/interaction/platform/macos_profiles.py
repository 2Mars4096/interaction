"""Typed macOS shortcut profiles for structured navigation commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from interaction.contracts import (
    EnvironmentSnapshot,
    NavigationCommand,
    NavigationScope,
    StructuredNavigationRequest,
)

KEY_CODE_TAB = 48
KEY_CODE_GRAVE = 50
KEY_CODE_LEFT = 123
KEY_CODE_RIGHT = 124
KEY_CODE_DOWN = 125
KEY_CODE_UP = 126


class ShortcutModifier(StrEnum):
    COMMAND = "command"
    SHIFT = "shift"
    OPTION = "option"
    CONTROL = "control"


@dataclass(frozen=True)
class ShortcutPress:
    """A single macOS shortcut press."""

    modifiers: tuple[ShortcutModifier, ...] = ()
    key_code: int | None = None
    keystroke: str | None = None

    def __post_init__(self) -> None:
        has_key_code = self.key_code is not None
        has_keystroke = self.keystroke is not None
        if has_key_code == has_keystroke:
            raise ValueError("shortcut presses require exactly one of key_code or keystroke")


@dataclass(frozen=True)
class NavigationBinding:
    """Map a structured navigation command to one or more concrete shortcut presses."""

    scope: NavigationScope
    command: NavigationCommand
    presses: tuple[ShortcutPress, ...] = ()
    supported_indexes: tuple[int, ...] = ()
    index_modifiers: tuple[ShortcutModifier, ...] = ()

    def __post_init__(self) -> None:
        is_index_binding = self.command == NavigationCommand.INDEX
        if is_index_binding:
            if self.presses:
                raise ValueError("index bindings must resolve from supported index metadata")
            if not self.supported_indexes:
                raise ValueError("index bindings require supported_indexes")
            return
        if not self.presses:
            raise ValueError("non-index bindings require one or more shortcut presses")
        if self.supported_indexes:
            raise ValueError("only index bindings may define supported_indexes")

    def resolve(self, navigation: StructuredNavigationRequest) -> tuple[ShortcutPress, ...] | None:
        if navigation.scope != self.scope or navigation.command != self.command:
            return None
        if self.command == NavigationCommand.INDEX:
            if navigation.index not in self.supported_indexes:
                return None
            return (
                ShortcutPress(
                    modifiers=self.index_modifiers,
                    keystroke=str(navigation.index),
                ),
            )
        return self.presses


@dataclass(frozen=True)
class AppMatchRule:
    """Deterministic app-family matching by bundle id or visible app name."""

    bundle_ids: tuple[str, ...] = ()
    app_names: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def matches(self, environment: EnvironmentSnapshot) -> bool:
        active_bundle_id = _normalize(environment.active_bundle_id)
        if active_bundle_id and active_bundle_id in {_normalize(value) for value in self.bundle_ids}:
            return True
        active_app = _normalize(environment.active_app)
        if not active_app:
            return False
        names = {_normalize(value) for value in (*self.app_names, *self.aliases)}
        return active_app in names


@dataclass(frozen=True)
class MacOSAppProfile:
    """Shortcut profile for a global or app-family specific structured navigation surface."""

    profile_id: str
    display_name: str
    bindings: tuple[NavigationBinding, ...]
    match: AppMatchRule | None = None

    def matches(self, environment: EnvironmentSnapshot) -> bool:
        if self.match is None:
            return True
        return self.match.matches(environment)

    def resolve(self, navigation: StructuredNavigationRequest) -> tuple[ShortcutPress, ...] | None:
        for binding in self.bindings:
            shortcut_presses = binding.resolve(navigation)
            if shortcut_presses is not None:
                return shortcut_presses
        return None

    def supported_scopes(self) -> tuple[str, ...]:
        scopes = {binding.scope.value for binding in self.bindings}
        return tuple(sorted(scopes))


@dataclass(frozen=True)
class NavigationResolution:
    """Result of resolving a structured navigation command through app profiles."""

    navigation: StructuredNavigationRequest
    profile: MacOSAppProfile | None
    shortcut_presses: tuple[ShortcutPress, ...] = ()
    message: str | None = None
    fallback_profile_id: str | None = None

    @property
    def supported(self) -> bool:
        return self.message is None and bool(self.shortcut_presses)


class MacOSProfileRegistry:
    """Resolve structured navigation commands through deterministic app profiles."""

    def __init__(
        self,
        *,
        global_profile: MacOSAppProfile,
        app_profiles: tuple[MacOSAppProfile, ...],
    ) -> None:
        self.global_profile = global_profile
        self.app_profiles = app_profiles
        self._profiles_by_id = {
            profile.profile_id: profile for profile in (global_profile, *app_profiles)
        }

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(profile.profile_id for profile in (self.global_profile, *self.app_profiles))

    def supported_scopes(self) -> tuple[str, ...]:
        scopes: set[str] = set()
        for profile in (self.global_profile, *self.app_profiles):
            scopes.update(profile.supported_scopes())
        return tuple(sorted(scopes))

    def resolve_profile(
        self,
        environment: EnvironmentSnapshot,
        *,
        requested_profile_id: str | None = None,
    ) -> tuple[MacOSAppProfile | None, str | None]:
        profile_id = requested_profile_id or environment.active_profile_id
        if profile_id:
            profile = self._profiles_by_id.get(profile_id)
            if profile is None:
                return None, f"Unknown macOS app profile '{profile_id}'."
            if profile is self.global_profile or profile.matches(environment):
                return profile, None
            return (
                None,
                f"Requested macOS profile '{profile_id}' does not match the active app {self._app_label(environment)}.",
            )

        for profile in self.app_profiles:
            if profile.matches(environment):
                return profile, None
        return self.global_profile, None

    def resolve_navigation(
        self,
        environment: EnvironmentSnapshot,
        navigation: StructuredNavigationRequest,
    ) -> NavigationResolution:
        if navigation.count != 1:
            return NavigationResolution(
                navigation=navigation,
                profile=None,
                message=(
                    f"Structured navigation only supports count=1 right now; received count={navigation.count}."
                ),
            )

        profile, profile_error = self.resolve_profile(
            environment,
            requested_profile_id=navigation.app_profile,
        )
        if profile_error is not None:
            return NavigationResolution(
                navigation=navigation,
                profile=None,
                message=profile_error,
            )
        assert profile is not None

        shortcut_presses = profile.resolve(navigation)
        if shortcut_presses is not None:
            return NavigationResolution(
                navigation=navigation,
                profile=profile,
                shortcut_presses=shortcut_presses,
            )

        if profile.profile_id != self.global_profile.profile_id:
            global_shortcuts = self.global_profile.resolve(navigation)
            if global_shortcuts is not None:
                return NavigationResolution(
                    navigation=navigation,
                    profile=self.global_profile,
                    shortcut_presses=global_shortcuts,
                    fallback_profile_id=profile.profile_id,
                )

        return NavigationResolution(
            navigation=navigation,
            profile=profile,
            message=(
                f"Structured navigation '{_navigation_label(navigation)}' is not supported for "
                f"{profile.display_name} ({profile.profile_id})."
            ),
        )

    @staticmethod
    def _app_label(environment: EnvironmentSnapshot) -> str:
        if environment.active_bundle_id and environment.active_app:
            return f"{environment.active_app} ({environment.active_bundle_id})"
        if environment.active_app:
            return environment.active_app
        if environment.active_bundle_id:
            return environment.active_bundle_id
        return "the current app"


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.casefold().split())


def _navigation_label(navigation: StructuredNavigationRequest) -> str:
    if navigation.command == NavigationCommand.INDEX:
        return f"{navigation.scope.value} {navigation.index}"
    return f"{navigation.scope.value} {navigation.command.value}"


GLOBAL_MACOS_PROFILE = MacOSAppProfile(
    profile_id="global_macos",
    display_name="Global macOS",
    bindings=(
        NavigationBinding(
            scope=NavigationScope.APP,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.APP,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.WINDOW,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_GRAVE,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.WINDOW,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_GRAVE,
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.SPACE,
            command=NavigationCommand.LEFT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_LEFT,
                    modifiers=(ShortcutModifier.CONTROL,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.SPACE,
            command=NavigationCommand.RIGHT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_RIGHT,
                    modifiers=(ShortcutModifier.CONTROL,),
                ),
            ),
        ),
    ),
)

CHROME_CHROMIUM_PROFILE = MacOSAppProfile(
    profile_id="chrome_chromium_family",
    display_name="Chrome / Chromium",
    match=AppMatchRule(
        bundle_ids=(
            "com.google.Chrome",
            "org.chromium.Chromium",
            "com.brave.Browser",
            "company.thebrowser.Browser",
        ),
        app_names=("Google Chrome", "Chromium", "Brave Browser", "Arc"),
        aliases=("Chrome", "Brave", "Arc"),
    ),
    bindings=(
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.CLOSE,
            presses=(
                ShortcutPress(
                    keystroke="w",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.REOPEN,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.INDEX,
            supported_indexes=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            index_modifiers=(ShortcutModifier.COMMAND,),
        ),
        NavigationBinding(
            scope=NavigationScope.WINDOW,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="n",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
    ),
)

SAFARI_PROFILE = MacOSAppProfile(
    profile_id="safari_browser_family",
    display_name="Safari",
    match=AppMatchRule(
        bundle_ids=("com.apple.Safari",),
        app_names=("Safari",),
    ),
    bindings=(
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    keystroke="]",
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    keystroke="[",
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.CLOSE,
            presses=(
                ShortcutPress(
                    keystroke="w",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.REOPEN,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.INDEX,
            supported_indexes=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            index_modifiers=(ShortcutModifier.COMMAND,),
        ),
        NavigationBinding(
            scope=NavigationScope.WINDOW,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="n",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
    ),
)

NOTES_PROFILE = MacOSAppProfile(
    profile_id="notes_family",
    display_name="Notes",
    match=AppMatchRule(
        bundle_ids=("com.apple.Notes",),
        app_names=("Notes",),
    ),
    bindings=(),
)

FINDER_PROFILE = MacOSAppProfile(
    profile_id="finder_family",
    display_name="Finder",
    match=AppMatchRule(
        bundle_ids=("com.apple.finder",),
        app_names=("Finder",),
    ),
    bindings=(
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.WINDOW,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="n",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.CLOSE,
            presses=(
                ShortcutPress(
                    keystroke="w",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
    ),
)

MAIL_PROFILE = MacOSAppProfile(
    profile_id="mail_family",
    display_name="Mail",
    match=AppMatchRule(
        bundle_ids=("com.apple.mail",),
        app_names=("Mail",),
        aliases=("Apple Mail",),
    ),
    bindings=(
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.CLOSE,
            presses=(
                ShortcutPress(
                    keystroke="w",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.WINDOW,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_GRAVE,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
    ),
)

TERMINAL_FAMILY_PROFILE = MacOSAppProfile(
    profile_id="terminal_family",
    display_name="Terminal / iTerm",
    match=AppMatchRule(
        bundle_ids=(
            "com.apple.Terminal",
            "com.googlecode.iterm2",
            "dev.warp.Warp-Stable",
        ),
        app_names=("Terminal", "iTerm2", "Warp"),
        aliases=("iTerm",),
    ),
    bindings=(
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEW,
            presses=(
                ShortcutPress(
                    keystroke="t",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.CLOSE,
            presses=(
                ShortcutPress(
                    keystroke="w",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.LEFT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_LEFT,
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.OPTION),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.RIGHT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_RIGHT,
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.OPTION),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.UP,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_UP,
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.OPTION),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.DOWN,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_DOWN,
                    modifiers=(ShortcutModifier.COMMAND, ShortcutModifier.OPTION),
                ),
            ),
        ),
    ),
)

EDITOR_FAMILY_PROFILE = MacOSAppProfile(
    profile_id="editor_family",
    display_name="VS Code / Cursor / Zed",
    match=AppMatchRule(
        bundle_ids=(
            "com.microsoft.VSCode",
            "com.todesktop.230313mzl4w4u92",
            "dev.zed.Zed",
        ),
        app_names=("Visual Studio Code", "Cursor", "Zed"),
        aliases=("VS Code", "Code"),
    ),
    bindings=(
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.NEXT,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.TAB,
            command=NavigationCommand.BACK,
            presses=(
                ShortcutPress(
                    key_code=KEY_CODE_TAB,
                    modifiers=(ShortcutModifier.CONTROL, ShortcutModifier.SHIFT),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.LEFT,
            presses=(
                ShortcutPress(
                    keystroke="k",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
                ShortcutPress(
                    key_code=KEY_CODE_LEFT,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.RIGHT,
            presses=(
                ShortcutPress(
                    keystroke="k",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
                ShortcutPress(
                    key_code=KEY_CODE_RIGHT,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.UP,
            presses=(
                ShortcutPress(
                    keystroke="k",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
                ShortcutPress(
                    key_code=KEY_CODE_UP,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
        NavigationBinding(
            scope=NavigationScope.PANE,
            command=NavigationCommand.DOWN,
            presses=(
                ShortcutPress(
                    keystroke="k",
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
                ShortcutPress(
                    key_code=KEY_CODE_DOWN,
                    modifiers=(ShortcutModifier.COMMAND,),
                ),
            ),
        ),
    ),
)

DEFAULT_MACOS_PROFILE_REGISTRY = MacOSProfileRegistry(
    global_profile=GLOBAL_MACOS_PROFILE,
    app_profiles=(
        CHROME_CHROMIUM_PROFILE,
        SAFARI_PROFILE,
        NOTES_PROFILE,
        FINDER_PROFILE,
        MAIL_PROFILE,
        TERMINAL_FAMILY_PROFILE,
        EDITOR_FAMILY_PROFILE,
    ),
)
