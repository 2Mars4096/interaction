# macOS test app

Open **[Interaction.app](../dist/Interaction.app)** from Finder. The app is built locally in `dist/`; no Terminal is needed to use it.

## First test

1. Choose **Dictation**.
2. Check **Control my Mac** to insert text. Leave it unchecked to preview recognition and actions without typing or moving the pointer.
3. Click **Start**, then click the **Test pad** during the four-second countdown.
4. Say a short sentence. Try “comma” and “new line”. Edit corrections manually.
5. Click **Stop**, or press **Command–Shift–Escape** from any app.

The first run may request Microphone and Speech Recognition permission. Live typing or pointer control also needs Accessibility permission; macOS may request Automation access. **Privacy & access…** explains permissions and speech/log storage, and opens Privacy Settings. Enable the Interaction, Python, or helper entry macOS identifies, then restart the session.

## Modes

| Mode | What to try |
| --- | --- |
| Voice commands | “open Safari”, “switch to Notes”, “scroll down”, “press escape”, “type hello” followed by “yes” or “cancel” |
| Dictation | Speak into the test pad or another focused text field. Automatic deletion/replacement is blocked; edit corrections manually. |
| Gaze cursor | Click **Calibrate gaze…** first, then start and look at large targets. Preview shows the gaze dot; live mode also moves the pointer. |
| Gaze + voice | Calibrate first, look at a target, say “click this”, and keep looking there while confirming. One command per Start. |

Calibration opens dot overlays. Look at each dot and press Space, Enter, or click the overlay to capture it. The app refuses interactive calibration if it cannot display the dots.

**Start** runs only one session at a time. Stop, closing the window, and quitting cancel the worker and its microphone/camera helpers. If another app already owns the global shortcut, the footer shows the fallback: activate Interaction and press Command–Period, or click Stop.

Voice modes use repeated capture windows; short gaps between windows remain. “Stop listening” ends a voice session at the capture boundary. The global shortcut stops it without waiting for that boundary.

## Logs and data

**Activity** shows live transcripts, prompts, and errors. Cursor mode reports tracking changes and execution failures immediately, stops on failed pointer execution, and saves diagnostics as it runs. Stop preserves those records and adds a cancellation entry. Preview movement is counted separately from real movement. **Show logs** opens `.interaction/logs` in this checkout. Calibration and settings reuse the existing `.interaction` folder. The test pad is temporary; copy any text you want to keep before closing the app.

The app's unchecked execution toggle explicitly overrides saved live-control settings. Preview does still use the microphone or camera when you press Start; it suppresses desktop input.

## Build again after code changes

```bash
python scripts/build_macos_app.py
open dist/Interaction.app
```

The bundle contains a snapshot of the Python engine and a compiled native AppKit launcher. It uses the Python environment selected at build time (`--python /path/to/python`), which must retain NumPy, OpenCV, Pydantic, and Tk. Apple's Command Line Tools are required to build the native app and speech helper. `--runtime-dir /path/to/data` selects another state/log location.

This is an ad-hoc-signed local test build for this Mac, not a portable or notarized release. Moving the bundle on this Mac is fine while its configured Python and runtime paths remain available. Rebuild to include later source changes. Translation is unavailable, and webcam gaze accuracy still needs hands-on testing. Automatic deletion/replacement is disabled in live mode until field/caret ownership can be verified.

## Speech and transcript privacy

Speech uses Apple’s recognition service; on-device-only processing is not enforced. Transcripts are persisted in the configured logs folder, which is inside this Dropbox project by default and may therefore sync. The app’s Privacy & access dialog makes both behaviours explicit. No runtime files were moved by this update.
