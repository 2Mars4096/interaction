# Product Spec

## Product Summary

Interaction is a macOS-first multimodal control system for general computer users. It combines realtime voice input with webcam-based gaze tracking so users can issue commands like "open this", "scroll down", or "translate that" with less keyboard and mouse effort.

## Product Decisions

- **User:** general computer user
- **Platform:** macOS first
- **Expansion path:** cross-platform later
- **Tracking hardware:** standard webcam only
- **First demo:** basic controls that can start replacing parts of keyboard and mouse use

## Problem

Traditional computer interaction makes targeting and action expression separate burdens:

- the mouse is good at targeting but requires continuous manual motion
- the keyboard is good at symbolic control but requires explicit command syntax and hand movement
- voice alone lacks precise grounding
- gaze alone lacks expressive intent

The product hypothesis is that voice plus gaze can reduce friction when the system reliably fuses "what" with "where".

## Design Goals

- Make common computer actions feel more direct.
- Minimize setup friction by using only a webcam.
- Keep the system explainable and interruptible.
- Demonstrate real utility for ordinary desktop tasks.
- Build a control architecture that can grow into accessibility and productivity use cases.

## Non-Goals For V1

- Full desktop automation
- Agentic task planning across many applications
- Precision-grade cursor replacement for tiny targets
- Sensitive or destructive actions without strong confirmation
- Hardware-specific integrations beyond a normal webcam

## Primary Use Cases

- Open or click the thing currently being looked at.
- Scroll without using the mouse wheel or trackpad.
- Switch to a named application.
- Press simple navigation keys by voice.
- Enter short dictated text into the current field.
- Translate selected or dictated text.

## MVP Command Families

- **Targeting:** highlight, focus, click
- **Navigation:** scroll, back, escape, tab, enter
- **App control:** open app, switch app
- **Text control:** type short text, translate text
- **Recovery:** cancel, undo last action when possible, ask for clarification

## UX Requirements

- The system must visibly show what it heard.
- The system must visibly show what target it believes the user means.
- The system must differentiate between auto-executed actions and confirmation-required actions.
- The user must be able to interrupt or cancel quickly.

## MVP Release Criteria

- A new user can calibrate and perform the showcase tasks on macOS with only a webcam and microphone.
- The system can complete a small set of basic control tasks with acceptable latency and low accidental-action rate.
- The broker policy is explicit enough that risky actions are not silently executed.
- The gaze subsystem is trustworthy on large targets before the product attempts fine-grained precision.
