---
name: SafeJudge
description: Restrained academic workflow explorer
colors:
  foreground: "#25282d"
  surface: "#fff"
  canvas: "#f2f3f5"
  focus-blue: "#165dff"
  muted: "#646a75"
  border: "#dadde2"
  node-border: "#cfd4dc"
typography:
  headline:
    fontFamily: '"Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif'
    fontSize: "27px"
    fontWeight: 650
    lineHeight: 1.4
    letterSpacing: "-0.7px"
  title:
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontSize: "14px"
    lineHeight: 1.7
  label:
    fontSize: "12px"
rounded:
  sm: "6px"
  md: "8px"
  lg: "12px"
spacing:
  compact: "8px"
  control-gap: "10px"
  node-inset: "24px"
  panel-inset: "32px"
components:
  workflow-node:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    width: "280px"
  canvas:
    backgroundColor: "{colors.canvas}"
  node-body:
    padding: "24px 24px 21px"
    rounded: "{rounded.lg}"
  canvas-controls:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "2px"
---

# Design System: SafeJudge

## Overview

A flat academic interface: white, text-first nodes sit on a cool gray canvas. Compact navigation, readable labels, and restrained two-dimensional movement support close inspection. The sidebar is absent; there is no promotional hero or decorative node artwork.

This is a scan of the built interface in `frontend/src/styles.css` and `frontend/src/App.tsx`, aligned with the visual contract in `frontend/index.html` and confirmed commitments in `PRODUCT.md`. Review evidence lives in `.impeccable/review/desktop.png`, `detail.png`, `mobile.png`, and `user-935.png`.

## Colors

The palette is neutral, with a sparse focus-blue accent. Foreground also supplies the component library's primary color; blue does not fill general action buttons. Blue marks the active stage underline, selected node ports, and keyboard focus. Main navigation uses a dark underline. Surface, canvas, and fine neutral borders separate content without colored category fills.

## Typography

The system sans-serif stack supports Chinese and Latin content without external font loading. Page headlines lead the hierarchy; node titles are moderately weighted, followed by model metadata and compact explanatory copy. Labels, stage numbers, and canvas percentages remain quiet; numbers use tabular figures where alignment matters. Detail panels use a larger title (24px) and relaxed explanatory leading (1.85). Long model identifiers and source paths wrap rather than widening containers.

## Layout

The desktop header is 72px high with 4% horizontal gutters. A compact title row and horizontal stage navigation precede the flexible canvas. Nodes have a fixed base width; ordinary stage fitting stops at 0.85 zoom so the presenter deliberately pans to inspect wide stages. The full-flow overview may fit down to 0.12 for orientation. Manual zoom ranges from 0.12 to 1.5; opening details moves the camera toward the selected node.

The interface targets desktop presentation only. The graph remains the primary surface, the detail sheet is 440px wide, and configuration content is capped at 1260px. Earlier mobile layouts were removed at the user's request.

## Elevation & Depth

Surfaces are flat at rest. White nodes and visible fine borders establish separation from the gray canvas. A faint hover shadow accompanies a darker node border. The detail sheet uses a very light overlay without backdrop blur; inherited component shadows remain subordinate. Never add ambient glows, glass layers, or decorative depth.

## Shapes

Nodes use softly rounded large corners; compact controls use medium corners. Fine one-pixel borders and thin gray directional connections provide structure. Small neutral pill labels identify node kinds. Circular connection ports are tiny and functional. Control icons are small single-stroke symbols, with no decorative icons in node headings.

## Components

- **Navigation:** horizontal site links and stage buttons, with explicit current-state indicators. Stage navigation can scroll horizontally at narrow widths.
- **Buttons and cards:** reuse the real shadcn/ui Button and Card components already in the project. Outline and ghost actions remain neutral; configuration cards remove their resting shadow.
- **Workflow nodes:** custom React Flow nodes with a text button, optional model line, summary, and optional divided detail action. Hover strengthens the border; selection strengthens the border and colors the ports. Nodes are inspectable, not draggable or connectable.
- **Details:** reuse shadcn/ui Sheet and Tabs. Functional, input/output, and model/source views use text, separators, and a quiet inset formula block. The sheet opens with a short horizontal translation and fade.
- **Configuration fields:** native selects retain labels, neutral borders, and compact sizing. They update the local demonstration only; interface copy states that boundary.
- **Accessibility and motion:** preserve visible blue keyboard focus and semantic button/link labels. Reduced-motion preferences suppress camera animation and shorten CSS transitions. Normal camera movement lasts roughly 450–550ms; most visual state transitions last 200ms.

## Do's and Don'ts

- **Do** preserve white-node/gray-canvas contrast, readable stage zoom, and intentional panning.
- **Do** retain the small-screen node list and the same accessible detail interactions.
- **Do** keep model configuration visually separate from experimental results.
- **Don't** restore a left sidebar, decorate nodes with icons, or assign vivid category palettes.
- **Don't** add theatrical 3D, metallic art, slogans, or oversized promotional typography.
- **Don't** ship generated previews as page rasters or backgrounds. They are design references; the interface is live HTML, CSS, and graph components.

## Desktop motion refinement

The desktop canvas is the sole presentation surface; the mobile node-list alternative was removed at the user's request. Scroll progress reveals a 2.6px pure-blue segment along actual SVG connections over the gray baseline. Labels render above the blue path. Completing the scroll advances the stage after a short pause. Focus raises the node to 1.18–1.35 zoom while retaining clear text; unrelated nodes receive 2px blur and 0.6 opacity, surrounding navigation 1.6px blur and 0.7 opacity. This selective treatment keeps the active node sharp. The transparent dialog backdrop blocks accidental background interaction.

Entering focus saves the exact viewport. Closing restores that saved viewport over 420ms instead of fitting the entire stage again, while the panel exits in 240ms and background clarity returns in 360ms. Content remains mounted through the exit. Timers are canceled on reentry, stage changes, and unmount. Reduced-motion mode disables line tracing and camera travel, retaining clear static focus state.

## Whole-stage explanation progress
One wheel gesture reveals all connections in the current stage together over 700ms and retains the completed blue paths. Another fresh gesture advances to the next stage; reverse retracts the whole stage. Focus preserves progress. Inertia cannot queue transitions. Reduced motion updates the same state immediately. Blue indicates explanation progress, never live execution.

Overview uses pointer-centered wheel zoom and double-click zoom, plus drag panning and fit controls. Stage mode reserves scrolling for explanation progress.

## Separate navigation and motion
Wheel input exclusively zooms around the pointer in every canvas view. Stage labels and arrows handle navigation. Entering a stage automatically reveals all its connections after camera settling, retaining blue paths; focus and zoom never restart the reveal. Re-entering a stage replays it.
