# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React standalone frontend in `frontend/`, locally hosted. Confirmed by the user on 2026-09-24.

## Users

Project presenter and thesis-defense audience. The presenter explores the system while a separate slide deck carries the research narrative.

## Product Purpose

Show MultiAgent-SafeJudge's complete data flow, each node's function and input/output boundaries, and independently demonstrate model configuration.

## Operating Context

Local defense demonstration with no live model inference. Current scope is the flow and nodes. Reserve navigation and a route for later case studies, but do not build case data views yet.

## Capabilities and Constraints

Clickable workflow nodes with camera movement and focus transitions; contextual node details; model configuration independent of case records; scroll-linked effects; staged exploration instead of fitting every detail into one viewport. Architecture truth comes from README.md and source/config files. Do not turn model configuration into fabricated experimental outputs.

## Brand Commitments

Restrained, serious academic interface for a thesis defense. Primarily flat website design with mature reusable card and button components; no theatrical 3D scenes, metallic artwork, large AI-generated visuals, slogans, promotional hero headings, or stereotypical red/blue/green AI palettes. Camera focus means modest 2D canvas pan/zoom. Scroll effects should preserve academic clarity. The user explicitly requests reference research, generated preview images, then implementation, using Impeccable. Generated previews are design references only and must not become page backgrounds.

## Evidence on Hand

README.md, src/safejudge/, config/models.toml, config/juries/. Existing backend changes belong to ongoing user work and must be preserved.

## Confirmed Visual Refinement

Remove the left sidebar. Use compact horizontal stage navigation. A very small amount of pure blue is allowed for active indicators, selected connection ports, and keyboard focus only; the overall surface remains neutral. Refine node cards and controls using mature components. References: Linear's March 2026 interface refresh, Vercel Geist, React Flow Base Node, and shadcn/ui. The user accepted the overall direction with two refinements: stronger separation of white nodes from light-gray canvas, and simpler icons. Latest reference: `.impeccable/mocks/academic-final.png`. No decorative node icons; use small single-stroke control icons.

## Product Principles

- Complement the defense slides with explorable structure.
- Make each node's data boundary and responsibility clear.
- Separate model configuration from recorded experiments.
- Use motion for spatial continuity and progressive disclosure.
- Keep real results distinct from illustrative configuration.

## Desktop presentation scope

User requests desktop only; no mobile adaptation. Scroll-linked pure-blue connection tracing is permitted. Focus uses subtle background blur while preserving the current node; exit restores the exact previous viewport smoothly.
