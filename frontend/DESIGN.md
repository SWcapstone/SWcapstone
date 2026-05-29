# DESIGN.md — Toss-inspired UI

## Overview

Toss-inspired UI is a bright, calm, and highly legible fintech interface built on a soft white canvas, vivid blue action color, large rounded surfaces, and direct Korean microcopy. The system avoids visual noise and makes every screen feel fast, trustworthy, and effortless.

The design language is not decorative. Brand energy comes from clarity: large numbers, generous spacing, friendly one-line explanations, and one obvious next action. Financial information should feel easy to understand within three seconds.

The default surface is a near-white background (`{colors.canvas}` — #F9FAFB) with white elevated cards (`{colors.surface}` — #FFFFFF). Primary actions use Toss-like vivid blue (`{colors.primary}` — #3182F6), while text stays dark and neutral. Borders and shadows are used only to separate content gently, never to create heavy depth.

Typography uses a Korean-friendly sans-serif stack such as Pretendard, Inter, Apple SD Gothic Neo, and Noto Sans KR. Headlines are bold and simple. Body copy is short, friendly, and plain. The UI should sound like a helpful Korean fintech app, not a bank document.

**Key Characteristics:**
- Soft white / light gray canvas with white rounded cards.
- Vivid blue used only for primary actions, selected states, and key highlights.
- Large, readable numbers are the visual anchor of financial screens.
- Rounded cards and buttons create a friendly, mobile-first feeling.
- Copy is short, direct, and conversational.
- One screen should have one clear purpose and one main CTA.
- UI chrome stays minimal: no heavy gradients, no strong shadows, no dense tables.
- Empty states, success states, and errors should feel calm and helpful.

---

## Colors

### Brand & Accent

- **Primary Blue** (`{colors.primary}` — #3182F6): Main action color. Used for primary CTA buttons, active states, selected items, progress indicators, and important links.
- **Primary Blue Hover** (`{colors.primary-hover}` — #1B64DA): Pressed / hover state for primary actions.
- **Primary Blue Light** (`{colors.primary-light}` — #E8F3FF): Soft background for selected cards, info badges, active list items, and subtle highlights.
- **Accent Blue Soft** (`{colors.accent-blue-soft}` — #F0F6FF): Very light blue surface used for recommendation cards or calm informative sections.

Primary blue must not be overused. A typical screen should have only one dominant blue element: the main CTA or the currently selected item.

### Surface

- **Canvas** (`{colors.canvas}` — #F9FAFB): Default app background.
- **Surface** (`{colors.surface}` — #FFFFFF): Cards, bottom sheets, modals, main content containers.
- **Surface Muted** (`{colors.surface-muted}` — #F2F4F6): Input backgrounds, disabled button backgrounds, grouped list areas.
- **Surface Elevated** (`{colors.surface-elevated}` — #FFFFFF): Floating cards, sticky bottom action areas, modal surfaces.
- **Surface Pressed** (`{colors.surface-pressed}` — #E5E8EB): Pressed state for secondary buttons or list rows.

### Text

- **Text Primary** (`{colors.text-primary}` — #191F28): Main titles, key values, important labels.
- **Text Secondary** (`{colors.text-secondary}` — #4E5968): Body copy, descriptions, secondary labels.
- **Text Tertiary** (`{colors.text-tertiary}` — #8B95A1): Helper text, captions, timestamps, placeholders.
- **Text Disabled** (`{colors.text-disabled}` — #B0B8C1): Disabled text, inactive controls.
- **Text On Primary** (`{colors.text-on-primary}` — #FFFFFF): Text on primary blue buttons.

### Hairlines & Borders

- **Border Light** (`{colors.border-light}` — #E5E8EB): Card outlines, input borders, list separators.
- **Border Medium** (`{colors.border-medium}` — #D1D6DB): Focused secondary controls or stronger dividers.
- **Divider** (`{colors.divider}` — #F2F4F6): Very soft list dividers.

Borders should be subtle. Prefer spacing and background contrast over visible outlines.

### Semantic

- **Success** (`{colors.success}` — #00C853): Completed payments, verified states, successful actions.
- **Success Light** (`{colors.success-light}` — #E6F9EF): Success background.
- **Warning** (`{colors.warning}` — #FFB020): Caution states, pending actions.
- **Warning Light** (`{colors.warning-light}` — #FFF4DB): Warning background.
- **Error** (`{colors.error}` — #F04452): Failed actions, invalid inputs, destructive warnings.
- **Error Light** (`{colors.error-light}` — #FFE8EA): Error background.

Semantic colors should be used sparingly. Red is only for real problems, not for decoration.

---

## Typography

### Font Family

The system uses a Korean-friendly sans-serif typeface.

Recommended stack:

```css
font-family:
  "Pretendard",
  "Inter",
  "Apple SD Gothic Neo",
  "Noto Sans KR",
  system-ui,
  sans-serif;