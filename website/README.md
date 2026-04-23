# Evo-RL Project Website

This folder contains a static project webpage for Evo-RL.

## Files

- `index.html`: page structure and content
- `styles.css`: visual system and responsive layout
- `app.js`: chart rendering, reveal animation, interaction, language/theme switching
- `assets/images`: shared static images used by website and root README
- `assets/gifs`: shared GIF gallery used by website and root README
- `assets/videos`: optional local mp4 placeholders (currently not used by the visual gallery)

## Local Preview

From repo root:

```bash
cd website
python -m http.server 8000
```

Open `http://localhost:8000`.

## UI Controls

- Language switch: `EN` / `中` in the top-right corner
- Theme switch: `Day` / `Night` in the top-right corner
- Preferences persist via `localStorage` (`evorl_lang`, `evorl_theme`)

## Replace Visual Media

The website and root README now share media under `assets/`:

- `assets/images/*`
- `assets/gifs/*`

## Replace Curve Data

Edit `metrics` in `app.js`:

- `value.points`
- `advantage.points`
- `success.points`

Each array corresponds to rounds `R1 ... R8` in the chart.
