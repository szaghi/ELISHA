# ELISHA · visual identity

**Mark:** Futatsu-bi 二つ火 (two embers). **Ink:** Kōyō 紅葉 (autumn maples).
Part of the sumi family: brushed mark, faint three-level AMR ground, carved seal (E) lower right.

Two embers in one tea hearth. The rings raked in the ash are contours of a real
finite-difference Poisson solve with two point sources in a Dirichlet square:
small rings around each ember, one ring that takes in both at the waist, and the
outer rings squaring off to the rim. Superposition, drawn.

Colour follows temperature: embers and seal in maple crimson, the inner rings in
persimmon, the waist in ginkgo gold, the cold rings and the hearth in ink.

## In this repository

Applied 2026-11-21: `docs/public/` holds the logos, favicons, manifest and icons; `docs/.vitepress/theme/` the theme
(`custom.css`, `index.ts`); `config.mts` the favicon links, theme colours and navbar logo; `docs/index.md` the hero
image and the feature icons; `README.md` the hero mark. This folder keeps the tokens and the generator, and VitePress
skips it (`srcExclude`). The paths in the table below are relative to `docs/`; the generator is in `source/`.

One fix to the bundle: the two `icons/*/dirichlet-neumann.svg` had an unescaped `&` in their `<title>`, which made them
malformed XML (browsers refused them); it is `&amp;` now. The `<title>` is not written by `source/`, so the generator
needs no change.

## Files

| File | Use |
|---|---|
| `public/logo.svg`, `logo-dark.svg` | Full mark (ground and seal), hero image |
| `public/logo-mark.svg`, `logo-mark-dark.svg` | Mark only, nav bar |
| `public/favicon.svg` | Adaptive favicon (switches with prefers-color-scheme) |
| `public/favicon.ico`, `favicon-16/32/48/96.png` | Fallback favicons |
| `public/apple-touch-icon.png`, `icon-192.png`, `icon-512.png`, `icon-maskable-512.png` | Home-screen and manifest icons |
| `public/logo-1024.png`, `logo-dark-1024.png` | README, slides |
| `public/icons/{light,dark}/*.svg` | Ten feature icons (FFT, AMR multigrid, periodic, Dirichlet/Neumann, isolated, flexible CG, MPI, GPU, round-off, licence) |
| `public/site.webmanifest` | Web app manifest |
| `.vitepress/theme/custom.css`, `index.ts` | Theme override |
| `tokens.json` | Colour tokens with contrast ratios (sumi family format) |
| `source/` | Python generator of the brush strokes (`marks2.mark_A2`) |

## Contrast

| | ground | ember / seal | link | button text |
|---|---|---|---|---|
| Washi | #f4ece0 | #a8201a 6.2:1 | #9a1d17 7.0:1 | white 7.3:1 |
| Night | #140d0a | #ff5e4a 6.4:1 | #ff7a69 7.5:1 | #1a0805 6.4:1 |

Persimmon and ginkgo reach about 4:1 on washi: use them for strokes and
decoration only, never for text.
