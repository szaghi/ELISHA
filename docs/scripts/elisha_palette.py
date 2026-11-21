"""The colours of ELISHA's documentation figures and animations: the Kōyō identity applied to data.

Why this module exists: the figures (make_doc_figures.py) draw on the identity's two grounds, washi (light) and Kōyō
night (dark, docs/brand/tokens.json), and every colour they use lives here once.

Validated with the dataviz skill's validate_palette.js (OKLab x100, Machado 2009 CVD simulation):

* categorical, in this fixed order, never cycled: ember crimson, indigo, ginkgo, wisteria. Adjacent pairs pass every
  check on both grounds (light on #f4ece0: worst CVD dE 21.2, normal-vision 23.9; dark on #140d0a: 21.9 and 24.5).
  All-pairs, crimson and ginkgo collapse under deuteranopia (dE 3.8 light, 1.5 dark): a scatter-like figure, where any
  two marks can meet, carries at most two coloured series (crimson, indigo); line figures add markers as a second
  encoding;
* the identity's own persimmon and ginkgo (#b5560a, #8f6f00 on washi) are about 4:1 on the ground: decoration, never
  text (docs/brand/IDENTITY.md);
* sequential (a potential of one sign): one hue, the ember, washi-light to bark-dark, monotone in lightness;
* diverging (a potential of both signs): indigo and crimson arms, the ground at zero;
* ordinal (the AMR levels): the ink, lighter for coarser levels.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# identity grounds (docs/brand/tokens.json)
WASHI, WASHI_ALT, INK_LIGHT, INK_MUTED_LIGHT, MESH_LIGHT = '#f4ece0', '#ebe2d3', '#1d1b18', '#5f5a52', '#6f6a62'
NIGHT, NIGHT_ALT, INK_DARK, INK_MUTED_DARK, MESH_DARK = '#140d0a', '#0f0a08', '#f0e6da', '#b8a99a', '#9a8b7d'

# categorical, fixed order: crimson, indigo, ginkgo, wisteria
CRIMSON, INDIGO, GINKGO, WISTERIA = '#b02a1e', '#2a63a8', '#8a6a00', '#6a4a9e'
CRIMSON_D, INDIGO_D, GINKGO_D, WISTERIA_D = '#e0503a', '#4a8ad0', '#a8861a', '#9070c8'
LINES = {'light': (CRIMSON, INDIGO, GINKGO, WISTERIA), 'dark': (CRIMSON_D, INDIGO_D, GINKGO_D, WISTERIA_D)}
MARKERS = ('o', 's', '^', 'D')

# the grounds and inks per theme
GROUND = {'light': WASHI, 'dark': NIGHT}
PANEL = {'light': WASHI, 'dark': NIGHT_ALT}
INK = {'light': INK_LIGHT, 'dark': INK_DARK}
MUTED = {'light': INK_MUTED_LIGHT, 'dark': INK_MUTED_DARK}
MESH = {'light': MESH_LIGHT, 'dark': MESH_DARK}

# AMR levels 0, 1, 2 (ordinal): block outlines, coarse light, fine dark (on washi) / coarse dim, fine bright (night)
LEVELS = {'light': ('#b9ad9c', '#6f6a62', '#1d1b18'), 'dark': ('#4a3c32', '#9a8b7d', '#f0e6da')}

# sequential: a potential well is deepest where the embers are, drawn darkest (light) or hottest (dark)
EMBER_LIGHT = LinearSegmentedColormap.from_list(
    'ember_light', ['#f4ece0', '#efcfb4', '#e29a72', '#c85a3a', '#a8201a', '#5e110b'])
EMBER_DARK = LinearSegmentedColormap.from_list(
    'ember_dark', ['#140d0a', '#3a120c', '#7a1d12', '#c43a24', '#ff7a52', '#ffd0a0'])
EMBER = {'light': EMBER_LIGHT, 'dark': EMBER_DARK}

# diverging (a potential of both signs): two hues and the ground at zero; the wells (phi < 0, drawn as -phi > 0) in
# crimson, the hills in indigo
DIVERGING = {
    'light': LinearSegmentedColormap.from_list(
        'koyo_div_light', ['#2a63a8', '#9ab5d6', '#f4ece0', '#e8a383', '#b02a1e', '#5e110b']),
    'dark': LinearSegmentedColormap.from_list(
        'koyo_div_dark', ['#8ab6e8', '#2c4f7a', '#140d0a', '#7a2214', '#e0503a', '#ffc09a']),
}
