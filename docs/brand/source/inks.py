BASE = dict(ground="#f3ede1", alt="#ebe3d3", raised="#f7f2e8", line="#ddd4c3", ink="#1d1b18", muted="#5f5a52", mesh="#6f6a62",
            n_ground="#0e1116", n_raised="#141820", n_line="#222831", n_ink="#e8ecf0", n_muted="#a9b1bb", n_mesh="#8b949e")
def ink(**k):
    d = dict(BASE); d.update(k); return d
INKS = {
 "Yamabuki": ink(kanji="山吹", gloss="kerria gold, spark yellow", accent="#9c5100", link="#8a4700", n_accent="#ffb43c", n_link="#ffc463",
              n_ground="#100d08", n_raised="#19140c", n_line="#2a2216",
              story="The yellow-orange of a spark at the moment it leaves the fire. Warmer than gold, darker than flame by day."),
 "Sumibi": ink(kanji="炭火", gloss="charcoal ember", accent="#b03e12", link="#9c3510", n_accent="#ff8a4c", n_link="#ff9d66",
              n_ground="#13100e", n_raised="#1b1714", n_line="#2b2420",
              story="The red that lives inside black charcoal. Washi by day; the night ground is the charcoal itself."),
 "Tamahagane": ink(kanji="玉鋼", gloss="tempered steel", accent="#4b3c9e", link="#43358f", n_accent="#b4a8ff", n_link="#c3b9ff",
              n_ground="#0d0f15", n_raised="#141722", n_line="#232838", n_ink="#e6e9f2",
              story="The violet of the temper colours, the hue steel shows just before it turns blue. Night is cold steel."),
 "Tamamushi": ink(kanji="玉虫", gloss="jewel-beetle iridescence", accent="#00715a", link="#006450", seal="#6b3fa0", n_accent="#3ee0b0", n_link="#5ee8be", n_seal="#c49cff",
              story="Spectral by nature: the jewel beetle is green in one light and violet in another, so the stroke is green and the seal is violet."),
 "Tenmoku": ink(kanji="天目", gloss="iron glaze, oil spots", accent="#8a3a12", link="#7a3310", n_accent="#a9d3f5", n_link="#bcdcf7", n_seal="#d98a4e",
              ground="#f2ebe0", n_ground="#120d0a", n_raised="#1b1410", n_line="#2c221b", n_ink="#ede6dc", n_muted="#b3a89b", n_mesh="#9a8f84",
              story="A kiln-changed glaze: rust-brown by day, and at night the silver-blue of the oil spots that the fire left on black iron."),
 "Uzumibi": ink(kanji="埋み火", gloss="buried embers", accent="#a3281a", link="#931f14", ground="#ecebe8", alt="#e3e1dd", raised="#f4f3f1", line="#d6d3cd", ink="#1c1c1c", muted="#5c5b58", mesh="#6c6b68",
              n_ground="#111111", n_raised="#191919", n_line="#2a2a2a", n_ink="#ecebe8", n_muted="#adaba6", n_mesh="#8e8c88", n_accent="#ff6a48", n_link="#ff8266",
              story="Almost no colour at all: cool ash grey, white and black, and one small red kept alive under it."),
 "Neon Aoi-hi": ink(kanji="青い火", gloss="blue flame", accent="#0b58b3", link="#0a4e9f", n_accent="#38d6ff", n_link="#6fe2ff",
              n_ground="#070b14", n_raised="#0d1322", n_line="#1b2438",
              story="The hottest part of a flame is blue. The neon of the set: an electric cyan glow on a deep night."),
}
