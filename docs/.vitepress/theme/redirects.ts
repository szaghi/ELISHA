// docs/.vitepress/theme/redirects.ts: moved pages of the docs, old site path -> new site path (relative to the site
// base, without .html), the scheme of ADAM's and SETH's docs themes. GitHub Pages has no server redirects, so an old
// URL renders the 404 page and index.ts replaces it, client side, by the new location. Keep the entries: external
// links and bookmarks still point at the old locations.

export const pageRedirects: Record<string, string> = {
  // 2026-11-21: the documentation refactored on SETH's plan; the in-depth guide pages became the Reference, and the
  // Examples page the Tutorial
  '/guide/solver': '/reference/solver',
  '/guide/errors-comm-mesh': '/reference/mesh',
  '/guide/spectral-solver': '/reference/spectral',
  '/guide/hierarchical-solver': '/reference/hierarchical',
  '/guide/device': '/reference/device',
  '/guide/l0-math': '/reference/math',
  '/guide/examples': '/tutorial/',
}
