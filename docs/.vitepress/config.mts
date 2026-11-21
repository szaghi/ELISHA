import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { withMermaid } from 'vitepress-plugin-mermaid'

// formal's sidebar of the API pages (docs/api/_sidebar.json), when the API has been generated
const apiSidebarFile = resolve(__dirname, '../api/_sidebar.json')
const apiSidebar = existsSync(apiSidebarFile) ? JSON.parse(readFileSync(apiSidebarFile, 'utf-8')) : []

export default withMermaid({
  title: 'ELISHA',
  description: 'Elliptic Laplacian Inversion, Spectral and Hierarchical Algorithms: FFT and AMR multigrid Poisson solvers for HPC',
  base: '/ELISHA/',
  srcExclude: ['ford.md', 'brand/**'],

  // the identity (Futatsu-bi in Koyo, docs/brand/): an adaptive SVG favicon with PNG and ICO fallbacks, the touch icon,
  // the web manifest and the theme colour of each scheme
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/ELISHA/favicon.svg' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/ELISHA/favicon-32.png' }],
    ['link', { rel: 'alternate icon', href: '/ELISHA/favicon.ico' }],
    ['link', { rel: 'apple-touch-icon', href: '/ELISHA/apple-touch-icon.png' }],
    ['link', { rel: 'manifest', href: '/ELISHA/site.webmanifest' }],
    ['meta', { name: 'theme-color', content: '#f4ece0', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#140d0a', media: '(prefers-color-scheme: dark)' }],
    // the theme-aware videos of the tutorial: <video class="light-only"> / <video class="dark-only">
    ['style', {}, '.dark-only{display:none}.dark .dark-only{display:block}.dark .light-only{display:none}'],
  ],

  markdown: {
    math: true,
    languages: ['fortran-free-form', 'fortran-fixed-form'],
    languageAlias: {
      fortran: 'fortran-free-form',
      f90: 'fortran-free-form',
      f03: 'fortran-free-form',
      f08: 'fortran-free-form',
    },
  },

  themeConfig: {
    logo: { light: '/logo-mark.svg', dark: '/logo-mark-dark.svg', alt: 'ELISHA' },
    siteTitle: 'ELISHA',

    nav: [
      { text: 'Start',        link: '/guide/' },
      { text: 'Tutorial',     link: '/tutorial/' },
      { text: 'Cookbook',     link: '/cookbook/' },
      { text: 'Reference',    link: '/reference/' },
      { text: 'Verification', link: '/verification/' },
      { text: 'API',          link: '/api/' },
      { text: 'GitHub',       link: 'https://github.com/szaghi/ELISHA' },
    ],

    // One sidebar for the whole site, in reading order: it is also the chain of the Previous / Next links at the foot
    // of every page, so reading straight through goes start -> tutorial -> recipes -> reference -> verification.
    sidebar: {
      // the module pages written by formal (docs:build runs it first), with its own sidebar
      '/api/': [
        { text: 'Back to the documentation', link: '/guide/' },
        { text: 'API', items: [{ text: 'Overview', link: '/api/' }] },
        ...apiSidebar,
      ],
      '/': [
        { text: 'Start here', items: [
          { text: 'About',              link: '/guide/' },
          { text: 'Installation',       link: '/guide/installation' },
          { text: 'Quick start',        link: '/guide/quickstart' },
          { text: 'Concepts',           link: '/guide/concepts' },
        ] },
        { text: 'Tutorial', collapsed: false, items: [
          { text: 'All chapters',              link: '/tutorial/' },
          { text: '1. A periodic box',         link: '/tutorial/01-box' },
          { text: '2. Faces',                  link: '/tutorial/02-faces' },
          { text: '3. A star in empty space',  link: '/tutorial/03-isolated' },
          { text: '4. Refine where it matters', link: '/tutorial/04-amr' },
          { text: '5. Boundary values',        link: '/tutorial/05-values' },
          { text: '6. Watching the FCG',       link: '/tutorial/06-iterations' },
          { text: '7. Many ranks',             link: '/tutorial/07-mpi' },
          { text: '8. Two embers',             link: '/tutorial/08-embers' },
          { text: '9. On the GPU',             link: '/tutorial/09-gpu' },
        ] },
        { text: 'Cookbook', collapsed: true, items: [
          { text: 'All recipes',               link: '/cookbook/' },
          { text: 'Choose a path',             link: '/cookbook/choose' },
          { text: 'Which faces go together',   link: '/cookbook/faces' },
          { text: 'Your fields in and out',    link: '/cookbook/fields' },
          { text: 'Self-gravity',              link: '/cookbook/gravity' },
          { text: 'One setup, many solves',    link: '/cookbook/reuse' },
          { text: 'When it does not converge', link: '/cookbook/not-converged' },
          { text: 'Host or device arrays',     link: '/cookbook/device-arrays' },
        ] },
        { text: 'Reference', collapsed: true, items: [
          { text: 'Feature map',               link: '/reference/' },
          { text: 'The solver facade',         link: '/reference/solver' },
          { text: 'Mesh and communicator',     link: '/reference/mesh' },
          { text: 'Status codes',              link: '/reference/status' },
          { text: 'Parameters',                link: '/reference/parameters' },
          { text: 'The spectral solver',       link: '/reference/spectral' },
          { text: 'The hierarchical solver',   link: '/reference/hierarchical' },
          { text: 'The device backends',       link: '/reference/device' },
          { text: 'L0: the math',              link: '/reference/math' },
        ] },
        { text: 'Verification', collapsed: true, items: [
          { text: 'Overview: drivers, harness, device', link: '/verification/' },
          { text: 'SV-2 The spectral solver, periodic', link: '/verification/fft-poisson' },
          { text: 'SV-6 The spectral solver, isolated', link: '/verification/isolated-solver' },
          { text: 'SV-9, SV-12 The hierarchical solver on AMR', link: '/verification/multigrid-amr' },
          { text: 'SV-13 The James step', link: '/verification/james-step' },
          { text: 'Face kinds and the wide D(G)', link: '/verification/face-kinds' },
        ] },
        { text: 'Project', collapsed: true, items: [
          { text: 'Performance',               link: '/guide/performance' },
          { text: "Coming from ADAM's FLAIL",  link: '/guide/from-adam' },
          { text: 'Contributing',              link: '/guide/contributing' },
          { text: 'Changelog',                 link: '/guide/changelog' },
        ] },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/szaghi/ELISHA' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under your choice of GPL-3.0-or-later, BSD-2-Clause, BSD-3-Clause or MIT.',
      copyright: 'Copyright © 2026 Stefano Zaghi',
    },
  },

  mermaid: {},

  vite: {
    // Build with an explicit modern JS target so the docs compile regardless of
    // which mermaid/vitepress/esbuild versions npm resolves. Vite's default
    // es2020 target forces esbuild to down-level modern syntax (e.g. the
    // destructuring mermaid 11.16+ emits), which it refuses to do and the build
    // dies. es2022 needs no lowering and is within VitePress's browser floor.
    build: {
      target: 'es2022',
    },
  },
})
