import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid({
  title: 'ELISHA',
  description: 'Elliptic Laplacian Inversion: Spectral and Hierarchical Algorithms: FFT and AMR multigrid Poisson solvers for HPC',
  base: '/ELISHA/',
  srcExclude: ['ford.md'],

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
    nav: [
      { text: 'Home',   link: '/' },
      { text: 'Guide',  link: '/guide/' },
      { text: 'GitHub', link: 'https://github.com/szaghi/ELISHA' },
    ],

    // Add your guide pages here.
    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'About',        link: '/guide/' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start',  link: '/guide/quickstart' },
          ],
        },
        {
          text: 'Using ELISHA',
          items: [
            { text: 'The solver facade', link: '/guide/solver' },
            { text: 'Examples', link: '/guide/examples' },
            { text: "Coming from ADAM's FLAIL", link: '/guide/from-adam' },
            { text: 'Performance', link: '/guide/performance' },
          ],
        },
        {
          text: 'Layers',
          items: [
            { text: 'L0: the math', link: '/guide/l0-math' },
            { text: 'Errors, communicator and mesh', link: '/guide/errors-comm-mesh' },
            { text: 'The spectral solver', link: '/guide/spectral-solver' },
            { text: 'The hierarchical solver', link: '/guide/hierarchical-solver' },
            { text: 'The device backends', link: '/guide/device' },
          ],
        },
        {
          text: 'Project',
          items: [
            { text: 'Contributing', link: '/guide/contributing' },
            { text: 'Changelog',    link: '/guide/changelog' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/szaghi/ELISHA' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the <a href="http://www.gnu.org/licenses/gpl-3.0.html">GPL v3 License</a>.',
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
