// docs/.vitepress/theme/dark-figures.data.ts (copied from ADAM's docs theme by way of SETH's, unchanged below this line)
// The figures that have a night twin: every docs/public/<path>/<name>.png next to which <name>-dark.png exists
// (written by docs/scripts/make_doc_figures.py, on the Kōyō night ground). A VitePress data loader: the list is
// computed at build time (and refreshed by the dev server when a twin is added), so a new twin needs no edit here or in
// a page. index.ts swaps the images on the client.
import { existsSync } from 'node:fs'
import { relative, resolve } from 'node:path'

const PUBLIC = resolve(__dirname, '../../public')

export default {
  watch: ['../../public/**/*-dark.png'],
  load(files: string[]): string[] {
    return files
      .map((f) => relative(PUBLIC, f).split('\\').join('/').replace(/-dark\.png$/, '.png'))
      .filter((p) => existsSync(resolve(PUBLIC, p)))
      .sort()
  },
}
