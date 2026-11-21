<!-- docs/.vitepress/theme/HomeGallery.vue
     The landing page's "see it run", in the home layout's home-hero-after slot (index.ts): right under the hero, before
     the features. Two casts of the tutorial (theme-aware videos) and four figure cards, each linking to its chapter.
     Outside .vp-doc, so each figure is given with its night twin, the theme's .light-only / .dark-only (config.mts)
     showing one: the server-rendered page then matches the client's in both themes. -->
<script setup lang="ts">
import { withBase } from 'vitepress'

const casts = [
  {
    name: 't08-embers',
    ar: 1.0,
    alt: 'Two embers orbiting in free space on an adaptive mesh',
    link: '/tutorial/08-embers',
    caption: 'Two embers in free space: the mesh refined around them at every step, one isolated hierarchical solve per frame, the rings of the potential taking both in at the waist.',
  },
  {
    name: 't06-iterates',
    ar: 1.67,
    alt: 'The error of the FCG iterates of both V-cycles',
    link: '/tutorial/06-iterations',
    caption: "The error of each FCG iterate on an AMR mesh, log scale: the symmetric V-cycle (left) reaches the tolerance in 12 iterations, ADAM's (right) in 19.",
  },
]

const figures = [
  { src: '/tutorial/t01-box', link: '/tutorial/01-box', title: 'A periodic box', text: 'the residual at round-off, the error second order' },
  { src: '/tutorial/t03-isolated', link: '/tutorial/03-isolated', title: 'A star in empty space', text: 'free space on both paths, against a lattice of images' },
  { src: '/tutorial/t04-amr', link: '/tutorial/04-amr', title: 'Refine where it matters', text: 'three levels, a quarter of the coarse error with 4% of the fine cells' },
  { src: '/tutorial/t02-faces', link: '/tutorial/02-faces', title: 'Faces', text: 'what each kind writes into the ghost cell' },
]

const video = (name: string, theme: string, ext: string) => withBase(`/tutorial/casts/${name}-${theme}.${ext}`)
const figure = (src: string, dark: boolean) => withBase(`${src}${dark ? '-dark' : ''}.png`)
</script>

<template>
  <section class="elisha-run" aria-label="See it run">
    <div class="elisha-run-casts">
      <figure v-for="c in casts" :key="c.name" class="elisha-run-cast" :style="{ '--ar': c.ar }">
        <video class="light-only" autoplay loop muted playsinline :poster="video(c.name, 'light', 'png')"
               :aria-label="c.alt">
          <source :src="video(c.name, 'light', 'webm')" type="video/webm">
          <source :src="video(c.name, 'light', 'mp4')" type="video/mp4">
        </video>
        <video class="dark-only" autoplay loop muted playsinline :poster="video(c.name, 'dark', 'png')"
               :aria-label="c.alt">
          <source :src="video(c.name, 'dark', 'webm')" type="video/webm">
          <source :src="video(c.name, 'dark', 'mp4')" type="video/mp4">
        </video>
        <figcaption><a :href="withBase(c.link)">{{ c.caption }}</a></figcaption>
      </figure>
    </div>
    <div class="elisha-run-figures">
      <a v-for="f in figures" :key="f.src" :href="withBase(f.link)" class="elisha-run-card">
        <img class="light-only" :src="figure(f.src, false)" :alt="f.title" loading="lazy">
        <img class="dark-only" :src="figure(f.src, true)" :alt="f.title" loading="lazy">
        <strong>{{ f.title }}</strong>
        <span>{{ f.text }}</span>
      </a>
    </div>
  </section>
</template>
