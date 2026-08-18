# Agent Skills Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Nerfies-inspired, GitHub Pages-compatible project website for *Demystifying Agent Skills: Why They Work—Until They Don't* on the `website` branch.

**Architecture:** Use the Nerfies static HTML/CSS structure as the visual base, keep all experiment data in a standalone JavaScript data module, and render lightweight interactive charts with SVG/HTML and native JavaScript. The page remains deployable from a repository subpath without a build service.

**Tech Stack:** HTML5, CSS3, native JavaScript, inline SVG/HTML chart components, GitHub Pages.

---

### Task 1: Import the static template and establish site shell

**Files:**
- Create/modify: `index.html`
- Create/modify: `static/css/site.css`
- Create/modify: `static/js/site.js`
- Create: `static/images/`
- Modify: `.gitignore`

- [ ] **Step 1: Copy the Nerfies static shell into the website branch**

  Preserve the responsive centered layout, typography hierarchy, navigation pattern, and attribution/license notice. Remove paper-specific Nerfies copy and unused demo assets.

- [ ] **Step 2: Add the project page structure**

  Add semantic sections for hero, overview, representation, mechanisms, retrieval, and resources. Use repository-relative links so the page works under `/Procmem2Skills/`.

- [ ] **Step 3: Add the responsive visual system**

  Implement the warm-white background, ink text, restrained blue/teal/orange accents, framed media blocks, responsive tables, focus states, reduced-motion behavior, and mobile navigation.

- [ ] **Step 4: Commit the site shell**

  ```bash
  git add index.html static/css/site.css static/js/site.js static/images .gitignore
  git commit -m "Add Nerfies-inspired project page shell"
  ```

### Task 2: Extract and register paper content and assets

**Files:**
- Create: `static/data/results.js`
- Create: `static/data/content.js`
- Modify: `index.html`
- Create: `static/images/experimental-pipeline-procmem-skills.png`
- Create: `static/images/experimental-pipeline-retrieval.png`
- Create: `static/images/taxonomy-per-setting.pdf`
- Create: `static/images/retrieval-arm1-embedding.png`
- Create: `static/images/retrieval-arm2-agent-pick.png`
- Create: `static/images/retrieval-arm3-precision-success.png`

- [ ] **Step 1: Copy the supplied paper figures**

  Copy only the existing figures from `Demystifying_Agent_Skills/latex/assets/` that are used on the page. Preserve source filenames in a small asset manifest and use descriptive website filenames.

- [ ] **Step 2: Create content data**

  Store title, authors, abstract summary, research question text, taxonomy labels, resource URLs, and the checked headline findings separately from rendering code.

- [ ] **Step 3: Create result data**

  Encode only confirmed paper values for representation, outcome annotation, and retrieval. Include benchmark, setting, pool size, composition, arm, metric, value, and display label fields. Keep values as fractions in data and format percentages in the UI.

- [ ] **Step 4: Add visible source notes**

  Each chart module receives a short methodology note and a link to the relevant paper section or appendix table.

### Task 3: Implement representation and mechanism modules

**Files:**
- Modify: `index.html`
- Modify: `static/js/site.js`
- Modify: `static/css/site.css`

- [ ] **Step 1: Implement result cards**

  Add cards for the representative findings, including the matched Skill-vs-Workflow comparison, procedural anchoring contrast, and retrieval-use decline.

- [ ] **Step 2: Implement the representation explorer**

  Render grouped bars or lollipop marks for Raw, Workflow Memory, and Skill. Add benchmark and setting controls, exact-value tooltips, and an accessible text table updated with the same selection.

- [ ] **Step 3: Implement the mechanism panel**

  Add the three taxonomy categories and twelve modes as expandable semantic rows. Include the paired-trajectory explanation without reproducing the full appendix transcript.

- [ ] **Step 4: Add the pipeline and taxonomy figures**

  Add responsive framed images, click-to-enlarge behavior, captions, and alt text.

### Task 4: Implement the retrieval explorer

**Files:**
- Modify: `index.html`
- Modify: `static/js/site.js`
- Modify: `static/css/site.css`

- [ ] **Step 1: Add the three independent arm descriptions**

  State that Arm 1 embedding ranking, Arm 2 explicit agent selection, and Arm 3 full-pool execution are separate experiments; outputs from Arms 1 and 2 are not passed to Arm 3.

- [ ] **Step 2: Render retrieval controls**

  Add controls for arm, pool size, composition, and metric. Disable controls that do not apply to the selected arm rather than showing misleading empty values.

- [ ] **Step 3: Render the chart and exact-value panel**

  Use one consistent chart component with hover emphasis, value labels for the active series, and a synchronized table fallback containing precision, recall, F1, and success where available.

- [ ] **Step 4: Add the retrieval figures**

  Present the existing retrieval figures as supplementary visual evidence with captions and links to the interactive data view.

### Task 5: Add resources, documentation, and local preview

**Files:**
- Modify: `index.html`
- Modify: `README.md`
- Create: `404.html`
- Create: `.nojekyll`

- [ ] **Step 1: Add resource links**

  Add paper PDF, source repository, BibTeX, and relevant appendix references.

- [ ] **Step 2: Add project-subpath fallback**

  Provide a small `404.html` fallback and avoid absolute asset paths so GitHub Pages project hosting works.

- [ ] **Step 3: Document local preview and deployment**

  Add commands using a simple local HTTP server and GitHub Pages branch settings to `README.md`.

- [ ] **Step 4: Commit the complete content layer**

  ```bash
  git add index.html static README.md 404.html .nojekyll
  git commit -m "Add paper results and retrieval explorer"
  ```

### Task 6: Verify visual, interaction, and deployment behavior

**Files:**
- Modify: `static/css/site.css` or `static/js/site.js` as needed after QA

- [ ] **Step 1: Run a local static server**

  Serve the repository root over HTTP and verify that assets load under both `/` and `/Procmem2Skills/`-style paths.

- [ ] **Step 2: Verify desktop and mobile layouts**

  Check the hero, navigation, cards, figures, charts, tables, and retrieval controls at wide desktop and narrow mobile widths. Confirm no text or chart labels overlap.

- [ ] **Step 3: Verify interactions**

  Test keyboard navigation, hover/focus tooltips, selector changes, figure enlargement, reduced-motion mode, and accessible table fallbacks.

- [ ] **Step 4: Verify content integrity**

  Compare every displayed numeric value against `acl_latex.tex` and the supplied paper assets. Confirm no unsupported claims or stale TODOs are present.

- [ ] **Step 5: Commit final QA fixes**

  ```bash
  git add .
  git commit -m "Polish and verify project website"
  ```
