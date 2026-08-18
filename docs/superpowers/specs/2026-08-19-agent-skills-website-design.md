# Agent Skills Project Website Design

## Objective

Create a GitHub Pages-compatible project website for *Demystifying Agent Skills: Why They Work—Until They Don't* in the `website` branch of `zhiyuanjiang04/Procmem2Skills`. The page should follow the paper-project-page structure and visual language of Nerfies while presenting the paper's measured findings, experimental pipelines, and retrieval analysis in a compact, readable form.

The website is a static artifact. It must work when served from a repository subpath and must not require a server-side API, build step, API key, or runtime data service.

## Scope

### In scope

- A single responsive project page with anchor navigation.
- Paper title, author list, abstract-level summary, paper/code links, and BibTeX.
- The two existing experimental pipeline figures.
- Interactive summaries for representation, procedural anchoring, and retrieval.
- The three retrieval arms presented as independent diagnostics/experiments.
- Existing paper figures reused where they make the result easier to inspect.
- Exact values shown through hover tooltips and accessible text fallbacks.
- GitHub Pages deployment metadata and concise usage instructions.

### Out of scope

- A separate limitations section.
- User accounts, server-side storage, or experiment reruns.
- Replacing the paper's authoritative tables or adding results not present in the paper directory.
- A marketing landing page or decorative 3D scene.

## Information Architecture

1. **Hero**
   - Title, authors, affiliations, central research question, and links to paper/code/BibTeX.
   - A short result strip with the most representative numbers.
2. **Overview**
   - One-paragraph motivation and the two experimental pipeline images.
3. **Representation**
   - Skill vs. Workflow Memory comparison.
   - Interactive benchmark/setting selector and raw/workflow/skill values.
4. **Mechanisms**
   - Three taxonomy categories, twelve modes, and a compact paired-trajectory explanation.
   - Existing taxonomy visualization and a concise procedural-anchor result.
5. **Retrieval**
   - Independent sections for Arm 1 embedding retrieval, Arm 2 explicit agent selection, and Arm 3 real execution.
   - Controls for arm, pool size, pool composition, and metric where applicable.
   - Exact precision/recall/F1/success values in tooltips and a textual table fallback.
6. **Resources**
   - Paper PDF, source repository, BibTeX, and selected implementation notes.

## Visual Direction

- Use Nerfies-like centered composition, a restrained top navigation, large serif headings, compact sans-serif supporting text, and framed media blocks.
- Use a warm-white page background with dark ink text and a controlled blue/teal/orange accent system.
- Keep cards shallow and functional. Avoid gradients, oversized decorative blobs, and nested card stacks.
- Use consistent scientific chart colors across every experiment: raw as neutral gray, workflow as blue, skill as teal/green, and negative/comparison states as orange/red only when semantically needed.
- Apply subtle hover elevation, line emphasis, and tooltip transitions. Animation must not carry meaning by itself.
- All charts need visible labels or an adjacent data table; hover is an enhancement, not the only way to access a number.

## Data and Component Design

The page separates content data from rendering logic:

- `static/data/results.js` stores the validated paper values, benchmark names, settings, pool regimes, arm names, and citation metadata.
- `static/js/site.js` handles navigation state, chart rendering, selectors, tooltips, reduced-motion behavior, and accessible table updates.
- `static/css/site.css` contains the Nerfies-inspired layout and responsive styles.
- Existing paper figures are copied into `static/images/` with stable, descriptive filenames.

Interactive components:

- **Result cards**: show a label, value, comparison text, and source context.
- **Representation chart**: grouped bars or lollipop marks for Raw, Workflow Memory, and Skill; benchmark and setting controls update the exact-value panel.
- **Mechanism panel**: taxonomy categories expand to reveal their modes without hiding the main conclusion.
- **Retrieval chart**: a line/bar view changes with arm, pool size, pool composition, and metric; the selected state is reflected in a compact table.
- **Figure viewer**: existing pipeline figures support click-to-enlarge without changing the page layout.

## Content Rules

- Use only numbers and claims present in the supplied paper directory.
- Keep causal wording aligned with the paper: retrieval diagnostics are independent from the downstream execution experiment, and their outputs are not passed into Arm 3.
- State the main experimental scope near the relevant visualization rather than putting full experimental setup in every chart caption.
- Use concise explanatory copy and link to the paper for full methodological details.

## Deployment and Repository

- Base the implementation on the public Nerfies static template and retain its permissive attribution/license notices where required.
- Work in `/Users/jiangzhiyuan/Documents/PM2Skills/Procmem2Skills-website` on branch `website`.
- Keep the website self-contained under the repository root so GitHub Pages can serve it from `/Procmem2Skills/`.
- Add a README section describing local preview and GitHub Pages publishing.

## Validation

- Run a local static server and inspect desktop and mobile layouts.
- Verify all paper/code/PDF links and repository-relative asset paths.
- Test selectors, hover tooltips, figure enlargement, keyboard focus, and reduced-motion behavior.
- Check that all chart data has a visible non-hover fallback.
- Confirm that the page renders correctly under the GitHub Pages project subpath.
