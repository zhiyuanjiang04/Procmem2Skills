(function () {
  "use strict";

  const content = window.siteContent;
  const results = window.siteResults;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const pct = value => `${Number(value).toFixed(1)}%`;
  const esc = value => String(value).replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));

  function renderAuthors() {
    $("#authors").innerHTML = content.authors.map(author => {
      const marks = author.mark.replace("dagger", "†").replace(",", ", ");
      return `<span class="author">${esc(author.name)}<sup>${esc(marks)}</sup></span>`;
    }).join("");
  }

  function renderFindings() {
    $("#finding-cards").innerHTML = content.findings.map(item => `
      <article class="finding-card">
        <p class="eyebrow">${esc(item.eyebrow)}</p>
        <div class="finding-value">${esc(item.value)}</div>
        <h3>${esc(item.title)}</h3>
        <p>${esc(item.body)}</p>
      </article>`).join("");
  }

  function renderTaxonomy() {
    $("#taxonomy-list").innerHTML = content.taxonomy.map((item, index) => `
      <article class="taxonomy-item${index === 0 ? " is-open" : ""}">
        <button type="button" class="taxonomy-toggle" aria-expanded="${index === 0}" aria-controls="taxonomy-${item.id}">
          <span class="taxonomy-id">${esc(item.id)}</span><strong>${esc(item.title)}</strong><span aria-hidden="true">+</span>
        </button>
        <div class="taxonomy-body" id="taxonomy-${item.id}">${esc(item.body)}</div>
      </article>`).join("");
    $("#mode-row").innerHTML = content.modes.map(mode => `<span class="mode-pill">${esc(mode)}</span>`).join("");
    $$(".taxonomy-toggle").forEach(button => button.addEventListener("click", () => {
      const item = button.closest(".taxonomy-item");
      const open = item.classList.toggle("is-open");
      button.setAttribute("aria-expanded", String(open));
    }));
  }

  function setupRepresentationControls() {
    const benchmarkSelect = $("#representation-benchmark");
    const modelSelect = $("#representation-model");
    const benchmarks = [...new Set(results.representation.map(row => row.benchmark))];
    const models = [...new Set(results.representation.map(row => `${row.agent} / ${row.model}`))];
    benchmarkSelect.innerHTML = benchmarks.map(value => `<option>${esc(value)}</option>`).join("");
    modelSelect.innerHTML = models.map(value => `<option>${esc(value)}</option>`).join("");
    benchmarkSelect.addEventListener("change", renderRepresentation);
    modelSelect.addEventListener("change", renderRepresentation);
    renderRepresentation();
  }

  function renderRepresentation() {
    const benchmark = $("#representation-benchmark").value;
    const model = $("#representation-model").value;
    const row = results.representation.find(item => item.benchmark === benchmark && `${item.agent} / ${item.model}` === model);
    if (!row) return;
    const chart = $("#representation-chart");
    chart.innerHTML = `
      <div class="chart-key" aria-label="Representation chart legend">
        <span class="key-item"><i class="key-swatch raw"></i>Raw baseline marker</span>
        <span class="key-item"><i class="key-swatch workflow"></i>Workflow Memory</span>
        <span class="key-item"><i class="key-swatch skill"></i>Skill</span>
      </div>
      <div class="bar-chart" role="img" aria-label="Grouped horizontal bars comparing Workflow Memory and Skill against Raw across source-trace mixtures">
        <div></div><div class="chart-axis">0</div><div class="chart-axis">50</div><div class="chart-axis">100</div>
        ${results.settings.map((setting, index) => `
          <div class="chart-setting">${setting}</div>
          <div class="bar-track" style="--raw:${row.raw}%"><div class="bar workflow" style="width:${row.workflow[index]}%"><span class="bar-label">${pct(row.workflow[index])}</span></div></div>
          <div class="bar-track" style="--raw:${row.raw}%"><div class="bar skill" style="width:${row.skill[index]}%"><span class="bar-label">${pct(row.skill[index])}</span></div></div>
          <div class="bar-track" style="--raw:${row.raw}%"><div class="bar" style="width:${row.raw}%;background:var(--gray-bar)"><span class="bar-label">${pct(row.raw)}</span></div></div>`).join("")}
      </div>`;
    $("#representation-table-wrap").innerHTML = `
      <table class="data-table"><caption class="sr-only">Representation results for ${esc(model)} on ${esc(benchmark)}</caption>
        <thead><tr><th>Setting</th><th>Raw</th><th>Workflow</th><th>Skill</th></tr></thead>
        <tbody>${results.settings.map((setting, index) => `<tr><td>${setting}</td><td>${pct(row.raw)}</td><td>${pct(row.workflow[index])}</td><td><strong>${pct(row.skill[index])}</strong></td></tr>`).join("")}</tbody>
      </table>`;
  }

  const armCopy = {
    embedding: ["ARM 1 / EMBEDDING RANKING", "Task instructions are matched to skill descriptions with Qwen3-Embedding-0.6B. No downstream task is executed."],
    selection: ["ARM 2 / LET AGENTS PICK", "The agent explicitly selects useful skills from the pool, without executing the downstream task or receiving verifier feedback."],
    execution: ["ARM 3 / REAL EXECUTION", "The full candidate pool is available during task execution; parsed actual-use metrics are reported alongside final verifier success."]
  };

  function setupRetrievalControls() {
    $$(".arm-button").forEach(button => button.addEventListener("click", () => {
      $$(".arm-button").forEach(item => item.classList.remove("is-active"));
      button.classList.add("is-active");
      $("#retrieval-metric").dataset.arm = button.dataset.arm;
      setRetrievalMetrics(button.dataset.arm);
      renderRetrieval();
    }));
    $("#retrieval-composition").addEventListener("change", renderRetrieval);
    $("#retrieval-metric").addEventListener("change", renderRetrieval);
    $("#retrieval-k").addEventListener("change", renderRetrieval);
    setRetrievalMetrics("embedding");
    renderRetrieval();
  }

  function setRetrievalMetrics(arm) {
    const metricSelect = $("#retrieval-metric");
    const metrics = arm === "embedding" ? [["precision", "Top-1 precision"]] : arm === "selection" ? [["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"]] : [["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["success", "Task success"]];
    metricSelect.innerHTML = metrics.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
    metricSelect.value = metrics[0][0];
  }

  function retrievalSeries(arm, metric, composition) {
    if (arm === "embedding") {
      return results.embedding.filter(row => row.composition === composition).map(row => ({ k: row.k, value: row[metric] }));
    }
    return results.averageRetrieval(arm, metric, composition);
  }

  function makeSvgChart(series, arm, metric) {
    const width = 760;
    const height = 255;
    const left = 46;
    const right = 16;
    const top = 18;
    const bottom = 38;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const x = index => left + (series.length === 1 ? plotWidth / 2 : index * plotWidth / (series.length - 1));
    const y = value => top + plotHeight - (value / 100) * plotHeight;
    const points = series.map((row, index) => `${x(index)},${y(row.value)}`).join(" ");
    const grid = [0, 25, 50, 75, 100].map(value => `<line class="gridline" x1="${left}" x2="${width - right}" y1="${y(value)}" y2="${y(value)}"/><text class="axis-label" x="4" y="${y(value) + 4}">${value}</text>`).join("");
    const xLabels = series.map((row, index) => `<text class="axis-label" text-anchor="middle" x="${x(index)}" y="${height - 12}">${row.k}</text>`).join("");
    const circles = series.map((row, index) => `<circle class="${arm}" cx="${x(index)}" cy="${y(row.value)}" r="4" tabindex="0"><title>Pool size ${row.k}: ${pct(row.value)}</title></circle>`).join("");
    return `<svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(metric)} across skill pool sizes"><g>${grid}${xLabels}<polyline class="series-line ${arm}" points="${points}"/>${circles}</g></svg>`;
  }

  function renderRetrieval() {
    const arm = $("#retrieval-metric").dataset.arm || "embedding";
    const composition = $("#retrieval-composition").value;
    const metric = $("#retrieval-metric").value;
    const selectedK = Number($("#retrieval-k").value);
    const [label, copy] = armCopy[arm];
    $("#retrieval-arm-label").textContent = label;
    $("#retrieval-arm-copy").textContent = copy;
    const series = retrievalSeries(arm, metric, composition);
    $("#retrieval-chart").innerHTML = `<div class="chart-key"><span class="key-item"><i class="key-swatch ${arm === "embedding" ? "workflow" : arm === "selection" ? "selection" : "skill"}"></i>${esc($("#retrieval-metric").selectedOptions[0].textContent)} · ${esc(composition)}</span><span class="key-item">x-axis: candidate-pool size k</span></div>${makeSvgChart(series, arm, metric)}`;
    const atFive = series.find(row => row.k === 5);
    const atHundred = series.find(row => row.k === 100);
    $("#retrieval-highlight").innerHTML = [atFive, atHundred].filter(Boolean).map((row, index) => `<div class="highlight-chip"><strong>${pct(row.value)}</strong><span>Average ${$("#retrieval-metric").selectedOptions[0].textContent.toLowerCase()} at k=${index === 0 ? 5 : 100}</span></div>`).join("");
    const tableRows = series.map(row => `<tr${row.k === selectedK ? ' class="selected-row"' : ""}><td>k=${row.k}</td><td>${pct(row.value)}</td></tr>`).join("");
    $("#retrieval-table-wrap").innerHTML = `<table class="data-table"><caption class="sr-only">${esc(label)} ${esc(composition)} results for ${esc(metric)}</caption><thead><tr><th>Pool size</th><th>${esc($("#retrieval-metric").selectedOptions[0].textContent)}</th></tr></thead><tbody>${tableRows}</tbody></table>`;
  }

  function setupLightbox() {
    const lightbox = $("#lightbox");
    const image = $("#lightbox-image");
    const close = () => { lightbox.hidden = true; document.body.classList.remove("no-scroll"); image.src = ""; };
    $$('[data-lightbox]').forEach(button => button.addEventListener("click", () => {
      image.src = button.dataset.lightbox;
      image.alt = $("img", button).alt;
      lightbox.hidden = false;
      document.body.classList.add("no-scroll");
    }));
    $("#lightbox-close").addEventListener("click", close);
    lightbox.addEventListener("click", event => { if (event.target === lightbox) close(); });
    document.addEventListener("keydown", event => { if (event.key === "Escape" && !lightbox.hidden) close(); });
  }

  function setupNavigation() {
    const toggle = $(".nav-toggle");
    const links = $("#nav-links");
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    $$(".nav-links a").forEach(link => link.addEventListener("click", () => {
      links.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
    }));
  }

  function setupBibtex() {
    $("#bibtex-code").textContent = content.bibtex;
    $("#copy-bibtex").addEventListener("click", async event => {
      try {
        await navigator.clipboard.writeText(content.bibtex);
        event.currentTarget.textContent = "Copied";
        setTimeout(() => { event.currentTarget.textContent = "Copy"; }, 1400);
      } catch (error) {
        event.currentTarget.textContent = "Select text";
      }
    });
  }

  function init() {
    $("#paper-title").innerHTML = content.title.replace(": ", ":<br><em>").replace(" - Until They Don't", " - Until They Don't</em>");
    $("#hero-question").textContent = content.question;
    renderAuthors();
    renderFindings();
    renderTaxonomy();
    setupRepresentationControls();
    setupRetrievalControls();
    setupLightbox();
    setupNavigation();
    setupBibtex();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
