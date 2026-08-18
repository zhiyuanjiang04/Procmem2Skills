window.siteResults = {
  settings: ["5s0f", "4s1f", "3s2f", "2s3f", "1s4f", "0s5f"],
  representation: [
    { agent: "Codex", model: "GPT-5.3-Codex", benchmark: "Terminal-Bench 2", raw: 59.35, workflow: [44.52, 40.00, 41.94, 36.77, 27.10, 28.39], skill: [75.48, 72.90, 78.06, 68.39, 70.97, 51.61] },
    { agent: "Codex", model: "GPT-5.3-Codex", benchmark: "SkillsBench", raw: 50.83, workflow: [52.50, 56.67, 64.17, 60.83, 51.67, 58.33], skill: [72.50, 61.67, 62.50, 70.83, 61.67, 45.00] },
    { agent: "Codex", model: "GPT-5.3-Codex", benchmark: "Terminal-Bench-Pro", raw: 53.94, workflow: [73.33, 73.33, 69.70, 66.67, 61.21, 47.88], skill: [74.55, 79.39, 73.33, 66.67, 58.18, 43.03] },
    { agent: "Gemini CLI", model: "Gemini-3.1-Pro-Preview", benchmark: "Terminal-Bench 2", raw: 50.00, workflow: [62.31, 53.08, 64.62, 60.00, 58.46, 52.31], skill: [79.23, 76.15, 74.62, 70.00, 69.23, 47.69] },
    { agent: "Gemini CLI", model: "Gemini-3.1-Pro-Preview", benchmark: "SkillsBench", raw: 47.62, workflow: [55.24, 52.38, 56.19, 55.24, 48.57, 42.86], skill: [74.29, 61.90, 66.67, 67.62, 60.00, 42.86] },
    { agent: "Gemini CLI", model: "Gemini-3.1-Pro-Preview", benchmark: "Terminal-Bench-Pro", raw: 56.15, workflow: [53.08, 69.23, 64.62, 68.46, 59.23, 47.69], skill: [66.92, 63.08, 54.62, 50.77, 56.92, 46.15] }
  ],
  embedding: [
    ["Random", 5, 97.7], ["Random", 10, 95.5], ["Random", 20, 95.5], ["Random", 50, 92.0], ["Random", 100, 84.1],
    ["Similar", 5, 70.5], ["Similar", 10, 63.6], ["Similar", 20, 60.2], ["Similar", 50, 56.8], ["Similar", 100, 53.4],
    ["Dissimilar", 5, 96.6], ["Dissimilar", 10, 96.6], ["Dissimilar", 20, 96.6], ["Dissimilar", 50, 94.3], ["Dissimilar", 100, 93.2]
  ].map(([composition, k, precision]) => ({ composition, k, precision })),
  retrieval: []
};

const pushRetrievalRows = (model, rows) => {
  rows.forEach(([composition, k, selectionP, selectionR, selectionF1, executionP, executionR, executionF1, success]) => {
    window.siteResults.retrieval.push({ model, composition, k, arm: "selection", precision: selectionP, recall: selectionR, f1: selectionF1 });
    window.siteResults.retrieval.push({ model, composition, k, arm: "execution", precision: executionP, recall: executionR, f1: executionF1, success });
  });
};

pushRetrievalRows("Gemini CLI + Gemini-3.1-Pro-Preview", [
  ["Random", 5, 74.4, 75.0, 74.7, 18.6, 69.4, 29.3, 39.2], ["Random", 10, 72.7, 72.7, 72.7, 7.1, 66.4, 12.9, 38.1], ["Random", 20, 80.1, 81.8, 81.0, 3.6, 66.0, 6.8, 39.2], ["Random", 50, 81.2, 84.1, 82.6, 1.4, 65.8, 2.8, 37.6], ["Random", 100, 77.4, 83.0, 80.1, 0.7, 66.0, 1.4, 39.0],
  ["Similar", 5, 54.3, 69.3, 60.9, 17.6, 70.1, 28.1, 38.8], ["Similar", 10, 61.3, 79.5, 69.2, 7.0, 67.3, 12.6, 38.5], ["Similar", 20, 58.4, 77.3, 66.6, 3.7, 66.2, 6.9, 34.0], ["Similar", 50, 59.3, 76.1, 66.7, 1.4, 66.2, 2.7, 36.7], ["Similar", 100, 55.4, 73.9, 63.3, 0.7, 66.7, 1.4, 36.1],
  ["Dissimilar", 5, 74.4, 76.1, 75.3, 14.6, 63.2, 23.7, 34.0], ["Dissimilar", 10, 80.1, 81.8, 81.0, 8.7, 66.0, 15.3, 38.8], ["Dissimilar", 20, 82.4, 83.0, 82.7, 5.1, 65.5, 9.5, 35.6], ["Dissimilar", 50, 79.3, 80.7, 80.0, 1.6, 64.4, 3.1, 38.1], ["Dissimilar", 100, 83.5, 84.1, 83.8, 0.7, 61.6, 1.5, 34.5]
]);

pushRetrievalRows("Codex + GPT-5.4", [
  ["Random", 5, 81.8, 84.1, 82.9, 33.1, 41.6, 36.8, 24.3], ["Random", 10, 83.0, 86.4, 84.6, 39.2, 59.7, 47.3, 35.5], ["Random", 20, 84.1, 87.5, 85.8, 35.3, 69.3, 46.8, 40.9], ["Random", 50, 71.7, 87.5, 78.8, 15.8, 63.5, 25.4, 35.0], ["Random", 100, 62.2, 85.2, 71.9, 8.1, 73.6, 14.6, 44.8],
  ["Similar", 5, 51.9, 81.8, 63.5, 51.3, 72.4, 60.0, 44.5], ["Similar", 10, 44.4, 83.0, 57.8, 37.5, 66.0, 47.9, 40.7], ["Similar", 20, 35.8, 77.3, 48.9, 27.7, 66.1, 39.1, 44.3], ["Similar", 50, 37.8, 76.1, 50.5, 13.1, 61.8, 21.6, 42.3], ["Similar", 100, 31.9, 70.5, 43.9, 6.7, 54.3, 11.9, 43.0],
  ["Dissimilar", 5, 83.3, 85.2, 84.3, 42.5, 63.0, 50.7, 37.3], ["Dissimilar", 10, 83.0, 85.2, 84.1, 29.7, 61.3, 40.0, 35.0], ["Dissimilar", 20, 82.8, 85.2, 84.0, 12.7, 60.5, 21.0, 31.8], ["Dissimilar", 50, 77.5, 85.2, 81.2, 7.2, 65.1, 12.9, 39.5], ["Dissimilar", 100, 72.0, 85.2, 78.0, 2.7, 60.2, 5.2, 38.2]
]);

window.siteResults.averageRetrieval = function(arm, metric, composition) {
  const rows = window.siteResults.retrieval.filter(row => row.arm === arm && row.composition === composition);
  const grouped = new Map();
  rows.forEach(row => {
    if (!grouped.has(row.k)) grouped.set(row.k, []);
    grouped.get(row.k).push(row[metric]);
  });
  return [...grouped.entries()].map(([k, values]) => ({ k, value: values.reduce((a, b) => a + b, 0) / values.length }));
};
