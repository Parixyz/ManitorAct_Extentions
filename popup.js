const STOPWORDS = new Set(["the", "and", "for", "with", "www", "http", "https", "com", "org", "net", "to", "of", "in", "on", "a", "an", "is", "home", "page"]);

const CLASS_LABELS = ["Work", "Personal Project", "Research", "Teaching Assistant", "Course Work", "Entertainment", "Personal", "Other"];

const CLASS_TERMS = {
  Work: ["jira", "slack", "meeting", "client", "sprint", "calendar", "mail", "docs", "confluence"],
  "Personal Project": ["github", "repo", "feature", "build", "prototype", "debug", "deploy", "sideproject"],
  Research: ["uav", "drone", "mobile", "paper", "arxiv", "study", "dataset", "experiment", "method"],
  "Teaching Assistant": ["1p02", "ta", "grading", "office", "hours", "rubric", "student", "tutorial", "moodle"],
  "Course Work": ["assignment", "lecture", "quiz", "course", "syllabus", "midterm", "final", "homework", "lab"],
  Entertainment: ["youtube", "netflix", "movie", "music", "video", "stream", "twitch", "watch"],
  Personal: ["shopping", "amazon", "reddit", "instagram", "facebook", "travel", "food"],
  Other: []
};

function parseUrl(url = "") {
  try {
    const u = new URL(url);
    return { domain: u.hostname.replace(/^www\./, ""), path: u.pathname.replace(/^\//, "") || "root" };
  } catch {
    return { domain: "unknown", path: "root" };
  }
}

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((t) => t.length > 2 && !STOPWORDS.has(t));
}

function buildDocs(activities) {
  const basic = activities.map((a) => {
    const { domain, path } = parseUrl(a.url);
    const title = tokenize(a.title || "");
    const pathTokens = tokenize(path.replace(/[/-]/g, " "));
    const content = tokenize(a.contentSnippet || "");
    const tokens = [...title, ...pathTokens, ...domain.split(/[.-]/), ...content.slice(0, 120), `domain_${domain}`].filter(Boolean);
    const bigrams = [];
    for (let i = 0; i < Math.min(tokens.length - 1, 20); i += 1) bigrams.push(`bg_${tokens[i]}_${tokens[i + 1]}`);
    return { tokens: [...tokens, ...bigrams], domain };
  });

  return basic.map((doc, idx) => {
    const prev = basic[idx + 1];
    const next = basic[idx - 1];
    return [
      ...doc.tokens,
      ...(prev ? [`prev_domain_${prev.domain}`, ...prev.tokens.slice(0, 8).map((x) => `prev_${x}`)] : []),
      ...(next ? [`next_domain_${next.domain}`, ...next.tokens.slice(0, 8).map((x) => `next_${x}`)] : [])
    ];
  });
}

function buildTfIdfVectors(activities) {
  const docs = buildDocs(activities);
  const vocab = [...new Set(docs.flat())];
  const index = Object.fromEntries(vocab.map((t, i) => [t, i]));
  const df = new Array(vocab.length).fill(0);
  docs.forEach((doc) => new Set(doc).forEach((t) => { df[index[t]] += 1; }));

  const vectors = docs.map((doc) => {
    const counts = new Array(vocab.length).fill(0);
    doc.forEach((t) => { counts[index[t]] += 1; });
    const total = Math.max(1, doc.length);
    return counts.map((c, i) => (c / total) * (Math.log((docs.length + 1) / (df[i] + 1)) + 1));
  });

  return { docs, vectors, vocab };
}

function cosine(a, b) {
  let dot = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return !na || !nb ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function termVector(tokensByClass) {
  return CLASS_LABELS.map((label) => tokensByClass.reduce((acc, token) => acc + (CLASS_TERMS[label].includes(token) ? 1 : 0), 0));
}

function heuristicLabel(activity, tokens) {
  const text = `${activity.title || ""} ${activity.url || ""} ${activity.contentSnippet || ""}`.toLowerCase();
  if (text.includes("1p02")) return "Teaching Assistant";
  const hasUavMobile = /\b(uav|drone|mobile)\b/.test(text);
  const hasCourse = /\b(course|assignment|lecture|quiz|midterm|final|lab)\b/.test(text);
  if (hasUavMobile && hasCourse) return "Course Work";
  if (hasUavMobile) return "Research";
  if (/\b(youtube|netflix|watch|video|music)\b/.test(text)) return "Entertainment";
  if (/\b(ta|grading|rubric|moodle|office hours|student)\b/.test(text)) return "Teaching Assistant";
  if (/\b(jira|slack|meeting|confluence|calendar)\b/.test(text)) return "Work";
  if (/\b(github|repo|deploy|prototype|feature)\b/.test(text)) return "Personal Project";
  if (/\b(course|assignment|lecture|quiz)\b/.test(text)) return "Course Work";
  return "Other";
}

function trainNaiveBayes(rows) {
  const model = { classCounts: {}, tokenCounts: {}, totals: {}, vocab: new Set() };
  CLASS_LABELS.forEach((c) => { model.classCounts[c] = 0; model.tokenCounts[c] = {}; model.totals[c] = 0; });
  rows.forEach(({ label, tokens }) => {
    model.classCounts[label] += 1;
    tokens.forEach((t) => {
      model.vocab.add(t);
      model.tokenCounts[label][t] = (model.tokenCounts[label][t] || 0) + 1;
      model.totals[label] += 1;
    });
  });
  return model;
}

function predictNaiveBayes(model, tokens) {
  const vocabSize = Math.max(1, model.vocab.size);
  let best = { label: "Other", score: -Infinity };
  const totalClasses = Object.values(model.classCounts).reduce((a, b) => a + b, 0) + CLASS_LABELS.length;

  CLASS_LABELS.forEach((label) => {
    let score = Math.log((model.classCounts[label] + 1) / totalClasses);
    tokens.forEach((t) => {
      score += Math.log(((model.tokenCounts[label][t] || 0) + 1) / (model.totals[label] + vocabSize));
    });
    if (score > best.score) best = { label, score };
  });

  return best.label;
}

function knnLabel(targetTokens, labeledRows, k = 5) {
  const setA = new Set(targetTokens);
  const ranked = labeledRows.map((row) => {
    const setB = new Set(row.tokens);
    let common = 0;
    setA.forEach((t) => { if (setB.has(t)) common += 1; });
    return { label: row.label, sim: common / Math.sqrt(Math.max(1, setA.size * setB.size)) };
  }).sort((a, b) => b.sim - a.sim).slice(0, k);
  const votes = {};
  ranked.forEach((r) => { votes[r.label] = (votes[r.label] || 0) + r.sim; });
  return Object.entries(votes).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
}

function classifyActivities(activities, docs, userLabels) {
  const rows = [];
  activities.forEach((a, i) => rows.push({ label: userLabels[a.id] || heuristicLabel(a, docs[i]), tokens: docs[i] }));
  const model = trainNaiveBayes(rows);

  return activities.map((a, i) => {
    if (userLabels[a.id]) return userLabels[a.id];
    const nb = predictNaiveBayes(model, docs[i]);
    const knn = knnLabel(docs[i], rows, 5);
    return knn && knn === nb ? nb : nb;
  });
}

function semanticSimilarity(i, j, activities, vectors, docs, labels) {
  const urlA = parseUrl(activities[i].url);
  const urlB = parseUrl(activities[j].url);
  if (activities[i].normalizedUrl && activities[i].normalizedUrl === activities[j].normalizedUrl) return 1;
  const lexical = cosine(vectors[i], vectors[j]);
  const classSim = labels[i] === labels[j] ? 1 : 0;
  const semA = termVector(docs[i]);
  const semB = termVector(docs[j]);
  const termSim = cosine(semA, semB);
  const sameDomainBoost = urlA.domain === urlB.domain ? 0.3 : 0;
  return Math.min(1, lexical * 0.5 + termSim * 0.35 + classSim * 0.15 + sameDomainBoost);
}

function kmeans(vectors, k, iterations = 10) {
  const centroids = vectors.slice(0, k).map((v) => [...v]);
  const assign = new Array(vectors.length).fill(0);
  for (let it = 0; it < iterations; it += 1) {
    for (let i = 0; i < vectors.length; i += 1) {
      let best = 0; let bestS = -1;
      for (let c = 0; c < k; c += 1) {
        const s = cosine(vectors[i], centroids[c]);
        if (s > bestS) { bestS = s; best = c; }
      }
      assign[i] = best;
    }
    const sums = Array.from({ length: k }, () => new Array(vectors[0].length).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < vectors.length; i += 1) {
      const c = assign[i]; counts[c] += 1;
      for (let d = 0; d < vectors[i].length; d += 1) sums[c][d] += vectors[i][d];
    }
    for (let c = 0; c < k; c += 1) if (counts[c]) for (let d = 0; d < sums[c].length; d += 1) centroids[c][d] = sums[c][d] / counts[c];
  }
  return assign;
}

function agglomerative(vectors, k) {
  const clusters = vectors.map((_, i) => [i]);
  const avg = (a, b) => {
    let s = 0; let n = 0;
    a.forEach((i) => b.forEach((j) => { s += cosine(vectors[i], vectors[j]); n += 1; }));
    return n ? s / n : 0;
  };
  while (clusters.length > k) {
    let best = { i: 0, j: 1, s: -1 };
    for (let i = 0; i < clusters.length; i += 1) for (let j = i + 1; j < clusters.length; j += 1) {
      const s = avg(clusters[i], clusters[j]);
      if (s > best.s) best = { i, j, s };
    }
    clusters[best.i] = [...clusters[best.i], ...clusters[best.j]];
    clusters.splice(best.j, 1);
  }
  const out = new Array(vectors.length).fill(0);
  clusters.forEach((c, idx) => c.forEach((p) => { out[p] = idx; }));
  return out;
}

function chooseBestK(vectors, maxK) {
  const upper = Math.max(2, Math.min(maxK, Math.min(8, vectors.length)));
  let bestK = 2; let best = -Infinity;
  for (let k = 2; k <= upper; k += 1) {
    const a = kmeans(vectors, k, 6);
    let intra = 0; let inter = 0; let ic = 0; let ec = 0;
    for (let i = 0; i < vectors.length; i += 1) for (let j = i + 1; j < vectors.length; j += 1) {
      const s = cosine(vectors[i], vectors[j]);
      if (a[i] === a[j]) { intra += s; ic += 1; } else { inter += s; ec += 1; }
    }
    const score = intra / Math.max(1, ic) - inter / Math.max(1, ec);
    if (score > best) { best = score; bestK = k; }
  }
  return bestK;
}

function topTerms(clusterVectors, vocab) {
  const sums = new Array(vocab.length).fill(0);
  clusterVectors.forEach((v) => { for (let i = 0; i < v.length; i += 1) sums[i] += v[i]; });
  return sums.map((value, idx) => ({ term: vocab[idx], value })).sort((a, b) => b.value - a.value).slice(0, 4).map((t) => t.term.replace(/^(prev_|next_|domain_|bg_)/, ""));
}

function renderActivities(activities) {
  const ul = document.querySelector("#activityList");
  ul.innerHTML = "";
  activities.slice(0, 25).forEach((a) => {
    const li = document.createElement("li");
    li.textContent = `${new Date(a.timestamp).toLocaleTimeString()} — ${a.title}`;
    ul.appendChild(li);
  });
}

function renderTopicGroups(activities, labels) {
  const map = {};
  labels.forEach((l, i) => { if (!map[l]) map[l] = []; map[l].push(activities[i]); });
  const root = document.querySelector("#topicGroups");
  root.innerHTML = "";
  Object.entries(map).sort((a, b) => b[1].length - a[1].length).forEach(([label, list]) => {
    const card = document.createElement("article");
    card.className = "topic-card";
    card.innerHTML = `<h3>${label} (${list.length})</h3>`;
    const ul = document.createElement("ul");
    list.slice(0, 6).forEach((x) => { const li = document.createElement("li"); li.textContent = x.title; ul.appendChild(li); });
    card.appendChild(ul);
    root.appendChild(card);
  });
}

function renderClassificationTimeline(activities, labels, userLabels) {
  const root = document.querySelector("#classificationList");
  root.innerHTML = "";
  activities.slice(0, 20).forEach((a, i) => {
    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `<div class='class-title'>${new Date(a.timestamp).toLocaleTimeString()} — ${a.title}</div>`;
    const sel = document.createElement("select");
    sel.dataset.id = a.id;
    const current = userLabels[a.id] || labels[i];
    CLASS_LABELS.forEach((l) => {
      const o = document.createElement("option");
      o.value = l; o.textContent = l; if (l === current) o.selected = true;
      sel.appendChild(o);
    });
    row.appendChild(sel);
    root.appendChild(row);
  });
}

function renderClusters(vectors, vocab, km, ag, k) {
  const root = document.querySelector("#clusters");
  root.innerHTML = "";
  [{ name: "K-Means", a: km }, { name: "Agglomerative", a: ag }].forEach((m) => {
    const card = document.createElement("article");
    card.className = "cluster-card";
    card.innerHTML = `<h3>${m.name} (${k} clusters)</h3>`;
    const ul = document.createElement("ul");
    for (let c = 0; c < k; c += 1) {
      const idx = m.a.map((x, i) => (x === c ? i : -1)).filter((x) => x >= 0);
      if (!idx.length) continue;
      const li = document.createElement("li");
      li.textContent = `Cluster ${c + 1}: ${idx.length} items • ${topTerms(idx.map((i) => vectors[i]), vocab).join(", ")}`;
      ul.appendChild(li);
    }
    card.appendChild(ul);
    root.appendChild(card);
  });
}

function renderSimilarityGraph(activities, vectors, assignments, docs, labels) {
  const canvas = document.querySelector("#graph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const n = Math.min(18, activities.length);
  const nodes = activities.slice(0, n).map((a, i) => {
    const angle = (Math.PI * 2 * i) / Math.max(1, n);
    const radius = 86 + (i % 4) * 14;
    return { x: canvas.width / 2 + Math.cos(angle) * radius, y: canvas.height / 2 + Math.sin(angle) * radius, label: parseUrl(a.url).domain.slice(0, 18), c: assignments[i] || 0 };
  });

  for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) {
    const sim = semanticSimilarity(i, j, activities, vectors, docs, labels);
    if (sim < 0.3) continue;
    ctx.strokeStyle = `rgba(80,130,190,${Math.min(0.8, sim)})`;
    ctx.lineWidth = 0.6 + sim * 2;
    ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); ctx.stroke();
  }

  const palette = ["#7db6ff", "#9fd0ff", "#b2e2ff", "#8ec5ff", "#78b0f0", "#9bc8ea", "#a8d5ff"];
  nodes.forEach((node) => {
    ctx.fillStyle = palette[node.c % palette.length];
    ctx.beginPath(); ctx.arc(node.x, node.y, 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#1b486f"; ctx.font = "10px sans-serif"; ctx.fillText(node.label, node.x + 8, node.y + 3);
  });
}

function renderSiteGraph(activities) {
  const canvas = document.querySelector("#siteGraph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const counts = {};
  activities.forEach((a) => { const d = parseUrl(a.url).domain; counts[d] = (counts[d] || 0) + 1; });
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = Math.max(1, ...rows.map((r) => r[1]));
  rows.forEach(([d, v], idx) => {
    const x = 26 + idx * 66; const h = (v / max) * 140; const y = canvas.height - 38 - h;
    ctx.fillStyle = "#97c8ff"; ctx.fillRect(x, y, 52, h);
    ctx.fillStyle = "#245784"; ctx.font = "10px sans-serif"; ctx.fillText(String(v), x + 18, y - 4); ctx.fillText(d.slice(0, 9), x, canvas.height - 18);
  });
}

function renderCorrelationMap(activities, labels) {
  const canvas = document.querySelector("#correlationMap");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const domains = [...new Set(activities.map((a) => parseUrl(a.url).domain))].slice(0, 7);
  const classes = CLASS_LABELS.slice(0, 6);
  const matrix = classes.map(() => domains.map(() => 0));
  activities.forEach((a, i) => {
    const di = domains.indexOf(parseUrl(a.url).domain); const ci = classes.indexOf(labels[i]);
    if (di >= 0 && ci >= 0) matrix[ci][di] += 1;
  });
  const max = Math.max(1, ...matrix.flat());
  classes.forEach((cl, r) => {
    ctx.fillStyle = "#2a5b87"; ctx.font = "11px sans-serif"; ctx.fillText(cl.slice(0, 14), 8, 34 + r * 26);
    domains.forEach((d, c) => {
      const v = matrix[r][c];
      ctx.fillStyle = `rgba(102,170,238,${Math.max(0.08, v / max)})`;
      ctx.fillRect(130 + c * 70, 16 + r * 26, 66, 22);
      ctx.fillStyle = "#173f63"; ctx.font = "10px sans-serif"; ctx.fillText(String(v), 157 + c * 70, 33 + r * 26);
    });
  });
  domains.forEach((d, c) => { ctx.fillStyle = "#2a5b87"; ctx.font = "10px sans-serif"; ctx.fillText(d.slice(0, 9), 132 + c * 70, 12); });
}

async function run() {
  const { activities = [], userLabels = {} } = await chrome.storage.local.get(["activities", "userLabels"]);
  const windowSize = Number(document.querySelector("#timeWindow")?.value || 60);
  const items = activities.slice(0, windowSize);

  renderActivities(items);
  if (items.length < 3) {
    ["#topicGroups", "#clusters", "#classificationList"].forEach((id) => {
      const el = document.querySelector(id);
      if (el) el.innerHTML = "<p class='muted'>Need more browsing activity to analyze.</p>";
    });
    return;
  }

  const { docs, vectors, vocab } = buildTfIdfVectors(items);
  const labels = classifyActivities(items, docs, userLabels);
  const k = chooseBestK(vectors, Number(document.querySelector("#clusterCount")?.value || 5));
  const km = kmeans(vectors, k);
  const ag = agglomerative(vectors, k);

  renderTopicGroups(items, labels);
  renderClassificationTimeline(items, labels, userLabels);
  renderClusters(vectors, vocab, km, ag, k);
  renderSimilarityGraph(items, vectors, km, docs, labels);
  renderSiteGraph(items);
  renderCorrelationMap(items, labels);
}

document.querySelector("#runCluster")?.addEventListener("click", run);
document.querySelector("#timeWindow")?.addEventListener("change", run);
document.querySelector("#saveCorrections")?.addEventListener("click", async () => {
  const { userLabels = {} } = await chrome.storage.local.get("userLabels");
  document.querySelectorAll("#classificationList select").forEach((el) => { userLabels[el.dataset.id] = el.value; });
  await chrome.storage.local.set({ userLabels });
  await run();
});
document.querySelector("#clearData")?.addEventListener("click", async () => {
  await chrome.storage.local.set({ activities: [], userLabels: {} });
  await run();
});

run();
