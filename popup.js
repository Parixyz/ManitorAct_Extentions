const STOPWORDS = new Set([
  "the", "and", "for", "with", "www", "http", "https", "com", "org", "net", "to", "of", "in", "on", "a", "an", "is", "home", "page"
]);

const CLASS_LABELS = [
  "Work",
  "Personal Project",
  "Research",
  "Teaching Assistant",
  "Course Work",
  "Entertainment",
  "Personal"
];

const TOPIC_LEXICON = {
  Entertainment: ["youtube", "netflix", "movie", "music", "spotify", "video", "anime", "game", "twitch", "disney"],
  Research: ["uav", "drone", "mobile", "paper", "arxiv", "experiment", "dataset", "survey", "benchmark", "lab"],
  "Teaching Assistant": ["1p02", "ta", "grading", "office", "hours", "rubric", "tutorial", "student", "moodle"],
  "Course Work": ["assignment", "lecture", "quiz", "course", "syllabus", "midterm", "final", "lab", "class"],
  Work: ["jira", "slack", "teams", "notion", "confluence", "meeting", "calendar", "drive", "mail"],
  "Personal Project": ["github", "repo", "prototype", "build", "deploy", "feature", "debug"],
  Personal: ["shopping", "amazon", "cart", "reddit", "instagram", "facebook"]
};

function parseUrl(url = "") {
  try {
    const parsed = new URL(url);
    return { domain: parsed.hostname.replace(/^www\./, ""), path: parsed.pathname.replace(/^\//, "") || "root" };
  } catch {
    return { domain: "unknown", path: "root" };
  }
}

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((t) => t.length > 2 && !STOPWORDS.has(t));
}

function domainTokens(domain) {
  return domain.split(/[.-]/).filter((token) => token.length > 2);
}

function buildContextDocs(activities) {
  const docs = activities.map((activity) => {
    const parsed = parseUrl(activity.url);
    const titleTokens = tokenize(activity.title || "");
    const pathTokens = tokenize(parsed.path.replace(/[/-]/g, " "));
    const baseTokens = [...titleTokens, ...pathTokens, ...domainTokens(parsed.domain), `domain_${parsed.domain}`];

    const bigrams = [];
    for (let i = 0; i < Math.min(titleTokens.length - 1, 10); i += 1) {
      bigrams.push(`bg_${titleTokens[i]}_${titleTokens[i + 1]}`);
    }

    return { tokens: [...baseTokens, ...bigrams], domain: parsed.domain };
  });

  return docs.map((doc, idx) => {
    const prev = docs[idx + 1];
    const next = docs[idx - 1];
    const out = [...doc.tokens];
    if (prev) {
      out.push(`prev_domain_${prev.domain}`);
      out.push(`transition_${prev.domain}_to_${doc.domain}`);
      out.push(...prev.tokens.slice(0, 5).map((x) => `prev_${x}`));
    }
    if (next) {
      out.push(`next_domain_${next.domain}`);
      out.push(...next.tokens.slice(0, 5).map((x) => `next_${x}`));
    }
    return out;
  });
}

function buildTfIdfVectors(activities) {
  const docs = buildContextDocs(activities);
  const vocab = [...new Set(docs.flat())];
  const termToIdx = Object.fromEntries(vocab.map((term, idx) => [term, idx]));
  const df = new Array(vocab.length).fill(0);

  docs.forEach((doc) => new Set(doc).forEach((term) => { df[termToIdx[term]] += 1; }));

  const vectors = docs.map((doc) => {
    const counts = new Array(vocab.length).fill(0);
    doc.forEach((term) => { counts[termToIdx[term]] += 1; });
    const total = Math.max(doc.length, 1);
    return counts.map((count, idx) => (count / total) * (Math.log((docs.length + 1) / (df[idx] + 1)) + 1));
  });

  return { vectors, vocab, docs };
}

function cosine(a, b) {
  let dot = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return !na || !nb ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function heuristicLabel(activity, tokens) {
  const text = `${activity.title || ""} ${activity.url || ""}`.toLowerCase();
  if (text.includes("1p02") || tokens.includes("1p02")) return "Teaching Assistant";
  if (tokens.some((t) => ["youtube", "watch", "video", "music"].includes(t))) return "Entertainment";

  const hasResearch = tokens.some((t) => ["uav", "drone", "mobile", "paper", "arxiv", "dataset", "experiment"].includes(t));
  const hasCourse = tokens.some((t) => ["assignment", "lecture", "quiz", "course", "midterm", "final"].includes(t));
  if (hasResearch && hasCourse) return "Course Work";
  if (hasResearch) return "Research";
  if (hasCourse) return "Course Work";
  if (tokens.some((t) => TOPIC_LEXICON.Work.includes(t))) return "Work";
  if (tokens.some((t) => TOPIC_LEXICON["Personal Project"].includes(t))) return "Personal Project";
  return "Personal";
}

function trainNaiveBayes(labeledDocs) {
  const model = { classCounts: {}, tokenCounts: {}, totalTokens: {}, vocabulary: new Set() };
  CLASS_LABELS.forEach((c) => {
    model.classCounts[c] = 0; model.tokenCounts[c] = {}; model.totalTokens[c] = 0;
  });

  labeledDocs.forEach(({ label, tokens }) => {
    model.classCounts[label] += 1;
    tokens.forEach((token) => {
      model.vocabulary.add(token);
      model.tokenCounts[label][token] = (model.tokenCounts[label][token] || 0) + 1;
      model.totalTokens[label] += 1;
    });
  });

  return model;
}

function predictNaiveBayes(model, tokens) {
  const vocabSize = Math.max(model.vocabulary.size, 1);
  let bestLabel = "Personal";
  let bestScore = -Infinity;

  CLASS_LABELS.forEach((label) => {
    const classPrior = (model.classCounts[label] + 1) / (Object.values(model.classCounts).reduce((a, b) => a + b, 0) + CLASS_LABELS.length);
    let score = Math.log(classPrior);

    tokens.forEach((token) => {
      const num = (model.tokenCounts[label][token] || 0) + 1;
      const den = model.totalTokens[label] + vocabSize;
      score += Math.log(num / den);
    });

    if (score > bestScore) {
      bestScore = score;
      bestLabel = label;
    }
  });

  return { label: bestLabel, score: bestScore };
}

function knnLabel(doc, labeledDocs, k = 5) {
  if (!labeledDocs.length) return null;
  const setDoc = new Set(doc);
  const scored = labeledDocs.map((row) => {
    const setRow = new Set(row.tokens);
    let overlap = 0;
    setDoc.forEach((t) => { if (setRow.has(t)) overlap += 1; });
    const sim = overlap / Math.sqrt(setDoc.size * setRow.size || 1);
    return { label: row.label, sim };
  }).sort((a, b) => b.sim - a.sim).slice(0, k);

  const votes = {};
  scored.forEach((s) => { votes[s.label] = (votes[s.label] || 0) + s.sim; });
  return Object.entries(votes).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
}

function classifyActivities(activities, docs, userLabels) {
  const supervised = [];
  activities.forEach((a, i) => {
    const manual = userLabels[a.id];
    if (manual) supervised.push({ label: manual, tokens: docs[i] });
  });

  activities.forEach((a, i) => {
    if (!userLabels[a.id]) supervised.push({ label: heuristicLabel(a, docs[i]), tokens: docs[i] });
  });

  const model = trainNaiveBayes(supervised);

  return activities.map((a, i) => {
    const manual = userLabels[a.id];
    if (manual) return manual;
    const nb = predictNaiveBayes(model, docs[i]).label;
    const knn = knnLabel(docs[i], supervised, 5);
    if (knn && knn === nb) return nb;
    return nb;
  });
}

function contextualSimilarity(i, j, activities, vectors, labels) {
  const lexical = cosine(vectors[i], vectors[j]);
  const domainBoost = parseUrl(activities[i].url).domain === parseUrl(activities[j].url).domain ? 0.2 : 0;
  const classBoost = labels[i] === labels[j] ? 0.2 : 0;
  const delta = Math.abs((activities[i].timestamp || 0) - (activities[j].timestamp || 0));
  const temporalBoost = Math.max(0, 1 - delta / (1000 * 60 * 25)) * 0.12;
  return Math.min(1, lexical * 0.62 + domainBoost + classBoost + temporalBoost);
}

function kmeans(vectors, k, maxIterations = 12) {
  const centroids = vectors.slice(0, k).map((v) => [...v]);
  const assignments = new Array(vectors.length).fill(0);
  for (let iter = 0; iter < maxIterations; iter += 1) {
    for (let i = 0; i < vectors.length; i += 1) {
      let bestIdx = 0; let bestScore = -1;
      for (let j = 0; j < centroids.length; j += 1) {
        const score = cosine(vectors[i], centroids[j]);
        if (score > bestScore) { bestScore = score; bestIdx = j; }
      }
      assignments[i] = bestIdx;
    }
    const sums = Array.from({ length: k }, () => new Array(vectors[0].length).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < vectors.length; i += 1) {
      const c = assignments[i]; counts[c] += 1;
      for (let d = 0; d < vectors[i].length; d += 1) sums[c][d] += vectors[i][d];
    }
    for (let j = 0; j < k; j += 1) if (counts[j]) for (let d = 0; d < sums[j].length; d += 1) centroids[j][d] = sums[j][d] / counts[j];
  }
  return assignments;
}

function agglomerative(vectors, k) {
  const clusters = vectors.map((_, idx) => [idx]);
  const avg = (a, b) => {
    let total = 0; let cnt = 0;
    a.forEach((i) => b.forEach((j) => { total += cosine(vectors[i], vectors[j]); cnt += 1; }));
    return cnt ? total / cnt : 0;
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
  let bestK = 2; let bestScore = -Infinity;
  for (let k = 2; k <= upper; k += 1) {
    const a = kmeans(vectors, k, 8);
    let intra = 0; let inter = 0; let ic = 0; let ec = 0;
    for (let i = 0; i < vectors.length; i += 1) for (let j = i + 1; j < vectors.length; j += 1) {
      const sim = cosine(vectors[i], vectors[j]);
      if (a[i] === a[j]) { intra += sim; ic += 1; } else { inter += sim; ec += 1; }
    }
    const score = intra / Math.max(ic, 1) - inter / Math.max(ec, 1);
    if (score > bestScore) { bestScore = score; bestK = k; }
  }
  return bestK;
}

function topTerms(clusterVectors, vocab, count = 5) {
  const sums = new Array(vocab.length).fill(0);
  clusterVectors.forEach((v) => { for (let i = 0; i < v.length; i += 1) sums[i] += v[i]; });
  return sums.map((value, idx) => ({ term: vocab[idx], value })).sort((a, b) => b.value - a.value).slice(0, count).map((t) => t.term.replace(/^(prev_|next_|domain_|bg_)/, ""));
}

function renderActivities(activities) {
  const list = document.querySelector("#activityList");
  list.innerHTML = "";
  activities.slice(0, 25).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${new Date(item.timestamp).toLocaleTimeString()} — ${item.title}`;
    list.appendChild(li);
  });
}

function renderTopicGroups(activities, labels) {
  const groups = {};
  labels.forEach((label, i) => {
    if (!groups[label]) groups[label] = [];
    groups[label].push(activities[i]);
  });
  const parent = document.querySelector("#topicGroups");
  parent.innerHTML = "";
  Object.entries(groups).sort((a, b) => b[1].length - a[1].length).forEach(([label, items]) => {
    const card = document.createElement("article");
    card.className = "topic-card";
    card.innerHTML = `<h3>${label} (${items.length})</h3>`;
    const ul = document.createElement("ul");
    items.slice(0, 6).forEach((x) => { const li = document.createElement("li"); li.textContent = x.title; ul.appendChild(li); });
    card.appendChild(ul);
    parent.appendChild(card);
  });
}

function renderClassificationTimeline(activities, labels, userLabels) {
  const parent = document.querySelector("#classificationList");
  parent.innerHTML = "";
  activities.slice(0, 20).forEach((activity, i) => {
    const row = document.createElement("div");
    row.className = "class-row";
    const current = userLabels[activity.id] || labels[i];
    row.innerHTML = `<div class="class-title">${new Date(activity.timestamp).toLocaleTimeString()} — ${activity.title}</div>`;
    const select = document.createElement("select");
    select.dataset.id = activity.id;
    CLASS_LABELS.forEach((label) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      if (label === current) option.selected = true;
      select.appendChild(option);
    });
    row.appendChild(select);
    parent.appendChild(row);
  });
}

function renderClusters(vectors, vocab, kmeansAssignments, agAssignments, k) {
  const parent = document.querySelector("#clusters");
  parent.innerHTML = "";
  [
    { name: "K-Means", assignments: kmeansAssignments },
    { name: "Agglomerative", assignments: agAssignments }
  ].forEach((method) => {
    const card = document.createElement("article");
    card.className = "cluster-card";
    const title = document.createElement("h3");
    title.textContent = `${method.name} (${k} clusters)`;
    card.appendChild(title);
    const ul = document.createElement("ul");
    for (let cluster = 0; cluster < k; cluster += 1) {
      const idxs = method.assignments.map((v, i) => (v === cluster ? i : -1)).filter((i) => i >= 0);
      if (!idxs.length) continue;
      const li = document.createElement("li");
      li.textContent = `Cluster ${cluster + 1}: ${idxs.length} items • ${topTerms(idxs.map((idx) => vectors[idx]), vocab, 4).join(", ")}`;
      ul.appendChild(li);
    }
    card.appendChild(ul);
    parent.appendChild(card);
  });
}

function renderSimilarityGraph(activities, vectors, assignments, labels) {
  const canvas = document.querySelector("#graph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const limit = Math.min(18, activities.length);
  const nodes = activities.slice(0, limit).map((activity, i) => {
    const angle = (Math.PI * 2 * i) / Math.max(limit, 1);
    const radius = 88 + (i % 4) * 16;
    return { label: parseUrl(activity.url).domain.slice(0, 18), x: canvas.width / 2 + Math.cos(angle) * radius, y: canvas.height / 2 + Math.sin(angle) * radius, cluster: assignments[i] ?? 0 };
  });

  const palette = ["#7db6ff", "#9fd0ff", "#b2e2ff", "#8ec5ff", "#78b0f0", "#9bc8ea", "#a8d5ff"];
  for (let i = 0; i < nodes.length; i += 1) for (let j = i + 1; j < nodes.length; j += 1) {
    const sim = contextualSimilarity(i, j, activities, vectors, labels);
    if (sim < 0.28) continue;
    ctx.strokeStyle = `rgba(80,130,190,${Math.min(0.76, sim)})`;
    ctx.lineWidth = 0.6 + sim * 2.2;
    ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); ctx.stroke();
  }
  nodes.forEach((n) => {
    ctx.fillStyle = palette[n.cluster % palette.length];
    ctx.beginPath(); ctx.arc(n.x, n.y, 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#1b486f"; ctx.font = "10px sans-serif"; ctx.fillText(n.label, n.x + 9, n.y + 4);
  });
}

function renderSiteGraph(activities) {
  const canvas = document.querySelector("#siteGraph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const domainCounts = {};
  activities.forEach((item) => { const d = parseUrl(item.url).domain; domainCounts[d] = (domainCounts[d] || 0) + 1; });
  const entries = Object.entries(domainCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const maxVal = Math.max(...entries.map((e) => e[1]), 1);
  entries.forEach(([domain, value], idx) => {
    const x = 26 + idx * 66; const h = (value / maxVal) * 140; const y = canvas.height - 38 - h;
    ctx.fillStyle = "#97c8ff"; ctx.fillRect(x, y, 52, h);
    ctx.fillStyle = "#245784"; ctx.font = "10px sans-serif";
    ctx.fillText(String(value), x + 18, y - 4); ctx.fillText(domain.slice(0, 9), x, canvas.height - 18);
  });
}

function renderCorrelationMap(activities, labels) {
  const canvas = document.querySelector("#correlationMap");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const domains = [...new Set(activities.map((x) => parseUrl(x.url).domain))].slice(0, 7);
  const classes = CLASS_LABELS.slice(0, 6);
  const matrix = classes.map(() => domains.map(() => 0));

  activities.forEach((item, i) => {
    const d = domains.indexOf(parseUrl(item.url).domain);
    const c = classes.indexOf(labels[i]);
    if (d >= 0 && c >= 0) matrix[c][d] += 1;
  });

  const maxV = Math.max(...matrix.flat(), 1);
  classes.forEach((klass, r) => {
    ctx.fillStyle = "#2a5b87"; ctx.font = "11px sans-serif"; ctx.fillText(klass.slice(0, 14), 8, 34 + r * 26);
    domains.forEach((domain, c) => {
      const val = matrix[r][c];
      ctx.fillStyle = `rgba(102,170,238,${Math.max(0.08, val / maxV)})`;
      ctx.fillRect(130 + c * 70, 16 + r * 26, 66, 22);
      ctx.fillStyle = "#173f63"; ctx.font = "10px sans-serif"; ctx.fillText(String(val), 157 + c * 70, 33 + r * 26);
    });
  });
  domains.forEach((domain, c) => { ctx.fillStyle = "#2a5b87"; ctx.font = "10px sans-serif"; ctx.fillText(domain.slice(0, 9), 132 + c * 70, 12); });
}

async function run() {
  const { activities = [], userLabels = {} } = await chrome.storage.local.get(["activities", "userLabels"]);
  const windowSize = Number(document.querySelector("#timeWindow").value) || 60;
  const items = activities.slice(0, windowSize);

  renderActivities(items);
  if (items.length < 3) {
    ["#topicGroups", "#clusters", "#classificationList"].forEach((selector) => {
      document.querySelector(selector).innerHTML = "<p class='muted'>Need more browsing activity to analyze.</p>";
    });
    return;
  }

  const { vectors, vocab, docs } = buildTfIdfVectors(items);
  const labels = classifyActivities(items, docs, userLabels);
  const autoK = chooseBestK(vectors, Number(document.querySelector("#clusterCount").value) || 5);
  const km = kmeans(vectors, autoK);
  const ag = agglomerative(vectors, autoK);

  renderTopicGroups(items, labels);
  renderClassificationTimeline(items, labels, userLabels);
  renderClusters(vectors, vocab, km, ag, autoK);
  renderSimilarityGraph(items, vectors, km, labels);
  renderSiteGraph(items);
  renderCorrelationMap(items, labels);
}

document.querySelector("#runCluster").addEventListener("click", run);
document.querySelector("#timeWindow").addEventListener("change", run);
document.querySelector("#saveCorrections").addEventListener("click", async () => {
  const { userLabels = {} } = await chrome.storage.local.get("userLabels");
  document.querySelectorAll("#classificationList select").forEach((el) => { userLabels[el.dataset.id] = el.value; });
  await chrome.storage.local.set({ userLabels });
  await run();
});
document.querySelector("#clearData").addEventListener("click", async () => {
  await chrome.storage.local.set({ activities: [], userLabels: {} });
  await run();
});

run();
