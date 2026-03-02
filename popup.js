const STOPWORDS = new Set([
  "the", "and", "for", "with", "www", "http", "https", "com", "org", "net", "to", "of", "in", "on", "a", "an", "is", "home", "page"
]);

const TOPIC_LEXICON = {
  Entertainment: ["youtube", "netflix", "movie", "music", "spotify", "video", "anime", "game", "twitch", "disney"],
  Learning: ["course", "tutorial", "docs", "learn", "education", "university", "wiki", "guide", "research"],
  Work: ["jira", "slack", "teams", "notion", "confluence", "meeting", "calendar", "drive", "mail"],
  Social: ["facebook", "instagram", "reddit", "x", "twitter", "linkedin", "chat", "community"],
  Shopping: ["amazon", "cart", "shop", "deal", "price", "product", "store", "checkout"],
  News: ["news", "times", "post", "breaking", "politics", "finance", "weather"]
};

function parseUrl(url = "") {
  try {
    const parsed = new URL(url);
    return {
      domain: parsed.hostname.replace(/^www\./, ""),
      path: parsed.pathname.replace(/^\//, "") || "root"
    };
  } catch {
    return { domain: "unknown", path: "root" };
  }
}

function normalizeDomain(url = "") {
  return parseUrl(url).domain;
}

function tokenize(text) {
  return text
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !STOPWORDS.has(token));
}

function domainTokens(domain) {
  return domain.split(/[.-]/).filter((token) => token.length > 2);
}

function buildContextDocs(activities) {
  const docs = activities.map((activity) => {
    const parsed = parseUrl(activity.url);
    const titleTokens = tokenize(activity.title || "");
    const pathTokens = tokenize(parsed.path.replace(/[/-]/g, " "));
    const baseTokens = [
      ...titleTokens,
      ...pathTokens,
      ...domainTokens(parsed.domain),
      `domain_${parsed.domain}`
    ];

    const bigrams = [];
    for (let i = 0; i < Math.min(titleTokens.length - 1, 8); i += 1) {
      bigrams.push(`bg_${titleTokens[i]}_${titleTokens[i + 1]}`);
    }

    return {
      baseTokens: [...baseTokens, ...bigrams],
      domain: parsed.domain,
      timestamp: activity.timestamp
    };
  });

  return docs.map((doc, idx) => {
    const prev = docs[idx + 1];
    const next = docs[idx - 1];
    const context = [...doc.baseTokens];

    if (prev) {
      context.push(`prev_domain_${prev.domain}`);
      context.push(`transition_${prev.domain}_to_${doc.domain}`);
      context.push(...prev.baseTokens.slice(0, 4).map((token) => `prev_${token}`));
    }
    if (next) {
      context.push(`next_domain_${next.domain}`);
      context.push(...next.baseTokens.slice(0, 4).map((token) => `next_${token}`));
    }

    return context;
  });
}

function buildTfIdfVectors(activities) {
  const docs = buildContextDocs(activities);
  const vocab = [...new Set(docs.flat())];
  const termToIdx = Object.fromEntries(vocab.map((term, idx) => [term, idx]));
  const df = new Array(vocab.length).fill(0);

  docs.forEach((doc) => {
    new Set(doc).forEach((term) => {
      df[termToIdx[term]] += 1;
    });
  });

  const vectors = docs.map((doc) => {
    const counts = new Array(vocab.length).fill(0);
    doc.forEach((term) => {
      counts[termToIdx[term]] += 1;
    });
    const total = Math.max(doc.length, 1);
    return counts.map((count, idx) => {
      const tf = count / total;
      const idf = Math.log((docs.length + 1) / (df[idx] + 1)) + 1;
      return tf * idf;
    });
  });

  return { vectors, vocab, docs };
}

function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function inferTopic(tokens) {
  const counts = Object.fromEntries(Object.keys(TOPIC_LEXICON).map((topic) => [topic, 0]));
  tokens.forEach((token) => {
    Object.entries(TOPIC_LEXICON).forEach(([topic, words]) => {
      if (words.includes(token)) counts[topic] += 1;
    });
  });
  const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return best && best[1] > 0 ? best[0] : "General";
}

function contextualSimilarity(i, j, activities, vectors, docs) {
  const lexical = cosine(vectors[i], vectors[j]);
  const domainA = normalizeDomain(activities[i].url);
  const domainB = normalizeDomain(activities[j].url);
  const domainBoost = domainA === domainB ? 0.18 : 0;
  const topicBoost = inferTopic(docs[i]) === inferTopic(docs[j]) ? 0.15 : 0;
  const delta = Math.abs((activities[i].timestamp || 0) - (activities[j].timestamp || 0));
  const temporalBoost = Math.max(0, 1 - delta / (1000 * 60 * 25)) * 0.12;
  return Math.min(1, lexical * 0.7 + domainBoost + topicBoost + temporalBoost);
}

function kmeans(vectors, k, maxIterations = 12) {
  const centroids = vectors.slice(0, k).map((v) => [...v]);
  const assignments = new Array(vectors.length).fill(0);

  for (let iter = 0; iter < maxIterations; iter += 1) {
    for (let i = 0; i < vectors.length; i += 1) {
      let bestIdx = 0;
      let bestScore = -1;
      for (let j = 0; j < centroids.length; j += 1) {
        const score = cosine(vectors[i], centroids[j]);
        if (score > bestScore) {
          bestScore = score;
          bestIdx = j;
        }
      }
      assignments[i] = bestIdx;
    }

    const sums = Array.from({ length: k }, () => new Array(vectors[0].length).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < vectors.length; i += 1) {
      const cluster = assignments[i];
      counts[cluster] += 1;
      for (let d = 0; d < vectors[i].length; d += 1) sums[cluster][d] += vectors[i][d];
    }

    for (let j = 0; j < k; j += 1) {
      if (!counts[j]) continue;
      for (let d = 0; d < sums[j].length; d += 1) centroids[j][d] = sums[j][d] / counts[j];
    }
  }

  return assignments;
}

function agglomerative(vectors, k) {
  const clusters = vectors.map((_, idx) => [idx]);

  const avgLink = (a, b) => {
    let total = 0;
    let cnt = 0;
    a.forEach((i) => {
      b.forEach((j) => {
        total += cosine(vectors[i], vectors[j]);
        cnt += 1;
      });
    });
    return cnt ? total / cnt : 0;
  };

  while (clusters.length > k) {
    let best = { i: 0, j: 1, score: -1 };
    for (let i = 0; i < clusters.length; i += 1) {
      for (let j = i + 1; j < clusters.length; j += 1) {
        const score = avgLink(clusters[i], clusters[j]);
        if (score > best.score) best = { i, j, score };
      }
    }
    clusters[best.i] = [...clusters[best.i], ...clusters[best.j]];
    clusters.splice(best.j, 1);
  }

  const assignments = new Array(vectors.length).fill(0);
  clusters.forEach((cluster, idx) => cluster.forEach((point) => {
    assignments[point] = idx;
  }));
  return assignments;
}

function chooseBestK(vectors, maxK) {
  const upper = Math.max(2, Math.min(maxK, Math.min(8, vectors.length)));
  let bestK = 2;
  let bestScore = -Infinity;

  for (let k = 2; k <= upper; k += 1) {
    const assignments = kmeans(vectors, k, 8);
    let intra = 0;
    let inter = 0;
    let intraCount = 0;
    let interCount = 0;

    for (let i = 0; i < vectors.length; i += 1) {
      for (let j = i + 1; j < vectors.length; j += 1) {
        const sim = cosine(vectors[i], vectors[j]);
        if (assignments[i] === assignments[j]) {
          intra += sim;
          intraCount += 1;
        } else {
          inter += sim;
          interCount += 1;
        }
      }
    }

    const score = (intra / Math.max(intraCount, 1)) - (inter / Math.max(interCount, 1));
    if (score > bestScore) {
      bestScore = score;
      bestK = k;
    }
  }
  return bestK;
}

function topTerms(clusterVectors, vocab, count = 5) {
  const sums = new Array(vocab.length).fill(0);
  clusterVectors.forEach((vector) => {
    for (let i = 0; i < vector.length; i += 1) sums[i] += vector[i];
  });
  return sums
    .map((value, idx) => ({ term: vocab[idx], value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count)
    .map((t) => t.term.replace(/^(prev_|next_|domain_|bg_)/, ""));
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

function renderTopicGroups(activities, docs) {
  const groups = {};
  activities.forEach((activity, idx) => {
    const topic = inferTopic(docs[idx]);
    if (!groups[topic]) groups[topic] = [];
    groups[topic].push(activity);
  });

  const parent = document.querySelector("#topicGroups");
  parent.innerHTML = "";
  Object.entries(groups)
    .sort((a, b) => b[1].length - a[1].length)
    .forEach(([topic, items]) => {
      const card = document.createElement("article");
      card.className = "topic-card";
      card.innerHTML = `<h3>${topic} (${items.length})</h3>`;
      const ul = document.createElement("ul");
      items.slice(0, 5).forEach((x) => {
        const li = document.createElement("li");
        li.textContent = x.title;
        ul.appendChild(li);
      });
      card.appendChild(ul);
      parent.appendChild(card);
    });
}

function renderClusters(vectors, vocab, kmeansAssignments, agAssignments, k) {
  const parent = document.querySelector("#clusters");
  parent.innerHTML = "";

  const methods = [
    { name: "K-Means", assignments: kmeansAssignments },
    { name: "Agglomerative", assignments: agAssignments }
  ];

  methods.forEach((method) => {
    const card = document.createElement("article");
    card.className = "cluster-card";
    const title = document.createElement("h3");
    title.textContent = `${method.name} (${k} clusters)`;
    card.appendChild(title);

    const ul = document.createElement("ul");
    for (let cluster = 0; cluster < k; cluster += 1) {
      const idxs = method.assignments.map((v, i) => (v === cluster ? i : -1)).filter((i) => i >= 0);
      if (!idxs.length) continue;
      const terms = topTerms(idxs.map((idx) => vectors[idx]), vocab, 3).join(", ");
      const li = document.createElement("li");
      li.textContent = `Cluster ${cluster + 1}: ${idxs.length} items • ${terms}`;
      ul.appendChild(li);
    }
    card.appendChild(ul);
    parent.appendChild(card);
  });
}

function renderSimilarityGraph(activities, vectors, assignments, docs) {
  const canvas = document.querySelector("#graph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const limit = Math.min(18, activities.length);
  const nodes = activities.slice(0, limit).map((activity, i) => {
    const angle = (Math.PI * 2 * i) / Math.max(limit, 1);
    const wobble = 12 * Math.sin(i * 1.7);
    const radius = 88 + (i % 4) * 13 + wobble;
    return {
      label: normalizeDomain(activity.url).slice(0, 18),
      x: canvas.width / 2 + Math.cos(angle) * radius,
      y: canvas.height / 2 + Math.sin(angle) * radius,
      cluster: assignments[i] ?? 0
    };
  });

  const palette = ["#7db6ff", "#9fd0ff", "#b2e2ff", "#8ec5ff", "#78b0f0", "#9bc8ea", "#a8d5ff", "#84c0e8"];

  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const sim = contextualSimilarity(i, j, activities, vectors, docs);
      if (sim < 0.28) continue;
      ctx.strokeStyle = `rgba(80,130,190,${Math.min(0.74, sim)})`;
      ctx.lineWidth = 0.6 + sim * 2.2;
      ctx.beginPath();
      ctx.moveTo(nodes[i].x, nodes[i].y);
      ctx.lineTo(nodes[j].x, nodes[j].y);
      ctx.stroke();
    }
  }

  nodes.forEach((node) => {
    ctx.fillStyle = palette[node.cluster % palette.length];
    ctx.beginPath();
    ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#1b486f";
    ctx.font = "10px sans-serif";
    ctx.fillText(node.label, node.x + 9, node.y + 4);
  });
}

function renderSiteGraph(activities) {
  const canvas = document.querySelector("#siteGraph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const domainCounts = {};
  activities.forEach((item) => {
    const domain = normalizeDomain(item.url);
    domainCounts[domain] = (domainCounts[domain] || 0) + 1;
  });

  const entries = Object.entries(domainCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const maxVal = Math.max(...entries.map((e) => e[1]), 1);
  const barW = 52;
  const gap = 14;

  entries.forEach(([domain, value], idx) => {
    const x = 26 + idx * (barW + gap);
    const h = (value / maxVal) * 140;
    const y = canvas.height - 38 - h;
    ctx.fillStyle = "#97c8ff";
    ctx.fillRect(x, y, barW, h);
    ctx.fillStyle = "#245784";
    ctx.font = "10px sans-serif";
    ctx.fillText(String(value), x + 18, y - 4);
    ctx.fillText(domain.slice(0, 9), x, canvas.height - 18);
  });
}

function renderCorrelationMap(activities, docs) {
  const canvas = document.querySelector("#correlationMap");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const domains = [...new Set(activities.map((x) => normalizeDomain(x.url)))].slice(0, 7);
  const topics = Object.keys(TOPIC_LEXICON).slice(0, 6);

  const matrix = topics.map(() => domains.map(() => 0));
  activities.forEach((item, i) => {
    const domain = normalizeDomain(item.url);
    const dIdx = domains.indexOf(domain);
    if (dIdx === -1) return;
    const topic = inferTopic(docs[i]);
    const tIdx = topics.indexOf(topic);
    if (tIdx >= 0) matrix[tIdx][dIdx] += 1;
  });

  const maxV = Math.max(...matrix.flat(), 1);
  const cellW = 70;
  const cellH = 26;

  topics.forEach((topic, r) => {
    ctx.fillStyle = "#2a5b87";
    ctx.font = "11px sans-serif";
    ctx.fillText(topic.slice(0, 11), 8, 34 + r * cellH);
    domains.forEach((domain, c) => {
      const val = matrix[r][c];
      const alpha = val / maxV;
      ctx.fillStyle = `rgba(102,170,238,${Math.max(0.08, alpha)})`;
      ctx.fillRect(130 + c * cellW, 16 + r * cellH, cellW - 4, cellH - 4);
      ctx.fillStyle = "#173f63";
      ctx.font = "10px sans-serif";
      ctx.fillText(String(val), 157 + c * cellW, 33 + r * cellH);
    });
  });

  domains.forEach((domain, c) => {
    ctx.fillStyle = "#2a5b87";
    ctx.font = "10px sans-serif";
    ctx.fillText(domain.slice(0, 9), 132 + c * cellW, 12);
  });
}

async function run() {
  const { activities = [] } = await chrome.storage.local.get("activities");
  const windowSize = Number(document.querySelector("#timeWindow").value) || 60;
  const items = activities.slice(0, windowSize);

  renderActivities(items);
  if (items.length < 3) {
    ["#topicGroups", "#clusters"].forEach((selector) => {
      document.querySelector(selector).innerHTML = "<p class='muted'>Need more browsing activity to analyze.</p>";
    });
    return;
  }

  const { vectors, vocab, docs } = buildTfIdfVectors(items);
  const maxRequested = Number(document.querySelector("#clusterCount").value) || 5;
  const autoK = chooseBestK(vectors, maxRequested);

  const km = kmeans(vectors, autoK);
  const ag = agglomerative(vectors, autoK);

  renderTopicGroups(items, docs);
  renderClusters(vectors, vocab, km, ag, autoK);
  renderSimilarityGraph(items, vectors, km, docs);
  renderSiteGraph(items);
  renderCorrelationMap(items, docs);
}

document.querySelector("#runCluster").addEventListener("click", run);
document.querySelector("#timeWindow").addEventListener("change", run);
document.querySelector("#clearData").addEventListener("click", async () => {
  await chrome.storage.local.set({ activities: [] });
  await run();
});

run();
