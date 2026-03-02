const STOPWORDS = new Set([
  "the", "and", "for", "with", "www", "http", "https", "com", "org", "net", "to", "of", "in", "on", "a", "an", "is"
]);

function textFromActivity(item) {
  return `${item.title} ${item.normalizedUrl}`.toLowerCase();
}

function tokenize(text) {
  return text
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !STOPWORDS.has(token));
}

function buildTfIdfVectors(activities) {
  const docs = activities.map((activity) => tokenize(textFromActivity(activity)));
  const vocab = [...new Set(docs.flat())];
  const termToIdx = Object.fromEntries(vocab.map((term, idx) => [term, idx]));

  const df = new Array(vocab.length).fill(0);
  docs.forEach((doc) => {
    const seen = new Set(doc);
    seen.forEach((term) => {
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

  return { vectors, vocab };
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
      for (let d = 0; d < vectors[i].length; d += 1) {
        sums[cluster][d] += vectors[i][d];
      }
    }

    for (let j = 0; j < k; j += 1) {
      if (!counts[j]) continue;
      for (let d = 0; d < sums[j].length; d += 1) {
        centroids[j][d] = sums[j][d] / counts[j];
      }
    }
  }

  return assignments;
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
    .map((t) => t.term);
}

function renderActivities(activities) {
  const list = document.querySelector("#activityList");
  list.innerHTML = "";
  activities.slice(0, 20).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${new Date(item.timestamp).toLocaleTimeString()} — ${item.title}`;
    list.appendChild(li);
  });
}

function renderClusters(activities, vectors, vocab, assignments, k) {
  const parent = document.querySelector("#clusters");
  parent.innerHTML = "";

  for (let cluster = 0; cluster < k; cluster += 1) {
    const indices = assignments
      .map((assigned, idx) => (assigned === cluster ? idx : -1))
      .filter((idx) => idx !== -1);

    if (!indices.length) continue;

    const card = document.createElement("article");
    card.className = "cluster-card";

    const heading = document.createElement("h3");
    const clusterTerms = topTerms(indices.map((idx) => vectors[idx]), vocab);
    heading.textContent = `Cluster ${cluster + 1} (${indices.length}) • ${clusterTerms.join(", ")}`;
    card.appendChild(heading);

    const ul = document.createElement("ul");
    indices.slice(0, 6).forEach((idx) => {
      const li = document.createElement("li");
      li.textContent = activities[idx].title;
      ul.appendChild(li);
    });
    card.appendChild(ul);
    parent.appendChild(card);
  }
}

function renderGraph(activities, vectors, assignments) {
  const canvas = document.querySelector("#graph");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const nodes = activities.slice(0, 16).map((activity, i) => {
    const angle = (Math.PI * 2 * i) / Math.max(activities.length, 1);
    const radius = 90 + (i % 3) * 18;
    return {
      label: activity.title.slice(0, 22),
      x: canvas.width / 2 + Math.cos(angle) * radius,
      y: canvas.height / 2 + Math.sin(angle) * radius,
      cluster: assignments[i] ?? 0
    };
  });

  const palette = ["#2b8a3e", "#1c7ed6", "#a61e4d", "#e67700", "#5f3dc4", "#087f5b", "#c2255c", "#495057"];

  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const sim = cosine(vectors[i], vectors[j]);
      if (sim < 0.22) continue;
      ctx.strokeStyle = `rgba(70,70,70,${Math.min(sim, 0.6)})`;
      ctx.lineWidth = 1 + sim;
      ctx.beginPath();
      ctx.moveTo(nodes[i].x, nodes[i].y);
      ctx.lineTo(nodes[j].x, nodes[j].y);
      ctx.stroke();
    }
  }

  nodes.forEach((node) => {
    const color = palette[node.cluster % palette.length];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#111";
    ctx.font = "10px sans-serif";
    ctx.fillText(node.label, node.x + 10, node.y + 3);
  });
}

async function run() {
  const { activities = [] } = await chrome.storage.local.get("activities");
  const items = activities.slice(0, 60);
  renderActivities(items);

  if (items.length < 2) {
    document.querySelector("#clusters").innerHTML = "<p>Need more browsing activity to cluster.</p>";
    return;
  }

  const { vectors, vocab } = buildTfIdfVectors(items);
  const requestedK = Number(document.querySelector("#clusterCount").value) || 3;
  const k = Math.max(2, Math.min(requestedK, Math.min(8, items.length)));
  const assignments = kmeans(vectors, k);

  renderClusters(items, vectors, vocab, assignments, k);
  renderGraph(items, vectors, assignments);
}

document.querySelector("#runCluster").addEventListener("click", run);
document.querySelector("#clearData").addEventListener("click", async () => {
  await chrome.storage.local.set({ activities: [] });
  await run();
});

run();
