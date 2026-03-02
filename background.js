const MAX_EVENTS = 500;

function normalizeUrl(url = "") {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname}`;
  } catch {
    return url;
  }
}

async function storeActivity(tab) {
  if (!tab?.url || tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://")) {
    return;
  }

  const entry = {
    id: crypto.randomUUID(),
    title: tab.title || "Untitled",
    url: tab.url,
    normalizedUrl: normalizeUrl(tab.url),
    timestamp: Date.now()
  };

  const { activities = [] } = await chrome.storage.local.get("activities");
  activities.unshift(entry);
  if (activities.length > MAX_EVENTS) {
    activities.length = MAX_EVENTS;
  }
  await chrome.storage.local.set({ activities });
}

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId);
  await storeActivity(tab);
});

chrome.tabs.onUpdated.addListener(async (_tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    await storeActivity(tab);
  }
});
