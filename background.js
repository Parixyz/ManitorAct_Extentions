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
  if (!tab?.url || tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://")) return;

  const entry = {
    id: crypto.randomUUID(),
    tabId: tab.id,
    title: tab.title || "Untitled",
    url: tab.url,
    normalizedUrl: normalizeUrl(tab.url),
    contentSnippet: "",
    timestamp: Date.now()
  };

  const { activities = [] } = await chrome.storage.local.get("activities");
  activities.unshift(entry);
  if (activities.length > MAX_EVENTS) activities.length = MAX_EVENTS;
  await chrome.storage.local.set({ activities });
}

async function attachContentToRecentActivity(payload, senderTabId) {
  const { activities = [] } = await chrome.storage.local.get("activities");
  const idx = activities.findIndex((a) => a.tabId === senderTabId || a.normalizedUrl === normalizeUrl(payload.url));
  if (idx === -1) return;
  activities[idx] = {
    ...activities[idx],
    title: payload.title || activities[idx].title,
    contentSnippet: payload.contentSnippet || activities[idx].contentSnippet
  };
  await chrome.storage.local.set({ activities });
}

chrome.runtime.onMessage.addListener((message, sender) => {
  if (message?.type === "PAGE_CONTENT") {
    attachContentToRecentActivity(message, sender.tab?.id);
  }
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId);
  await storeActivity(tab);
});

chrome.tabs.onUpdated.addListener(async (_tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") await storeActivity(tab);
});
