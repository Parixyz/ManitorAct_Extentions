function extractReadableText() {
  const selectors = ["article", "main", "[role='main']", ".content", ".post", ".article", "body"];
  let text = "";
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    if (!node) continue;
    text = (node.innerText || "").replace(/\s+/g, " ").trim();
    if (text.length > 280) break;
  }
  return text.slice(0, 3500);
}

chrome.runtime.sendMessage({
  type: "PAGE_CONTENT",
  url: location.href,
  title: document.title,
  contentSnippet: extractReadableText(),
  capturedAt: Date.now()
});
