# Activity Cluster Graph (Chrome Extension)

This extension tracks your active tabs, lists browsing activity, applies lightweight NLP clustering, and visualizes similarity as a graph.

## Features
- Captures tab activity (title, URL, timestamp).
- Lists recent activity in the popup.
- Uses tokenization + TF-IDF vectors for NLP features.
- Runs k-means clustering in-browser.
- Draws an activity similarity graph on a canvas.

## Load in Chrome
1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select this folder.

## Notes
- Data is stored in `chrome.storage.local`.
- Use the **Clear** button to erase stored activity.
