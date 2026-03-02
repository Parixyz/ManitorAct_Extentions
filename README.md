# Activity Intelligence (Chrome Extension)

A Chrome extension that captures browsing activity and automatically analyzes it with NLP.

## What it now does
- Tracks active tab events and stores local activity data.
- Automatically infers **NLP topics** (e.g., Entertainment, Learning, Work, Social, Shopping, News).
- Runs **multiple clustering methods** automatically:
  - K-Means
  - Agglomerative clustering
- Auto-selects cluster count using an internal separation score.
- Uses context-aware similarity (title/path/domain tokens, neighboring-page context, topic + time boosts).
- Renders multiple visualizations:
  - Similarity network graph
  - Individual website activity graph
  - Correlation map (topic vs website)

## Install (unpacked)
1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click **Load unpacked**.
4. Select this folder.

## Data
All activity data stays in `chrome.storage.local` and can be removed from the popup using **Clear data**.
