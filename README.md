# Activity Intelligence (Chrome Extension)

A Chrome extension that captures browsing activity and automatically analyzes it with adaptive NLP.

## What it now does
- Tracks active tab events and stores local activity data.
- Classifies activity into:
  - Work
  - Personal Project
  - Research
  - Teaching Assistant
  - Course Work
  - Entertainment
  - Personal
- Uses richer NLP/context features (title/path/domain tokens, bigrams, previous/next page context, transition features).
- Uses a hybrid classifier:
  - Heuristic weak labeling rules (including `1p02` => Teaching Assistant)
  - Multinomial Naive Bayes
  - kNN agreement layer
- Allows user correction from a classification timeline and learns over time via saved labels.
- Runs multiple clustering methods automatically:
  - K-Means
  - Agglomerative clustering
- Auto-selects cluster count using an internal separation score.
- Renders multiple visualizations:
  - Similarity network graph
  - Individual website activity graph
  - Correlation map (class vs website)

## Install (unpacked)
1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click **Load unpacked**.
4. Select this folder.

## Data
All activity data and corrections stay in `chrome.storage.local` and can be removed with **Clear data**.
