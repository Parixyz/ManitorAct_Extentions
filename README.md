# Activity Intelligence (Chrome Extension)

A Chrome extension that captures browsing activity and analyzes it with adaptive NLP + semantic similarity.

## What it now does
- Captures tab activity in the background and stores local history.
- Extracts readable on-page text (`contentSnippet`) via a content script for stronger semantic analysis.
- Classifies activity into:
  - Work
  - Personal Project
  - Research
  - Teaching Assistant
  - Course Work
  - Entertainment
  - Personal
  - Other
- Includes explicit domain logic, including:
  - `1p02` => Teaching Assistant
  - UAV/mobile + course context => Course Work
  - UAV/mobile without course context => Research
- Uses a hybrid classifier:
  - heuristic weak labeling
  - multinomial Naive Bayes
  - kNN agreement layer
- Supports user correction and stores labels in `chrome.storage.local` so predictions improve over time.
- Builds similarity graph from semantic signals (content + class + lexical), and forces same normalized URL to max similarity.
- Runs K-Means + Agglomerative clustering with auto-k selection.

## Install (unpacked)
1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click **Load unpacked**.
4. Select this folder.

## Data
All activity/content snippets and user corrections remain local in `chrome.storage.local` and can be cleared with **Clear data**.
