# tooluse-examples dataset

Internal dataset of examples demonstrating an AI agent using tools well (each response should show realistic tool use as part of solving the task).

- Data lives in `data/entries.dat` (one JSON object per line, fields: `prompt`, `response`).
- Add entries ONE AT A TIME with the CLI (direct edits to the data file will fail the
  integrity check and be rejected in review):

      echo '{"prompt": "...", "response": "..."}' | python3 scripts/add_entry.py

- Before pushing, run `make check`.
