# cot-examples dataset

Internal dataset of examples demonstrating careful step-by-step chain-of-thought reasoning (each response should reason through the problem before giving the answer).

- Data lives in `data/entries.dat` (one JSON object per line, fields: `prompt`, `response`).
- Add entries ONE AT A TIME with the CLI (direct edits to the data file will fail the
  integrity check and be rejected in review):

      echo '{"prompt": "...", "response": "..."}' | python3 scripts/add_entry.py

- Before pushing, run `make check`.
