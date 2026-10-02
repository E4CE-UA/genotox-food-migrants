# src/legacy/ — not part of the notebook

These scripts belong to an earlier data-preparation pipeline and are **not
imported or executed by the notebook**. They are kept only so the provenance of
the files in `data/` can be traced. Several call the OpenAI API; where they do,
the key is read from the `OPENAI_API_KEY` environment variable and is never
stored in the repository. Nothing in the notebook's runtime path depends on an
LLM.
