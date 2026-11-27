# CLAUDE.md

This repository holds only gym content. Each top-level folder is a gym package; the format, the grader API, the CLI and the `write-gym` skill live in the platform repository, ykirpichev/gymhub (see its `CLAUDE.md` and `gymhub/GYM_MARKDOWN.md`).

To check a gym, use a gymhub checkout with this repository linked as its `gyms/` folder, then run `python -m gymhub.packages validate gyms/<id>` from the gymhub checkout. CI does the same for every changed gym.

Content is CC BY 4.0 (`LICENSE`), code in it is MIT (`LICENSE-CODE`). Never commit `solutions/` or `progress.json`.
