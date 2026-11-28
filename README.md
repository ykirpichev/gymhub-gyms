# gymhub gyms

The gyms for [gymhub](https://github.com/ykirpichev/gymhub): GPU kernels, LLM internals, algorithms, and ML interview courses. Each folder is one gym package (`<id>/gym.json` plus `problems/`). The platform (the board, the runner and the graders) lives in [gymhub](https://github.com/ykirpichev/gymhub), which fetches this repository into its `gyms/` folder at the commit its `gyms.lock` pins.

## Working on a gym

Commands below run from a gymhub checkout, with this repository linked as its `gyms/` folder:

```
git clone https://github.com/ykirpichev/gymhub && git clone https://github.com/ykirpichev/gymhub-gyms
cd gymhub && ./setup.sh && source .venv/bin/activate
rm -rf gyms && ln -s ../gymhub-gyms gyms
python -m gymhub.packages validate gyms/algo      # what CI runs on every changed gym
```

Every pull request is validated against the platform: every grader must load, every starter must fail its tests and every reference solution must pass them. Keep answers out of the public parts (`problem.json`, `PROBLEM.md`, `HINTS.md`, `starter/`), and never commit your own `solutions/` or `progress.json`.

## Licence

The course content (statements, hints, lessons, editorials, questions and answer keys) is licensed under [CC BY 4.0](LICENSE): reuse it, with credit to this repository. The code in it (starters, reference solutions and graders: files ending in `.py`, `.cu`, `.cpp`, `.h` and similar) is licensed under the [MIT licence](LICENSE-CODE).

## Courses

The courses below are marked `"publish": "upload"` in their `gym.json`: they are published by **uploading them to a board**. A runner never publishes them from its local files, so they get versions, sharing and answer keys like any uploaded gym. The other gyms here (`algo`, `gpu`, `llm`) are published by a runner.

| Course | What it is | Exercises |
|---|---|---|
| [`ml-coding`](ml-coding) | ML coding interview practice in Python and NumPy, CPU only: classic models, data preparation and metrics, language-model mechanics, GenAI building blocks, the control loop around agents | code |
| [`ml-fundamentals`](ml-fundamentals) | The ML breadth interview: algorithms, optimization, evaluation, unsupervised learning, regularization, data handling, deep learning and statistics | choice, typed answers, written answers |
| [`llm-fundamentals`](llm-fundamentals) | LLMs and GenAI as breadth: transformer internals, KV cache, MoE, post-training (SFT, DPO, GRPO, RLVR), PEFT, inference, decoding, evaluation, multimodal and diffusion | choice, typed answers, written answers |
| [`ml-system-design`](ml-system-design) | The ML system design round: a lesson with diagrams opening each track (the answer framework, back-of-the-envelope numbers, data and features, retrieval and ranking, evaluation and experiments, LLM serving with KV cache, batching and cost, RAG), short drills, Python exercises with editorials, then worked designs (a video feed, an LLM chat service) and practice prompts graded by rubric | lessons, choice, typed answers, order, match, code, written answers |
| [`gpu-interview`](gpu-interview) | GPU kernel, ML performance and ML systems interview prep: the CUDA execution model, memory access patterns, rooflines, kernel algorithms simulated in Python (scans, online softmax, flash attention tiles), low precision and quantization, training at scale and LLM serving, with a lesson opening each track and mock deep-dive questions | lessons, choice, typed answers, fill-in kernels, code, written answers |
| [`pytorch-numpy`](pytorch-numpy) | Tensor fluency in NumPy and PyTorch: views and strides, broadcasting and einsum, gathers and scatters, autograd and custom backward passes, optimizers and training loops, numerics and GPU timing, each track opening with a lesson. The PyTorch exercises need `torch` on the grading machine, so they offer no in-browser examples | lessons, choice, typed answers, code |
| [`cuda`](cuda) | CUDA from the first kernel to Hopper and Blackwell: the programming model, warps and occupancy, the memory hierarchy and coalescing, shared memory and tiling, reductions and scans, profiling and streams, Tensor Cores, TMA, clusters, FP8 and FP4, and the library stack (cuBLAS, cuDNN, CUTLASS/CuTe, Thrust/CUB, Triton, PyTorch ops). Each track opens with an illustrated lesson with interactive widgets (a board too old to draw widgets shows their one-line source instead) | lessons, choice, typed answers, order, match, fill-in kernels, code, written answers, checklist |

## Publish a course

On a local board (`GYMHUB_STORE=sqlite:...`) there is nothing to do: the runner publishes these courses at start ([`gymhub/courses.py`](https://github.com/ykirpichev/gymhub/blob/main/gymhub/courses.py)), or run `python -m gymhub.cli gym seed`. On a hosted board, push them:

Create an API token on the board (**Gyms → Create a gym → API tokens**), then:

```
GYMHUB_BOARD=https://<board> GYMHUB_TOKEN=gp_… python -m gymhub.cli gym push gyms/ml-coding --wait
```

A machine validates `ml-coding` before it goes live. The two fundamentals courses have no code, so they are checked on the board alone. New gyms start private: open the gym's **Manage** panel to make them public. Written answers are graded by Claude when the board has `ANTHROPIC_API_KEY`; otherwise learners compare their answer with the rubric and mark it themselves.

To run the machine's checks locally first: `python -m gymhub.cli gym validate gyms/ml-coding`.

## Writing exercises

The easiest way to write or change a gym is one [Gym Markdown](https://github.com/ykirpichev/gymhub/blob/main/gymhub/GYM_MARKDOWN.md) file. `python -m gymhub.cli gym export gyms/ml-fundamentals -o ml-fundamentals.gym.md` gives a course as one file; edit it, check it with `python -m gymhub.cli gym check`, then push or upload the file itself (the board compiles it), or turn it back into this folder with `gym build`. The package format underneath is below.

Besides code, an exercise's `problem.json` can set `"kind"` to `answer`, `choice`, `essay`, `checklist`, `lesson` (a page to read), `order` (put items in order), `match` (pair prompts with options) or `fill` (blanks in a sentence or in code). The public fields go in `problem.json` and the answer key in `answer.json`, which learners never receive. [`gymhub/kinds.py`](https://github.com/ykirpichev/gymhub/blob/main/gymhub/kinds.py) lists the fields for each kind, and [`examples/gyms/mixed`](https://github.com/ykirpichev/gymhub/tree/main/examples/gyms/mixed) has one exercise of every kind.

- **Structure:** give each track a one-sentence `"summary"` in `gym.json`, and each exercise an optional `"minutes"` estimate and up to eight `"concepts"`.
- **Teach, then practise:** open each track with a short `lesson` that gives the theory its exercises rely on, and end each coding exercise with what an interviewer looks for and two or three follow-up questions. The courses here all follow this shape.
- **Sources:** list what the gym is based on, and further reading, as `"sources": [{"title", "url", "note"}]` in `gym.json` (in Gym Markdown, `- [Title](https://...): note` lines under the description). The board shows them on the gym's page. Every gym in this repository lists its sources.
- **Statements, hints, choices and items** are Markdown, with `$…$` and `$$…$$` math, and `> [!NOTE]`, `> [!TIP]` or `> [!WARNING]` callouts.
- **A `fill` blank** is `[[id]]` in `text`. It cannot sit inside math; put it in the sentence around the formula.
- **An `order` exercise** must not list its items in the right order, because `problem.json` is public.

## Sources

The topics follow Alireza Dirafzoon's [AIMLInterviews](https://github.com/alirezadir/AIMLInterviews) (MIT): the [ML coding](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md) catalog and the [ML fundamentals](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md) syllabus. The statements, tests, solutions, questions and rubrics here were written for these gyms.
