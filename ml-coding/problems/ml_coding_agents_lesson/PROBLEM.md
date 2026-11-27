# The control loop around an agent

An LLM agent is a loop: the model proposes an action (call a tool with arguments, or answer), the program executes it, and the result goes back into the context. The model is the least predictable part, so the code around it does the engineering: it validates, bounds and records everything. This track is that code.

**Tools and validation.** Each tool has a name, a description and a JSON-schema-like signature. Before running a call, validate the arguments: required fields present, types right, values in range, no unknown fields. Return a structured error the model can read and fix, instead of raising. A **registry** maps names to tools and rejects duplicates and unknown names.

**Plans.** A **planner-executor** splits the work: the planner produces steps with dependencies, the executor runs them in an order that respects the dependencies (a topological order), passes outputs along, and stops or re-plans on failure. Detect cycles before running anything.

**Retries and idempotency.** Networks and tools fail. Retry transient errors with exponential backoff and a cap, but never retry a non-idempotent action (sending an email, charging a card) without an **idempotency key**: the tool remembers which keys it has already executed and returns the stored result for a repeat, so a retry can't do the action twice.

**Memory.** The context window is finite. A **sliding window** keeps the most recent messages that fit a token budget, always keeping the system prompt and, often, a running summary of what was dropped.

**Budgets.** Every loop needs limits: a maximum number of steps, of tool calls, of tokens or of wall time, and a clear final state when one is hit. An agent that cannot stop is a bug, not a feature.

> [!NOTE]
> These exercises are deterministic: the "model" is a scripted list of actions, so you can test the loop's behaviour exactly. That is also how production agent loops are unit-tested.
