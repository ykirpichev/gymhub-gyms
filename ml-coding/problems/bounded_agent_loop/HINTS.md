## Hint 1
Count model responses, not tool calls. Check the budget and the end of the script before you consume another response, so a final answer that lands on the last allowed step is still `"final"`.

## Hint 2
Keep the previous `(tool, args)` and a streak. Compare args with `==` before you decide to call. When the streak reaches `loop_limit`, return `"loop_detected"` with the model entry already on the trace and no observation for that proposal.

## Hint 3
An unknown name and a raised exception are both observations, then the loop continues. Only `"final"`, `"step_budget"`, `"loop_detected"` and `"model_exhausted"` stop it. Leave `answer` as `None` for every stop except `"final"`.
