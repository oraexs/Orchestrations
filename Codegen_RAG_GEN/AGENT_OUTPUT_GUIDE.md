# LangGraph Agent Output Extraction Guide

## How LangGraph Returns Data After `.invoke()`

When you call `agent.invoke({"messages": [...]})`, the result is always:

```python
{
    "messages": [
        HumanMessage(content="..."),          # your input
        AIMessage(content="...", tool_calls=[...]),  # LLM decision
        ToolMessage(content="..."),           # tool execution result
        AIMessage(content="final answer"),    # final LLM response
    ]
}
```

**The agent never returns custom keys** — everything comes back through `messages`.

---

## Part 1: Extracting Different Types of Data

### 1. Extract the Final Text Response

```python
result = agent.invoke({"messages": [("user", "What is 2+2?")]})

last_msg = result["messages"][-1]
answer = last_msg.content
print(answer)  # "The answer is 4."
```

---

### 2. Extract Tool Call Arguments (what we do in this project)

Use case: You want to know *what arguments* the LLM passed to a tool.

```python
result = agent.invoke({"messages": [("user", "Create a HAL workspace")]})

for msg in result["messages"]:
    tool_calls = getattr(msg, "tool_calls", None)  # only AIMessage has this
    if not tool_calls:
        continue
    for tc in tool_calls:
        # tc is a dict: {"name": "tool_name", "args": {...}, "id": "..."}
        if tc["name"] == "create_workspace":
            workspace = tc["args"]["structure"]
            print(workspace)
            # {'mainfolder': ['MC9S08SC4_HAL'], 'subfolders': {...}, 'files': {...}}
```

---

### 3. Extract Tool Execution Results (what the tool returned)

Use case: You want the *output* of a tool, not the input args.

```python
from langchain_core.messages import ToolMessage

result = agent.invoke({"messages": [("user", "Search for ADC registers")]})

for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        print(f"Tool result: {msg.content}")
        # Tool result: [{"text": "ADC Status Register...", "score": 0.92}, ...]
```

---

### 4. Extract All Tool Calls in Order

Use case: Audit what tools were called and in what sequence.

```python
result = agent.invoke({"messages": [("user", "Ingest PDF and search for timers")]})

call_log = []
for msg in result["messages"]:
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            call_log.append({"tool": tc["name"], "args": tc["args"]})

print(call_log)
# [
#   {"tool": "ingest_pdf_tool",    "args": {"file_path": "MC9S08SC4.pdf"}},
#   {"tool": "retrieve_context_tool", "args": {"query": "timers", "top_k": 5}},
# ]
```

---

### 5. Extract Structured Data from Final Response (JSON in text)

Use case: You prompt the LLM to return JSON and need to parse it.

```python
import json, re

result = agent.invoke({
    "messages": [("user", "List all GPIO registers as JSON array")]
})

text = result["messages"][-1].content

# Extract JSON block from markdown code fence
match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
if match:
    data = json.loads(match.group(1))
    print(data)  # [{"register": "PTAD", "offset": "0x00"}, ...]
```

---

### 6. Extract Data Using a Pydantic Model (Structured Output)

Use case: Force the LLM to always return a specific schema.

```python
from pydantic import BaseModel
from langchain_groq import ChatGroq

class RegisterInfo(BaseModel):
    name: str
    offset: str
    description: str

llm = ChatGroq(model="llama-3.3-70b-versatile")
structured_llm = llm.with_structured_output(RegisterInfo)

result = structured_llm.invoke("Describe the PTAD register of MC9S08SC4")
print(result.name)         # "PTAD"
print(result.offset)       # "0x00"
print(result.description)  # "Port A Data Register"
```

> **Note:** `with_structured_output` works on the LLM directly, not on a ReAct agent.
> Use this when you don't need tool calls — just structured LLM responses.

---

### 7. Stream Agent Output Token by Token

Use case: Show progress while the agent is thinking (long-running tasks).

```python
for chunk in agent.stream({"messages": [("user", "Generate HAL code")]}):
    # chunk is a dict with the node name as key
    for node_name, node_output in chunk.items():
        msgs = node_output.get("messages", [])
        for msg in msgs:
            content = getattr(msg, "content", "")
            if content:
                print(content, end="", flush=True)
```

---

## Part 2: Effective Agent Usage Patterns

### Pattern 1: Single-Shot (No Memory)
Best for: One-off queries, stateless tasks.

```python
# Fresh agent every time — no conversation history
result = agent.invoke({"messages": [("user", "Generate ADC init code")]})
```

---

### Pattern 2: Multi-Turn Conversation (With Memory)
Best for: Interactive sessions where context builds up.

```python
checkpointer = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=checkpointer)

THREAD = {"configurable": {"thread_id": "session_42"}}

# Turn 1
agent.invoke({"messages": [("user", "Ingest MC9S08SC4.pdf")]}, config=THREAD)

# Turn 2 — agent remembers the PDF was already ingested
agent.invoke({"messages": [("user", "Now generate GPIO code")]}, config=THREAD)

# Turn 3 — agent still has full context
agent.invoke({"messages": [("user", "Add error handling to the GPIO code")]}, config=THREAD)
```

> Each `thread_id` is an independent conversation. Use different IDs for different users/tasks.

---

### Pattern 3: Sequential Agents (Pipeline)
Best for: This project — Intent Agent → Codegen Agent.

```python
# Agent 1: Decides workspace structure
workspace = run_intent_agent("Generate HAL for MC9S08SC4")

# Agent 2: Generates code using the workspace as context
result = run_pdf_grounded_codegen(request, workspace)

# Each agent has a focused, single responsibility
```

---

### Pattern 4: Parallel Agents
Best for: Independent tasks that don't depend on each other.

```python
from concurrent.futures import ThreadPoolExecutor

tasks = [
    "Generate ADC driver",
    "Generate GPIO driver",
    "Generate UART driver",
]

def run_task(prompt):
    return agent.invoke({"messages": [("user", prompt)]},
                        config={"configurable": {"thread_id": prompt}})

with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run_task, tasks))

for r in results:
    print(r["messages"][-1].content[:100])
```

---

### Pattern 5: Reflection / Self-Improvement
Best for: Validating and improving generated output.

```python
# Step 1: Generate
gen_result = codegen_agent.invoke({"messages": [("user", prompt)]}, config=cfg)
generated_code = gen_result["messages"][-1].content

# Step 2: Reflect
reflection_prompt = f"""
Review this generated C code for correctness and completeness:
{generated_code}

Issues found (or 'None'):"""

reflection = llm.invoke(reflection_prompt)

# Step 3: Improve if needed
if "None" not in reflection.content:
    improved = codegen_agent.invoke({
        "messages": [
            ("user", prompt),
            ("assistant", generated_code),
            ("user", f"Fix these issues: {reflection.content}"),
        ]
    }, config=cfg)
```

---

## Part 3: Common Mistakes and Fixes

| Mistake | Problem | Fix |
|---|---|---|
| `result["my_key"]` | LangGraph only returns `{"messages": [...]}` | Read from `result["messages"]` |
| Passing extra keys to `.invoke()` | Silently ignored by `create_react_agent` | Embed data in the message text instead |
| Same `thread_id` for different tasks | Conversation history bleeds between tasks | Use unique `thread_id` per task/user |
| Checking `tool_calls` on all messages | Only `AIMessage` has `tool_calls` | Use `getattr(msg, "tool_calls", None)` |
| Expecting tool result in `AIMessage` | Tool output is in `ToolMessage`, not `AIMessage` | Check `isinstance(msg, ToolMessage)` |
| No checkpointer for multi-turn | Agent forgets previous messages | Pass `checkpointer=MemorySaver()` to agent |

---

## Part 4: Quick Reference — Message Type Cheatsheet

```
HumanMessage   → user input text
               → .content = str

AIMessage      → LLM response (text or tool decision)
               → .content = str (final answer or empty if calling tools)
               → .tool_calls = [{"name": str, "args": dict, "id": str}]

ToolMessage    → result returned by executing a tool
               → .content = str (tool's return value as string)
               → .tool_call_id = str (matches AIMessage.tool_calls[].id)

SystemMessage  → system prompt (set via agent's prompt= parameter)
               → .content = str
```
