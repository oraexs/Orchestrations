"""Coding agent orchestration using local DB ingestion + semantic retrieval."""

from __future__ import annotations

import importlib
import re
import os
import copy
from typing import Any, Dict, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

# Fallback keeps functions callable even without langchain-core tool decorator.
def tool(func):
    return func

try:
    from agent.store_Db import (
        close_vector_db,
        ingest_document,
        retrieve_workspace_structure_by_prompt,
        semantic_search,
        store_workspace_structure,
    )
except ModuleNotFoundError:
    from store_Db import (
        close_vector_db,
        ingest_document,
        retrieve_workspace_structure_by_prompt,
        semantic_search,
        store_workspace_structure,
    )

class WorkspaceStructure(TypedDict):
    mainfolder: list[str]
    subfolders: Dict[str, list[str]]
    files: Dict[str, list[str]]

class CodegenRequest(TypedDict):
    pdf_path: str
    user_prompt: str
    top_k: int
    thread_id: str
    folder_structure: WorkspaceStructure


def _build_llm() -> Any:
    """Create LLM instance if optional dependency and API key are available."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        groq_mod = importlib.import_module("langchain_groq")
        chat_groq_cls = getattr(groq_mod, "ChatGroq")
    except (ModuleNotFoundError, AttributeError):
        return None

    return chat_groq_cls(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
    )


llm = _build_llm()


@tool
def ingest_pdf_tool(file_path: str) -> str:
    """Ingest supported file types into the local vector DB."""
    return ingest_document(file_path)


@tool
def retrieve_context_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k semantically similar chunks from local DB."""
    return semantic_search(query=query, top_k=top_k)


tools = [ingest_pdf_tool, retrieve_context_tool]

def _load_runtime_components() -> tuple[Any, Any, Any]:
    """Load LangGraph/LangChain components lazily to keep file import-safe."""
    try:
        prebuilt = importlib.import_module("langgraph.prebuilt")
        memory_mod = importlib.import_module("langgraph.checkpoint.memory")
        create_react = getattr(prebuilt, "create_react_agent")
        memory_saver_cls = getattr(memory_mod, "MemorySaver")
    except ModuleNotFoundError:
        legacy_mod = importlib.import_module("langgraph.agent")
        create_react = getattr(legacy_mod, "create_react_agent")
        memory_saver_cls = getattr(legacy_mod, "InMemorySaver")

    try:
        tools_mod = importlib.import_module("langchain_core.tools")
        tool_decorator = getattr(tools_mod, "tool")
    except ModuleNotFoundError:
        tool_decorator = tool

    return create_react, memory_saver_cls, tool_decorator


def create_coding_agent() -> Any:
    """Create coding agent with local DB-backed tools."""
    create_react_agent, memory_saver_cls, tool_decorator = _load_runtime_components()

    ingest_tool = tool_decorator(ingest_pdf_tool)
    retrieve_tool = tool_decorator(retrieve_context_tool)

    return create_react_agent(
        model=llm,
        tools=[ingest_tool, retrieve_tool],
        prompt=(
            "You are a coding assistant. "
            "When a PDF path is provided, first ingest it with ingest_pdf_tool. "
            "Then retrieve relevant chunks using retrieve_context_tool. "
            "Use retrieved chunks as grounding context to generate code. "
            "Note: You have to generate code only based on the retrieved context. "
            "When generating code for multiple files, output each file in this exact format:\n"
            "### FILE: <relative/path/to/file>\n"
            "```\n"
            "<code here>\n"
            "```\n"
            "Use this format for every file so the code can be mapped to the correct file path."
        ),
        checkpointer=memory_saver_cls(),
    )

#Need to define the tools for the agent to create the folders and ask the structure to store in the local DB to pass it to the next agent for the code generation.
@tool
def create_workspace(structure: WorkspaceStructure) -> None:
    """Create folders and files based on the provided workspace structure."""
    mainfolder = structure["mainfolder"]
    subfolders = structure["subfolders"]
    files = structure["files"]

    for folder in mainfolder:
        os.makedirs(folder, exist_ok=True)

    for parent, subs in subfolders.items():
        for sub in subs:
            #if the subfolder is existing already, then do not create again to avoid the FileExistsError, just continue with the next one
            if not os.path.exists(os.path.join(parent, sub)):
                os.makedirs(os.path.join(parent, sub), exist_ok=True)

    for folder, file_names in files.items():
        for file_name in file_names:
            file_path = os.path.join(folder, file_name)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path) or folder, exist_ok=True)
                #create the file if it does not exist, if it exists already, then do not create again to avoid the FileExistsError, just continue with the next one
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")  # Create an empty file
    

@tool
def modify_workspace(existing_structure: WorkspaceStructure, new_structure: WorkspaceStructure) -> None:
    """Modify the existing workspace by creating and deleting the folders and files based on the provided workspace structure."""
    main_folders = new_structure["mainfolder"]
    subfolders = new_structure["subfolders"]
    files = new_structure["files"]

    #If the folder name is changed based on the user intent, then we need to create the new folder and delete the old folder to reflect the changes in the workspace structure. Same for the files, if the file name is changed, then we need to create the new file and delete the old file to reflect the changes in the workspace structure.
    #First delete the old folders
    for folder in existing_structure["mainfolder"]:
        if folder not in main_folders:
            if os.path.exists(folder):
                os.rmdir(folder)
                old_folder_removed_flag = True
            else:
                old_folder_removed_flag = False

    #As the old folders are removed, we can create the new folders based on the user intent
    #This is only change for the workspace structure
    for folder in main_folders:
        if folder not in existing_structure["mainfolder"]:
            create_workspace(new_structure)
    

def Intent_agent() -> Any:
    """Create intent agent to analyze user intent and provide feedback."""
    create_react_agent, memory_saver_cls, tool_decorator = _load_runtime_components()

    make_workspace_tool = tool_decorator(create_workspace)

    return create_react_agent(
        model=llm,
        tools=[make_workspace_tool],
        prompt=(
            "You are an assistant that analyzes user intent based on their prompt and provides feedback. "
            "When the user provides a prompt, analyze it to understand their intent regarding the workspace structure needed for code generation. "
        ),
        checkpointer=memory_saver_cls(),
    )
#This will run the intent based on the user input and create the workspace structure for the code generation task. The structure will be stored in the local DB to pass it to the next agent for the code generation.
def run_intent_agent(userPrompt) -> Any:

    intent_structure: WorkspaceStructure = {
        "mainfolder": [],
        "subfolders": {},
        "files": {},
    }

    # Reuse a previously stored workspace structure when the prompt is similar.
    stored_structure, stored_prompt = retrieve_workspace_structure_by_prompt(userPrompt)
    if stored_structure:
        intent_structure["mainfolder"] = stored_structure.get("mainfolder", [])
        intent_structure["subfolders"] = stored_structure.get("subfolders", {})
        intent_structure["files"] = stored_structure.get("files", {})

    intent_result = Intent_agent().invoke(
        {
            "messages": [
                (
                    "user",
                    "Analyze the user's intent and provide feedback. "
                    +userPrompt+
                    "This is the existing workspace structure: \n" + str(intent_structure) + "\n"
                    "Stored prompt used to retrieve this structure (if any): " + stored_prompt + "\n"
                    "If the LLM returned folders, subfolders and files are not present in the current workspace then create the folders, subfolders and files which are being missing"
                    "Pseudo code: \n"
                    "if the folder/subfolder/file does not exist in the current workspace structure, then create the folder/subfolder/file based on the user intent and return the updated workspace structure with the newly created folders/subfolders/files. \n",
                )
            ],
            "structure": intent_structure, #Pass this structure to the agent so that it can modify the structure based on the user intent and create the missing folders and files as a skeleton for the code generation task.
        },
        config={"configurable": {"thread_id": "intent_thread_1"}},
    )

    """Run the intent agent."""
    import json

    # intent_result is {"messages": [...]} from LangGraph.
    # The workspace structure is inside the create_workspace tool call arguments.
    messages = intent_result.get("messages", [])
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name == "create_workspace":
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                structure = args.get("structure", {}) if isinstance(args, dict) else {}
                if structure:
                    intent_structure["mainfolder"] = structure.get("mainfolder", [])
                    intent_structure["subfolders"] = structure.get("subfolders", {})
                    intent_structure["files"] = structure.get("files", {})
                    #Need to store this workspace structure in the local DB to pass it to the agent for generating the skeleton filesystem for the next time based on the reflection node feedback for the improvements.
                    #Also, helps to store the workspace structure for the next agent request to modify the existing files only if needed additional files could be added
                    store_workspace_structure(intent_structure, request_text=userPrompt)
                    break

    print("Intent_structure: \n", intent_structure["mainfolder"], "\n", intent_structure["subfolders"], "\n", intent_structure["files"])

    return copy.deepcopy(intent_structure)
    
_coding_agent: Any = None


def get_coding_agent() -> Any:
    """Lazily initialize and cache coding agent instance."""
    global _coding_agent
    if _coding_agent is None:
        _coding_agent = create_coding_agent()
    return _coding_agent

def run_pdf_grounded_codegen(req: CodegenRequest, workspace_structure: WorkspaceStructure) -> Dict[str, Any]:
    """Execute ingest -> semantic retrieval -> agent code generation."""
    pdf_path = req["pdf_path"]
    user_prompt = req["user_prompt"]
    top_k = req.get("top_k", 5)
    thread_id = req.get("thread_id", "user_thread_1")

    ingest_status = ingest_pdf_tool.invoke({"file_path": pdf_path}) if hasattr(ingest_pdf_tool, "invoke") else ingest_pdf_tool(pdf_path)
    retrieved_context = (
        retrieve_context_tool.invoke({"query": user_prompt, "top_k": top_k})
        if hasattr(retrieve_context_tool, "invoke")
        else retrieve_context_tool(user_prompt, top_k)
    )

    agent_input = {
        "messages": [
            (
                "user",
                "Generate code for this task:\n"
                + user_prompt
                + "\n\nUse this retrieved context from local DB:\n"
                + str(retrieved_context)
                + "\n\nWorkspace structure:\n"
                + str(workspace_structure)
                + "\n\nGenerate code only based on the retrieved context and the workspace structure. Do not use any external information. If the retrieved context is not sufficient to generate the code, say that you don't have enough information to generate the code.",
            )
        ]
    }

    agent_result = get_coding_agent().invoke(
        agent_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    agent_text = _extract_agent_text(agent_result)
    code_per_file = _parse_code_per_file(agent_text, workspace_structure)

    return {
        "ingest_status": ingest_status,
        "retrieved_context": retrieved_context,
        "agent_result": agent_result,
        "workspace": workspace_structure,
        "code": code_per_file,  # {file_path: code_string}
    }


def _parse_code_per_file(text: str, workspace_structure: WorkspaceStructure) -> Dict[str, str]:
    """Parse agent output into {file_path: code} mapping.

    Expects sections formatted as:
        ### FILE: <path>
        ```
        <code>
        ```
    Falls back to writing the full text into every file in the workspace.
    """
    pattern = re.compile(
        r"###\s*FILE:\s*([^\n]+)\n```[^\n]*\n([\s\S]*?)```",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if matches:
        return {path.strip(): code.strip() for path, code in matches}

    # Fallback: map full text to every file declared in workspace
    result: Dict[str, str] = {}
    for folder, file_names in workspace_structure["files"].items():
        for file_name in file_names:
            result[os.path.join(folder, file_name)] = text
    return result


def _extract_agent_text(agent_result: Any) -> str:
    """Extract a printable/writable text response from agent output."""
    if isinstance(agent_result, str):
        return agent_result

    if isinstance(agent_result, dict):
        messages = agent_result.get("messages")
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", None)
            if content is not None:
                return str(content)
        return str(agent_result)

    return str(agent_result)

#Adding the reflection pattern to the agent for self improvement based on the results:
class ReflectionState(TypedDict, total=False):
    agent_result: Any
    feedback: str
    improved_result: Any

def reflection_node(state: ReflectionState) -> ReflectionState:
    """LangGraph node to reflect on agent results and generate feedback."""
    agent_result = state["agent_result"]
    # Simple heuristic feedback generation (could be replaced with LLM-based analysis)
    if "error" in str(agent_result).lower():
        feedback = "The agent result contains an error. Please review the retrieved context and try again."
        improved_result = None
    else:
        feedback = "The agent result looks good. No improvements needed."
        improved_result = run_pdf_grounded_codegen()

    return {
        "feedback": feedback,
        "improved_result": improved_result,
    }

if __name__ == "__main__":
    try:
        if llm is None:
            raise RuntimeError("Configure ChatGroq and GROQ_API_KEY before running this module.")

        workspace = run_intent_agent("Add ADC module for the existing HAL codebase")

        # request: CodegenRequest = {
        #     "pdf_path": r"C:\Users\OFE1COB\Desktop\PP\LangGraph\Codegen_RAG_GEN\MC9S08SC4.pdf",
        #     "user_prompt": "Generate HAL module(Hardware abstraction layer) code for MC9S08SC4",
        #     "top_k": 8,
        #     "thread_id": "user_thread_1",
        #     "folder_structure": workspace,
        # }

        # result = run_pdf_grounded_codegen(request, workspace)
        # reflection_state: ReflectionState = {"agent_result": result["agent_result"]}
        # reflection_result = reflection_node(reflection_state)

        # # Traverse workspace and write generated code into target files.
        # # result["code"] is {file_path: code_string} so each file gets its own code.
        # for file_path_out, code_content in result["code"].items():
        #     os.makedirs(os.path.dirname(file_path_out) or ".", exist_ok=True)
        #     with open(file_path_out, "w", encoding="utf-8") as f:
        #         f.write(code_content)
        #     print(f"Written: {file_path_out}")

        # print("Ingestion:", result["ingest_status"])
        # print("Retrieved chunks:", len(result["retrieved_context"]))
        # print("code:", result["code"])
        # print("Workspace:", result["workspace"])
        # print("Code files written:", list(result["code"].keys()))
    finally:
        close_vector_db()