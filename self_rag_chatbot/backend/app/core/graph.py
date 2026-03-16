"""
Builds and compiles the Self-RAG StateGraph.

`build_graph(checkpointer)` is called once at startup (inside the FastAPI
lifespan) and the compiled app is stored on `app.state.rag_graph`.
"""
from langgraph.graph import END, START, StateGraph

from app.core.nodes import (
    check_is_sup,
    check_is_use,
    decide_retrieval,
    generate_direct,
    generate_from_context,
    is_relevant,
    no_answer_found,
    retrieve,
    revise_answer,
    rewrite_question,
    route_after_decide,
    route_after_isuse,
    route_after_issup,
    route_after_relevance,
    save_memory,
    should_summarize,
    summarize_conversation,
)
from app.core.state import State


def build_graph(checkpointer):
    """
    Wire up all nodes and edges, then compile with the supplied checkpointer.
    Returns a `CompiledStateGraph` that supports `astream_events`.
    """
    g = StateGraph(State)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("retrieve", retrieve)
    g.add_node("generate_direct", generate_direct)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", check_is_sup)
    g.add_node("is_use", check_is_use)
    g.add_node("revise_answer", revise_answer)
    g.add_node("rewrite_question", rewrite_question)
    g.add_node("save_memory", save_memory)
    g.add_node("summarize_conversation", summarize_conversation)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Entry point
    g.add_edge(START, "decide_retrieval")

    # Route: use retrieval or answer directly from history
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )

    # Direct-generation path → save memory
    g.add_edge("generate_direct", "save_memory")

    # Retrieval path
    g.add_edge("retrieve", "is_relevant")

    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("generate_from_context", "is_sup")

    g.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {"is_use": "is_use", "revise_answer": "revise_answer"},
    )

    g.add_edge("revise_answer", "is_sup")  # loop until fully_supported or max retries

    g.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "save_memory": "save_memory",
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("rewrite_question", "retrieve")  # retry retrieval with new query

    g.add_edge("no_answer_found", "save_memory")

    # Memory management
    g.add_conditional_edges(
        "save_memory",
        should_summarize,
        {True: "summarize_conversation", False: END},
    )

    g.add_edge("summarize_conversation", END)

    return g.compile(checkpointer=checkpointer)


def build_initial_state(question: str) -> dict:
    """
    Create a fresh per-turn input state.
    The checkpointer automatically loads `chat_history` and `summary`
    from Postgres for the given thread_id, so we don't need to pass them here.
    """
    return {
        "question": question,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "is_sup": "",
        "evidence": [],
        "retries": 0,
        "is_use": "",
        "useful_reason": "",
    }
