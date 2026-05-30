from action_schemas import (
    SCHEMA_NAME_CLUSTER,
    SCHEMA_ACTION_CALL,
    ACTION_SCHEMAS,
    _SYSTEM_PROMPT_NAME_CLUSTER,
    _SYSTEM_PROMPT_ACTION_FEEDBACK,
)


def _call_cluster_labeler(llm_service, user_content: str) -> dict:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_NAME_CLUSTER},
        {"role": "user", "content": user_content},
    ]
    return llm_service.chat_json(messages=messages, schema=SCHEMA_NAME_CLUSTER)


def name_cluster(llm_service, examples: list[dict], reason: str | None = None) -> dict:
    if not examples:
        return {"name": "Empty Cluster", "description": "No examples available.", "evidence": []}

    example_texts = "\n\n".join(f"- {ex['Text']}" for ex in examples)
    reason_block = f"\n        User's reason for this operation:\n        {reason}\n" if reason else ""
    user_content = f"""
        Here are representative reviews from one cluster:
        {reason_block}
        {example_texts}

        Generate:
        1. A short descriptive cluster name.
        2. A concise neutral description.
        3. Evidence explaining why these reviews belong together.
    """
    return _call_cluster_labeler(llm_service, user_content)


def rename_cluster(llm_service, examples: list[dict], old_name: str, reason: str | None = None) -> dict:
    example_texts = "\n\n".join(f"- {ex['Text']}" for ex in examples)
    user_content = f"""
        Current cluster name:
        {old_name}

        Reason for renaming:
        {reason if reason else "No specific reason provided."}

        Representative reviews from this cluster:

        {example_texts}

        Revise the cluster metadata:
        1. A better short descriptive cluster name.
        2. A concise neutral description.
        3. Evidence explaining why the revised label fits these reviews.
    """
    return _call_cluster_labeler(llm_service, user_content)


def _format_history(history: list[dict]) -> str:
    lines = []
    for t in history:
        args = t["parsed_action"].get("arguments", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        lines.append(
            f"- Turn {t['turn_id']}: \"{t['user_input']}\" "
            f"→ {t['action_executed']}({args_str}) "
            f"[{t['status']}]"
        )
    return "\nConversation history:\n" + "\n".join(lines) + "\n"


def parse_feedback(
    llm_service,
    user_feedback: str,
    cluster_metadata: dict,
    history: list[dict] | None = None,
) -> dict:
    cluster_summary = "\n".join(
        f"- Cluster {cluster_id}: {meta['name']} | {meta['description']}"
        for cluster_id, meta in cluster_metadata.items()
    )

    history_block = _format_history(history) if history else ""

    user_content = f"""
    Available actions:
    {ACTION_SCHEMAS}

    Current clusters:
    {cluster_summary}
    {history_block}
    User feedback:
    {user_feedback}

    Select exactly one action.

    Return JSON with:
    - tool_name: the selected action name
    - arguments: the arguments for that action
    """

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_ACTION_FEEDBACK},
        {"role": "user", "content": user_content},
    ]

    return llm_service.chat_json(messages=messages, schema=SCHEMA_ACTION_CALL)


if __name__ == "__main__":
    from llm import LLMService

    metadata = {
        0: {"name": "Battery Issues", "description": "Reviews mentioning battery problems.", "evidence": ["battery", "charge", "power"]},
        1: {"name": "Great Value", "description": "Reviews praising the product's value for money.", "evidence": ["price", "value", "cost"]},
        2: {"name": "Poor Build Quality", "description": "Reviews criticizing the product's build quality.", "evidence": ["build", "quality", "durability"]},
    }

    feedback = "I think cluster 0 should be renamed to 'Battery Performance' because it includes both positive and negative reviews about battery life, not just issues."

    llm_service = LLMService()
    result = parse_feedback(llm_service, feedback, metadata)
    print(result)