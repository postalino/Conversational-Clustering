def _call_cluster_labeler(llm_service, user_content: str) -> dict:
    
    from action_schemas import SCHEMA_NAME_CLUSTER, _SYSTEM_PROMPT_NAME_CLUSTER
    
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_NAME_CLUSTER},
        {"role": "user", "content": user_content},
    ]
    return llm_service.chat_json(messages=messages, schema=SCHEMA_NAME_CLUSTER)


def name_cluster(llm_service, examples: list[dict]) -> dict:
    if not examples:
        return {"name": "Empty Cluster", "description": "No examples available.", "evidence": []}

    example_texts = "\n\n".join(f"- {ex['Text']}" for ex in examples)
    user_content = f"""
        Here are representative reviews from one cluster:

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


def parse_feedback(
    llm_service,
    user_feedback: str,
    cluster_metadata: dict,
) -> dict:
    
    from action_schemas import SCHEMA_ACTION_CALL, _SYSTEM_PROMPT_ACTION_FEEDBACK, ACTION_SCHEMAS
    
    cluster_summary = "\n".join(
        f"- Cluster {cluster_id}: {metadata['name']} | {metadata['description']}"
        for cluster_id, metadata in cluster_metadata.items()
    )

    user_content = f"""
    Available actions:
    {ACTION_SCHEMAS}

    Current clusters:
    {cluster_summary}

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

    return llm_service.chat_json(
        messages=messages,
        schema=SCHEMA_ACTION_CALL
    )
    
    


if __name__ == "__main__":
    # Test parse_feedback
    #1. generiamo un cluster_metadata di esempio
    metadata = {
        0: {"name": "Battery Issues", "description": "Reviews mentioning battery problems.", "evidence": ["battery", "charge", "power"]},
        1: {"name": "Great Value", "description": "Reviews praising the product's value for money.", "evidence": ["price", "value", "cost"]},
        2: {"name": "Poor Build Quality", "description": "Reviews criticizing the   product's build quality.", "evidence": ["build", "quality", "durability"]},
    }

    #2. definiamo un feedback di esempio
    feedback = "I think cluster 0 should be renamed to 'Battery Performance' because it includes both positive and negative reviews about battery life, not just issues."

    #3. chiamiamo parse_feedback
    from llm import LLMService
    llm_service = LLMService() 
    result = parse_feedback(llm_service, feedback, metadata)
    print(result)