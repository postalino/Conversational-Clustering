# Schemi JSON per le risposte dell'LLM
SCHEMA_NAME_CLUSTER = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"}
        },
    },
    "required": ["name", "description", "evidence"],
}

SCHEMA_ACTION_CALL = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["tool_name", "arguments"]
}

# Prompts per il comportamento dell'LLM quando deve svolgere azioni specifiche

_SYSTEM_PROMPT_NAME_CLUSTER = (
    "You are an analytical assistant specialized in labeling clusters of product reviews. "
    "Your task is to generate neutral and descriptive labels. "
    "Do NOT use marketing language, exaggerated claims, or promotional tone. "
    "The cluster name should be short, precise, and descriptive of the common topic. "
    "The description should objectively summarize the shared theme of the reviews. "
    "The evidence field should explain which recurring elements in the examples support the label. "
    "Return ONLY valid JSON matching the provided schema."
)

_SYSTEM_PROMPT_ACTION_FEEDBACK = (
    "You are an action-selection assistant for an interactive clustering system. "
    "Your task is to read the user's feedback and select exactly one available action. "
    "You must choose only from the provided action schemas. "
    "Do not invent new actions, parameters, or cluster IDs. "
    "If the user request is ambiguous or missing required information, choose needs_clarification. "
    "If the user asks for something outside the available tools, choose no_action. "
    "Return ONLY valid JSON matching the provided output schema."
)

# Schema delle azioni che l'LLM può scegliere in risposta al feedback dell'utente

ACTION_SCHEMAS = {
    "rename_cluster": {
        "type": "function",
        "function": {
            "name": "rename_cluster",
            "description": (
                "Rename or improve the metadata of an existing cluster. "
                "Use this when the user says the current label is wrong, misleading, too generic, unclear, "
                "too long, too vague, or should be shorter, more compact, simpler, clearer, or more specific. "
                "This action does not move items and does not change cluster assignments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "integer",
                        "description": "ID of the cluster to rename."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of why the cluster label should be renamed or improved."
                    }
                },
                "required": ["cluster_id", "reason"]
            }
        }
    },

    "merge_clusters": {
        "type": "function",
        "function": {
            "name": "merge_clusters",
            "description": (
                "Merge two existing clusters. "
                "Use this when the user says two clusters are similar, overlapping, duplicates, redundant, "
                "or should belong together. The merged cluster will be relabeled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id_1": {
                        "type": "integer",
                        "description": "ID of the first cluster to merge."
                    },
                    "cluster_id_2": {
                        "type": "integer",
                        "description": "ID of the second cluster to merge."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of why these two clusters should be merged."
                    }
                },
                "required": ["cluster_id_1", "cluster_id_2", "reason"]
            }
        }
    },

    "split_cluster": {
        "type": "function",
        "function": {
            "name": "split_cluster",
            "description": (
                "Split one existing cluster. "
                "Use this when the user says a cluster is too broad, too generic, mixed, incoherent, "
                "contains multiple topics, or should be divided into more specific groups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "integer",
                        "description": "ID of the cluster to split."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of why this cluster should be split."
                    }
                },
                "required": ["cluster_id", "reason"]
            }
        }
    },

    "move_item": {
        "type": "function",
        "function": {
            "name": "move_item",
            "description": (
                "Move one review/item to another existing cluster. "
                "Use this only when the user explicitly identifies both the item ID and the target cluster ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "integer",
                        "description": "ID of the review/item to move."
                    },
                    "target_cluster_id": {
                        "type": "integer",
                        "description": "ID of the cluster where the item should be moved."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of why this item should be moved."
                    }
                },
                "required": ["item_id", "target_cluster_id", "reason"]
            }
        }
    },

    "needs_clarification": {
        "type": "function",
        "function": {
            "name": "needs_clarification",
            "description": (
                "Use this when the requested operation is supported, but required information is missing. "
                "For example: the user asks to split a cluster but does not specify which cluster; "
                "asks to merge clusters but does not provide both cluster IDs; "
                "or asks to move an item but does not provide both item ID and target cluster ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Clarifying question to ask the user."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of which required information is missing."
                    }
                },
                "required": ["question", "reason"]
            }
        }
    },

    "no_action": {
        "type": "function",
        "function": {
            "name": "no_action",
            "description": (
                "Use this when the user asks for something outside the available tools, "
                "or when no cluster operation should be applied. "
                "For example: unsupported analysis, plotting, exporting files, changing algorithms, "
                "answering general questions, or requests that cannot be handled by rename, merge, split, or move."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of why no available action can satisfy the request."
                    }
                },
                "required": ["reason"]
            }
        }
    }
}