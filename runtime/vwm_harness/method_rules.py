"""Shared execution rules for staged reasoning and co-evolution."""

SKILL_DISCOVERY = (
    "Loading a skill is capability discovery, not a commitment to generate. An agent "
    "may read a matching skill and then judge the visual operation opportunity none. "
    "This skill-only path is valid, including when the skill explains why an operation "
    "is unsuitable. selected_task_skill may therefore name a loaded skill while "
    "visual_operation_opportunity is none. Preserve the loaded skill in this case "
    "and report the operation opportunity independently."
)

P2_AUTONOMY = (
    "P1 is an advisory capability judgment, not a binding call gate. In P2 you may "
    "reconsider a previous no-skill or no-generation decision when the actual observations "
    "support a concrete unresolved relation and a useful visual operation. Explain that "
    "current judgment. Loading a skill does not require generation, and a P1 none judgment "
    "does not prohibit generation. Retain P1's recorded output and describe the "
    "current decision in the request justification."
)
