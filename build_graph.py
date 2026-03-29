workflow = StateGraph(AgentState)

# Añadir nodos
workflow.add_node("extract", extraction_node)
workflow.add_node("humanize", humanize_node)

# Definir el flujo (Edges)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "humanize")
workflow.add_edge("humanize", END)

# Compilar el grafo
ticket_agent = workflow.compile()