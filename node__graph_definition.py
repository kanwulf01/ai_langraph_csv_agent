def extraction_node(state: AgentState):
    """Nodo 1: Usa PandasAI para extraer la información cruda del CSV"""
    query = state["query"]
    smart_df = get_smart_df()
    
    if smart_df is None:
        return {"error": "CSV no cargado"}

    try:
        raw_result = smart_df.chat(query)
        # Extraer el valor real si viene encapsulado
        df_result = getattr(raw_result, "value", raw_result)
        
        # Preparar el payload de filas/columnas
        if hasattr(df_result, "to_dict"):
            payload = {
                "columns": list(df_result.columns),
                "rows": df_result.to_dict(orient="records")
            }
        else:
            payload = {
                "columns": ["Resultado"],
                "rows": [{"value": str(df_result)}]
            }
            
        return {
            "raw_data": str(raw_result),
            "formatted_payload": payload
        }
    except Exception as e:
        return {"error": str(e)}

def humanize_node(state: AgentState):
    """Nodo 2: Toma el resultado crudo y lo convierte en lenguaje natural"""
    if state.get("error"):
        return state

    messages = [
        {"role": "system", "content": "Eres un asistente de soporte analítico. Resume los datos, llega a conclusiones y usa iconos. No des detalles técnicos de la consulta."},
        {"role": "user", "content": f"Pregunta: {state['query']}\nDatos: {state['raw_data']}"}
    ]

    response = completion(
        model="gpt-4o-mini",
        messages=messages,
        api_key=OPENAI_API_KEY
    )
    
    return {"final_response": response.choices[0].message.content}