import os
import pandas as pd
from typing import Annotated, TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN ---
CSV_FILE = "tickets_limpios.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Definimos el esquema del estado que pasará por los nodos
class AgentState(TypedDict):
    query: str
    raw_data: Optional[Any]
    formatted_payload: Optional[dict]
    final_response: Optional[str]
    error: Optional[str]

# --- UTILIDADES DE DATOS (Nodes) ---

def get_smart_df():
    if not os.path.exists(CSV_FILE):
        return None
    df = pd.read_csv(CSV_FILE)
    llm = LiteLLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY) # Ajustado a modelo real
    return SmartDataframe(df, config={"llm": llm})