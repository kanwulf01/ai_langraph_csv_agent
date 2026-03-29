import logging
import os
import time
import requests
import pandas as pd
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from litellm import completion
import uvicorn

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()

FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")   
DOMAIN = os.getenv("FRESHDESK_DOMAIN")
BASE_URL = f"https://{DOMAIN}.freshdesk.com/api/v2/tickets"
AUTH = (f"{FRESHDESK_API_KEY}", "X") 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPEN_AI_KEY"))
CSV_FILE = "tickets_limpios.csv"

columnas_relevantes = [
    "id", "subject", "type", "status", "priority","responder_id", 
    "created_at", "updated_at", "group_id", "requester_id", 
    "custom_fields", "tags", "structured_description", "due_by",
]

priority_map = {1: "baja", 2: "media", 3: "alta", 4: "urgente"}

MAPPING_GROUPS = {
    47000663898.0:"assegur", 47000660904.0:"atlas", 47000662835.0:"banco Corrientes",
    47000664561.0:"basa", 47000664736.0:"bcp", 47000040503.0:"ccb",
    47000665381.0:"cci", 47000666833.0:"gestión N5", 47000664737.0:"itau",
    47000664135.0:"solar", 47000661108.0:"sudameris",
}

MAPPING_AGENT_ID = {
  47098525160.0: "Alan Breidyn Rios Briceño",
  47091617498.0: "Alejandro Denis",
  47084669541.0: "Alejandro Rodríguez",
  47084669512.0: "Alejo Alejandro Vallejo",
  47100373841.0: "Alonso Guillermo Chiquinquira Parra",
  47099692343.0: "DevOps N5",
  47020420465.0: "Support N5 Now",
}

STATUS_MAPPING_2 = {
    2:"empezando proceso", 3:"investigando el caso", 4:"resuelto",
    5:"cerrado", 8:"en progreso", 9:"Pendiente planificación",
    10:"Fusionado", 11:"Esperando respuesta del cliente", 12:"A la espera de despliegue",
    13:"Esperando confirmación", 14:"Pendiente de estimación", 15:"Enviado comercial",
    16:"Bloqueado", 17:"Estimado", 18:"Validando QA N5"
}


app_state = {}

def init_smart_dataframe():
    if not os.path.exists(CSV_FILE):
        print(f"Advertencia: El archivo {CSV_FILE} no existe todavía.")
        return None
    try:
        df = pd.read_csv(CSV_FILE, sep=",")
        df = df.dropna(axis=1, how='all')
        llm = LiteLLM(model="openai/gpt-5.4-mini", api_key=OPENAI_API_KEY) # Ajustado a un modelo estándar, cámbialo si usas uno custom
        return SmartDataframe(df, config={"llm": llm})
    except Exception as e:
        print(f"Error inicializando SmartDataframe: {e}")
        return None


class AgentState(TypedDict):
    query: str
    raw_result_str: str
    columns: List[Any]
    rows: Any
    total_rows: int
    final_response: str
    error: Optional[str]

def node_extract_data(state: AgentState) -> Dict:
    smart_df = app_state.get("smart_df")
    if smart_df is None:
        return {"error": "El archivo CSV no está cargado."}
    
    try:
        raw_result = smart_df.chat(state["query"])
        df = getattr(raw_result, "value", raw_result)
        raw_result_str = str(raw_result) if raw_result is not None else "Sin resultados."

        if hasattr(df, "to_dict"):
            columns = list(df.columns)
            rows = df.to_dict(orient="records")
            total_rows = len(rows)
        else:
            columns = []
            rows = [{"value": str(df)}]
            total_rows = 1

        return {
            "raw_result_str": raw_result_str,
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
            "error": None
        }
    except Exception as e:
        return {"error": f"Error en PandasAI: {str(e)}"}

def node_humanize_response(state: AgentState) -> Dict:
    if state.get("error"):
        return {"final_response": "Ocurrió un error en la extracción de datos, no pude generar una respuesta."}

    messages = [
        {
            "role": "system", 
            "content": """
                Eres un asistente de soporte técnico analítico y amable.
                Tu trabajo es tomar datos crudos de una base de datos de tickets, resumirlos, analizarlos
                y llegar siempre a una conclusion de forma conversacional, clara y profesional sobre los datos resumidos.
                Reglas:
                1. Nunca debes dar contexto del proceso al usuario.
                2. Nunca des detalles de los calculos o las querys ejecutadas.
                3. Debes solo analizar y devolver resultados del analisis de datos, no devuelvas los datos crudos recibidos.
                4. Formatea el texto para que sea totalmente legible por humanos y agrega iconos cuando sea necesario.
            """
        },
        {
            "role": "user", 
            "content": f"Pregunta original: {state['query']}\nDatos crudos obtenidos: {state['raw_result_str']}"
        }
    ]

    try:
        humanized_response = completion(
            model="gpt-5.4-mini", 
            messages=messages,
            api_key=OPENAI_API_KEY
        )
        final_result = humanized_response.choices[0].message.content
        return {"final_response": final_result}
    except Exception as e:
        return {"error": f"Error en generación LLM: {str(e)}", "final_response": "Hubo un problema al formatear la respuesta."}

workflow = StateGraph(AgentState)

workflow.add_node("extract_data", node_extract_data)
workflow.add_node("humanize_response", node_humanize_response)

workflow.add_edge(START, "extract_data")
workflow.add_edge("extract_data", "humanize_response")
workflow.add_edge("humanize_response", END)

ticket_agent_app = workflow.compile()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Cargando el modelo y los datos en memoria...")
    app_state["smart_df"] = init_smart_dataframe()
    yield
    app_state.clear()

app = FastAPI(title="AI Ticket Query API (Powered by LangGraph)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4200", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None

class QueryResuestCreateCsvByDate(BaseModel):
    start_date: str 
    end_date: str 

class QueryResponseNasted(BaseModel):
    sources: list
    content: str
    raw_response: str
    columns: list
    rows: Any
    total_rows: int

class QueryResponse(BaseModel):
    response: QueryResponseNasted


@app.get("/")
def read_root():
    is_loaded = "alive" if app_state.get("smart_df") is not None else "not_loaded"
    return {"status": "API de Tickets activa", "file_loaded": is_loaded}

@app.post("/ask", response_model=QueryResponse)
async def ask_csv(request: QueryRequest):
    if app_state.get("smart_df") is None:
        raise HTTPException(status_code=500, detail="El archivo CSV no está cargado. Ejecuta /create_csv primero.")
    
    # 🚀 Invocamos el Grafo de LangGraph
    initial_state = {
        "query": request.query,
        "raw_result_str": "",
        "columns": [],
        "rows": [],
        "total_rows": 0,
        "final_response": "",
        "error": None
    }
    
    result_state = ticket_agent_app.invoke(initial_state)

    if result_state.get("error") and not result_state.get("final_response"):
        raise HTTPException(status_code=500, detail=result_state["error"])

    return {
        "response": {
            "sources": [], 
            "content": result_state["final_response"],
            "raw_response": result_state["raw_result_str"],
            "columns": result_state["columns"],
            "rows": result_state["rows"],
            "total_rows": result_state["total_rows"]
        }
    }

def create_csv_file(all_data:list):
    df = pd.DataFrame(all_data)
    columnas_validas = [col for col in columnas_relevantes if col in df.columns]
    df_filtrado = df[columnas_validas].copy()

    if 'custom_fields' in df_filtrado.columns:
        custom_df = pd.json_normalize(df_filtrado['custom_fields'])
        df_filtrado = pd.concat([df_filtrado.drop(columns=['custom_fields']), custom_df], axis=1)

    if "priority" in df_filtrado.columns:
        df_filtrado["prioridad"] = df_filtrado["priority"].map(priority_map)
    if "group_id" in df_filtrado.columns:
        df_filtrado["cliente"] = df_filtrado["group_id"].map(MAPPING_GROUPS)
    if "status" in df_filtrado.columns:
        df_filtrado["estado"] = df_filtrado["status"].map(STATUS_MAPPING_2)
    if "responder_id" in df_filtrado.columns:
        df_filtrado["agente"] = df_filtrado["responder_id"].map(MAPPING_AGENT_ID)
    if "due_by" in df_filtrado.columns:
        df_filtrado["fecha de vencimiento"] = df_filtrado["due_by"].map(MAPPING_AGENT_ID)

    df_filtrado.to_csv(CSV_FILE, index=False)
    app_state["smart_df"] = init_smart_dataframe()
    return {"status": "CSV creado exitosamente", "file": CSV_FILE}

@app.post("/create_csv_by_date")
def create_csv_by_date(request: QueryResuestCreateCsvByDate):
    all_tickets = []
    end_date = request.end_date if request.end_date else time.strftime("%Y-%m-%d", time.gmtime())
    start_date = request.start_date if request.start_date else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30*24*3600))

    url = f"https://{DOMAIN}.freshdesk.com/api/v2/search/tickets?query=\"updated_at:>'{start_date}' AND updated_at:<'{end_date}'\""
    r = requests.get(url, auth=AUTH, timeout=60)
    r.raise_for_status()
    result = r.json()
    all_tickets.extend(result["results"])

    return create_csv_file(all_tickets)

@app.get("/create_csv")
def create_csv(date_filter:str):
    max_retries = 3
    retry_count = 0
    all_tickets = []
    page = 1
    max_pages = 100
    update_since = f"updated_since={date_filter}T00:00:00Z" 
    
    while page <= max_pages:
        url = f"{BASE_URL}?{update_since}&page={page}&per_page=100"
        try:
            r = requests.get(url, auth=AUTH, timeout=60)
            r.raise_for_status()
            articles_page = r.json()
            if not articles_page: 
                break
                
            all_tickets.extend(articles_page)
            link_header = r.headers.get("link", "")
            if not link_header or 'rel="next"' not in link_header:
                break
            
            page += 1
            retry_count = 0 
            time.sleep(1) 

        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise 
            time.sleep(10 * retry_count) 

    if not all_tickets:
        raise HTTPException(status_code=400, detail="No se encontraron tickets en Freshdesk.")

    return create_csv_file(all_tickets)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)