import logging
import os
import time
import requests
import pandas as pd
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from litellm import completion # Asegúrate de importar esto arriba
import uvicorn

load_dotenv()

# --- CONFIGURACIÓN DE ENTORNO ---
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")   
DOMAIN = os.getenv("FRESHDESK_DOMAIN")
BASE_URL = f"https://{DOMAIN}.freshdesk.com/api/v2/tickets"
AUTH = (f"{FRESHDESK_API_KEY}", "X") 

# Nota: Asegúrate de que en tu .env la variable se llame OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPEN_AI_KEY"))
CSV_FILE = "tickets_limpios.csv"

# --- CONFIGURACIÓN DE DATOS ---
columnas_relevantes = [
    "id", "subject", "type", "status", "priority","responder_id", 
    "created_at", "updated_at", "group_id", "requester_id", 
    "custom_fields", "tags", "structured_description", "due_by",
]

priority_map = {
    1: "baja", 2: "media", 3: "alta", 4: "urgente"
}

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
  47100373841.0: "Alonso Guillermo Chiquinquira Parra Parra",
  47088438781.0: "Amelia Bernal",
  47056780464.0: "Armando Machado",
  47096465804.0: "Camilo Valderrama",
  47090010845.0: "Celia Palacios",
  47096465774.0: "Christian Millan",
  47020650731.0: "Claudia Bello",
  47093836316.0: "Cristian arboleda",
  47092196481.0: "Cristian Gomez",
  47096465159.0: "Cristian Yanes Espitia",
  47098401255.0: "Daniel Abraham Portillo",
  47064482179.0: "Daniela Aranda",
  47060012501.0: "Debora Urbanitsch",
  47099692343.0: "DevOps N5",
  47086517196.0: "Diego Cabezas",
  47087223607.0: "Diego San San Esteban",
  47091599343.0: "Edson García",
  47090423471.0: "Edwin Chavarro",
  47083589715.0: "Elvis Salvatierra",
  47091364255.0: "Emiliano Elorza",
  47064482146.0: "Emilio Barat",
  47086437212.0: "Esteban Fernandez",
  47088542380.0: "Esteban Guzmán",
  47094874901.0: "Evelin Quiñonez",
  47093738025.0: "Fabio lefosse",
  47091774742.0: "Federico Beltrami",
  47097269932.0: "Fernado Martin Lamberti",
  47100373879.0: "Franco Turco",
  47096408078.0: "Gaston Barbaccia",
  47092553865.0: "Georgina Abdala",
  47092941096.0: "Gerardo Pezzuti",
  47064482202.0: "Gonzalo Fasce",
  47064482264.0: "Graciela.urdaneta",
  47080919806.0: "Guillermo Javier Liberona Álvarez",
  47088611455.0: "Hernan Buzzi",
  47081447774.0: "Implementacion Assegur",
  47095782930.0: "Implementacion BASA",
  47095782978.0: "Implementacion CCB",
  47095783119.0: "Implementacion CCI",
  47096403998.0: "Implementacion ITAU",
  47096381425.0: "Implementación Solar",
  47059291934.0: "Implementacion Sudameris",
  47096461821.0: "Jairo Gonzalez Boada",
  47091515469.0: "Javier Monge",
  47085576617.0: "Javier.rojas",
  47096465813.0: "Jean Carlos Nunez Hernandez",
  47091203695.0: "Jesica Gregorio",
  47086683841.0: "Jesus Castrillo",
  47086682795.0: "Jhon Sandoval",
  47088202821.0: "Jimena Jimena Lagrotta",
  47082512085.0: "Johan Tamayo",
  47092196619.0: "Jonathan Gollarza",
  47092683982.0: "Jorge Ramos",
  47088341265.0: "Jorge Vasquez",
  47086595635.0: "Juan Alcoba",
  47049945166.0: "Juan Ignacio Bowden",
  47098204233.0: "Juan Lorenzo Mejia Mejia",
  47041603374.0: "Juan Soto",
  47086683862.0: "Judelys Lagos",
  47082226437.0: "Julio Staude",
  47100373809.0: "Leonel Chavez Julca",
  47065976471.0: "Lina Giraldo",
  47084669488.0: "Luciana Escalada",
  47087654130.0: "Luis Padilla",
  47091364277.0: "Luis Veliz",
  47064482687.0: "Luisa Maria Vargas",
  47092063053.0: "Maria Bellati",
  47096464863.0: "Mariana Giselle Antelo Bollo",
  47095835515.0: "Mariano Javier Paz",
  47033197913.0: "Mariano Paolini",
  47092063076.0: "Mariela Mendez",
  47091364231.0: "Mariella Pastran",
  47091998334.0: "Mauricio Roh",
  47097474057.0: "Nicolás Rubiano",
  47097515618.0: "Noe Saul Mora Mora",
  47061906853.0: "Norberto Castellanos",
  47093532027.0: "Oscar Chavez",
  47086580535.0: "Pablo Boquin",
  47064881914.0: "Pablo Velan",
  47095835412.0: "Ramón Virgilio Jara",
  47086437228.0: "Rene Perez",
  47085290715.0: "Roberto Arias",
  47093738135.0: "Rodrigo Amurrio",
  47097515669.0: "Ruben Castillo",
  47091364218.0: "Samael FS",
  47083234565.0: "SEBASTIAN Moraga",
  47083448377.0: "Sebastian Pantuso",
  47083404019.0: "Sebastian Villagomez",
  47064482248.0: "Sergio Robles",
  47092783383.0: "Sergio Rondon",
  47091364265.0: "Sofia Mejia",
  47095126953.0: "Soporte Cross N5",
  47020420465.0: "Support N5 Now",
  47088542351.0: "Vanesa Yenyffer Choachi",
  47090438886.0: "vanessa alfonzo",
  47089111958.0: "Victor Suarez",
  47086580508.0: "Viviana Perez",
  47088815630.0: "Walther Choque"
}

STATUS_MAPPING_2 = {
    2:"Empezando proceso",
    3:"Investigando el caso",
    4:"Resuelto",
    5:"Cerrado",
    8:"En progreso",
    9:"Pendiente planificación",
    10:"Fusionado",
    11:"Esperando respuesta del cliente",
    12:"A la espera de despliegue",
    13:"Esperando confirmación",
    14:"Pendiente de estimación",
    15:"Enviado comercial",
    16:"Bloqueado",
    17:"Estimado",
    18:"Validando QA N5"
}

STATUS_MAPPING = {
      "2": [
        "Open",
        "Empezando el proceso"
      ],
      "3": [
        "Pending",
        "Investigando el Caso"
      ],
      "4": [
        "Resolved",
        "El ticket ha sido Resuelto"
      ],
      "5": [
        "Closed",
        "El ticket ha sido cerrado"
      ],
      "8": [
        "En Progreso",
        "Investigando el Caso"
      ],
      "9": [
        "Pendiente Planificación",
        "Pendiente Planificación"
      ],
      "10": [
        "Fusionado",
        "Fusionado"
      ],
      "11": [
        "Esperando respuesta del cliente",
        "Esperando respuesta de su parte"
      ],
      "12": [
        "A la espera de despliegue",
        "A la espera de despliegue"
      ],
      "13": [
        "Esperando confirmación",
        "Esperando confirmación."
      ],
      "14": [
        "Pendiente de estimación",
        "Estimando el Caso"
      ],
      "15": [
        "Enviado comercial",
        "Estimando el Caso"
      ],
      "16": [
        "Bloqueado",
        "Investigando el Caso"
      ],
      "17": [
        "Estimado",
        "Estimando el Caso"
      ],
      "18": [
        "Validando QA N5",
        "Investigando el Caso"
      ]
    }

# --- ESTADO GLOBAL DE LA APP ---
# Usamos un diccionario para mantener el dataframe inteligente en memoria
app_state = {}

def init_smart_dataframe():
    """Función para cargar el CSV en memoria una sola vez"""
    if not os.path.exists(CSV_FILE):
        print(f"Advertencia: El archivo {CSV_FILE} no existe todavía.")
        return None

    try:
        df = pd.read_csv(CSV_FILE, sep=",")
        df = df.dropna(axis=1, how='all')
        
        llm = LiteLLM(model="openai/gpt-5.4-mini", api_key=OPENAI_API_KEY)
        return SmartDataframe(df, config={"llm": llm})
    except Exception as e:
        print(f"Error inicializando SmartDataframe: {e}")
        return None

# --- CICLO DE VIDA DE LA APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta al iniciar el servidor
    print("Cargando el modelo y los datos en memoria...")
    app_state["smart_df"] = init_smart_dataframe()
    yield
    # Se ejecuta al apagar el servidor
    app_state.clear()

# Inicializamos FastAPI con el lifespan
app = FastAPI(title="AI Ticket Query API", lifespan=lifespan)

# --- CORS ---
origins = [
    "http://localhost:5173",     # Front local (React / Next / Vite)
    "http://localhost:4200",     # Angular
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS DE DATOS ---
class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None  # Corregido: Tipo explícito para evitar fallos de Pydantic

class QueryResuestCreateCsvByDate(BaseModel):
    start_date: str  # Formato ISO, e.g. "2026-01-01T00:00:00Z"
    end_date: str    # Formato ISO, e.g. "2026-01-31T23:59:59Z"

class QueryResponseNasted(BaseModel):
    sources: list
    content: str
    raw_response:str
    columns:list
    rows:Any
    total_rows:int

class QueryResponse(BaseModel):
    response: QueryResponseNasted

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    is_loaded = "alive" if app_state.get("smart_df") is not None else "not_loaded"
    return {"status": "API de Tickets activa", "file_loaded": is_loaded}


@app.post("/create_csv_by_date")
def create_csv_by_date(request: QueryResuestCreateCsvByDate):
    result = {}
    all_tickets = []
    end_date = ""
    start_date = ""
    r = None
    if not request.end_date or request.end_date == "":
        end_date = time.strftime("%Y-%m-%d", time.gmtime())
        print("Vino aca", end_date)
    else:
        end_date = request.end_date

    if not request.start_date or request.start_date == "":
        start_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30*24*3600))  # Últimos 30 días
    else:
        start_date = request.start_date

    url = f"https://{DOMAIN}.freshdesk.com/api/v2/search/tickets?query=\"updated_at:>'{start_date}' AND updated_at:<'{end_date}'\""
    print("url", url)
    r = requests.get(url, auth=AUTH, timeout=60)
    r.raise_for_status()
    result = r.json()
    print("total", result["total"])
    for i in result["results"]:
        #print(i)
        all_tickets.append(i)

    create_csv_file(all_tickets)


@app.get("/create_csv")
def create_csv(date_filter:str):
    max_retries = 3
    retry_count = 0
    all_tickets = []
    page = 1
    max_pages = 100
    print("date_filter", date_filter)
    update_since = f"updated_since={date_filter}T00:00:00Z"  # Ajusta esta fecha según tus necesidades
    while page <= max_pages:
        url = f"{BASE_URL}?{update_since}&page={page}&per_page=100"
        try:
            logging.info(f"Requesting página {page}: {url}")
            r = requests.get(url, auth=AUTH, timeout=60)
            r.raise_for_status()
            
            articles_page = r.json()
            if not articles_page:  # No hay más artículos
                logging.info(f"No hay más artículos en página {page}")
                break
                
            all_tickets.extend(articles_page)
            logging.info(f"Página {page}: {len(articles_page)} artículos descargados")

            # Verificar si hay siguiente página
            link_header = r.headers.get("link", "")
            logging.info(f"Link header: {link_header}")

            if not link_header or 'rel="next"' not in link_header:
                logging.info("No hay más páginas")
                break
            
            page += 1
            retry_count = 0  # Reset retry count en éxito
            time.sleep(1)  # Rate limiting

        except requests.exceptions.RequestException as e:
            retry_count += 1
            logging.error(f"Error al descargar página {page} (intento {retry_count}/{max_retries}): {e}")
            
            if retry_count >= max_retries:
                logging.error(f"Máximo de reintentos alcanzado para página {page}")
                raise 
            
            time.sleep(10 * retry_count)  # Backoff exponencial

    if not all_tickets:
        raise HTTPException(status_code=400, detail="No se encontraron tickets en Freshdesk.")

    df = pd.DataFrame(all_tickets)
    print("Total tickets descargados:", len(df))

    # Filtrar solo columnas existentes para evitar KeyError
    columnas_validas = [col for col in columnas_relevantes if col in df.columns]
    df_filtrado = df[columnas_validas].copy()

    print("Columnas después del filtrado:", df_filtrado.columns.tolist())

    # Expandir 'custom_fields' si existe (¡Crítico para PandasAI!)
    if 'custom_fields' in df_filtrado.columns:
        # Esto convierte {"cf_observacion": "X", "cf_producto": "Y"} en columnas nuevas
        custom_df = pd.json_normalize(df_filtrado['custom_fields'])
        df_filtrado = pd.concat([df_filtrado.drop(columns=['custom_fields']), custom_df], axis=1)

    # Mapear valores
    if "priority" in df_filtrado.columns:
        df_filtrado["prioridad"] = df_filtrado["priority"].map(priority_map)
    if "group_id" in df_filtrado.columns:
        df_filtrado["cliente"] = df_filtrado["group_id"].map(MAPPING_GROUPS)
    if "status" in df_filtrado.columns:
        df_filtrado["estado"] = df_filtrado["status"].map(STATUS_MAPPING_2)
    if "responder_id" in df_filtrado.columns:
        df_filtrado["agente"] = df_filtrado["responder_id"].map(MAPPING_AGENT_ID)
    if "due_by" in df_filtrado.columns:
        df_filtrado["fecha_de_vencimiento"] = df_filtrado["due_by"].map(MAPPING_AGENT_ID)
    if "created_at" in df_filtrado.columns:
        df_filtrado["fecha_creacion"] = df_filtrado["created_at"]

    # Guardar CSV
    df_filtrado.to_csv(CSV_FILE, index=False)
    print("CSV generado correctamente con campos aplanados.")

    # Recargar el modelo en memoria con el nuevo CSV
    app_state["smart_df"] = init_smart_dataframe()

    return {"status": "CSV creado exitosamente", "file": CSV_FILE}

@app.post("/ask", response_model=QueryResponse)
async def ask_csv(request: QueryRequest):
    smart_df = app_state.get("smart_df")
    
    if smart_df is None:
        raise HTTPException(status_code=500, detail="El archivo CSV no está cargado. Ejecuta /create_csv primero.")
    
    try:
        raw_result = smart_df.chat(request.query)
        code = smart_df.last_code_executed
        print("Código generado:", code)
        print("data cruda real: \n",type(raw_result))
        df = getattr(raw_result, "value", raw_result)
        #table_str = df.to_string(index=False, max_rows=None, max_cols=None)

        raw_result_str = str(raw_result) if raw_result is not None else "Sin resultados."

        summary = f"""
            Shape:{df.shape}
            Columns: {list(df.columns)}
            Sample:
            {df.head(50).to_string(index=False)}
            Stats:
            {df.describe(include='all').to_string()}
        """
        if hasattr(df, "to_dict"):
            payload = {
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records"),
                "total_rows": 0,
            }
        else:
            payload = {
                "columns": [],
                "rows": [{"value": str(df)}],
                "total_rows": 1,
            }
        #print(f"Resultado crudo: {raw_result_str}")

        # 2. ETAPA DE HUMANIZACIÓN (Transformamos la tabla en lenguaje natural)
        messages = [
            {
                "role": "system", 
                "content": """
                    Eres un asistente de soporte técnico analítico y amable
                    Tu trabajo es tomar datos crudos de una base de datos de tickets, resumirlos, analizarlos
                    y llegar siempre a una conclusion de forma conversacional, clara y profesional sobre los datos resumidos
                        1. nunca debes dar contexto del proceso al usuario.
                        2. Nunca des detalles de los calculos o las queryes ejecutadas.
                        3. Debes solo analizar y devolver resultados del analisis de datos no devuelvas los datos curdos recibidos.
                        4. Fomatea el texto en formato MARKDOWN simpre es obligatorio y agrega iconos cuando sea necesario.

                """
            },
            {
                "role": "user", 
                "content": f"Pregunta original: {request.query}\nDatos de resumen inteligente obtenidos: {summary}"
            }
        ]

        humanized_response = completion(
            model="gpt-5-mini-2025-08-07", 
            messages=messages,
            api_key=OPENAI_API_KEY
        )
        
        final_result = humanized_response.choices[0].message.content
        
        return {
            "response": {
                "sources": [], 
                "content": final_result,
                "raw_response":"",
                "columns":payload["columns"],
                "rows":payload["rows"],
                "total_rows":payload["total_rows"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la consulta: {str(e)}")
    

def create_csv_file(all_data:list):
    #print(all_data[1])
    df = pd.DataFrame(all_data)
    print("Total tickets descargados:", len(df))

    # Filtrar solo columnas existentes para evitar KeyError
    columnas_validas = [col for col in columnas_relevantes if col in df.columns]
    df_filtrado = df[columnas_validas].copy()

    print("Columnas después del filtrado:", df_filtrado.columns.tolist())

    # Expandir 'custom_fields' si existe (¡Crítico para PandasAI!)
    if 'custom_fields' in df_filtrado.columns:
        custom_df = pd.json_normalize(df_filtrado['custom_fields'])
        df_filtrado = pd.concat([df_filtrado.drop(columns=['custom_fields']), custom_df], axis=1)

    # Mapear valores
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

    # Guardar CSV
    df_filtrado.to_csv(CSV_FILE, index=False)
    print("CSV generado correctamente con campos aplanados.")

    # Recargar el modelo en memoria con el nuevo CSV
    app_state["smart_df"] = init_smart_dataframe()

    return {"status": "CSV creado exitosamente", "file": CSV_FILE}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)