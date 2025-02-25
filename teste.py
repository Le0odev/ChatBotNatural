import mysql.connector
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
import google.generativeai as genai
import re
from fastapi.middleware.cors import CORSMiddleware
import json
from functools import lru_cache
import asyncio
import time

# Inicializar FastAPI
app = FastAPI()

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://emporioverdegraos.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração da API Gemini
genai.configure(api_key="AIzaSyBWpMyp95TwrPPUxHBp0OWQGYNcs-UlS8M")

# Modelo de embeddings (usando um modelo mais avançado)
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Categorias e sinônimos para contexto
CATEGORIES = {
    "Suplementos": ["whey", "bcaa", "glutamina", "proteína", "aminoácidos", "suplementação", "suplemento", "protein", "proteico"],
    "Barrinhas": ["protein bar", "top whey bar", "crisp", "barra de proteína", "snack proteico", "barrinha", "barrinhas", "barra proteica"],
    "Amendoins": ["pasta de amendoim", "amendoim", "manteiga de amendoim", "nuts", "creme de amendoim", "amendoa", "castanha"],
    "Pré-treino": ["nuclear rush", "évora pw", "horus", "c4", "pré-workout", "energia", "pré-treino", "pre treino", "pre-treino", "preworkout"],
    "Creatina": ["creatina", "creapure", "monohidratada", "força", "creatin", "kre-alkalyn"],
    "Vitaminas/Minerais": ["zma", "vitamina", "multivitamínico", "mineral", "micronutrientes", "multi", "complexo", "complexo vitamínico"],
    "Encapsulados": ["cafeína", "thermo", "termogênico", "emagrecedor", "metabolismo", "valeriana", "passiflora", "cápsula", "suplemento em cápsula", "cápsulas", "emagrecimento"],
    "Óleos": ["óleo", "ômega 3", "óleo de coco", "óleo de peixe", "gorduras boas", "azeite", "gordura saudável", "omega", "mct"],
    "Cereais": ["cereal", "grãos", "aveia", "granola", "fibras", "mingau", "café da manhã", "farinha"],
    "Energéticos": ["energel", "carb up", "gel energético", "maltodextrina", "carboidratos", "energia", "energético", "carboidrato"],
    "Temperos": ["tempero", "condimento", "especiarias", "sabor", "temperar", "ervas"],
    "Chá": ["chá", "infusão", "erva", "bebida funcional", "cha", "chás", "tea"],
    "Chips": ["chips", "snack", "petisco", "lanche saudável", "salgadinho", "snacks", "lanche"]
}

INGREDIENTS_SYNONYMS = {
    "valeriana": ["valeriana", "valerian", "sedativo natural", "planta medicinal para ansiedade"],
    "passiflora": ["passiflora", "maracujá", "passion flower", "flor da paixão", "erva calmante"],
    "ansiedade": ["ansiedade", "anxiety", "estresse", "stress", "nervoso", "insônia", "tensão", "acalmar"],
    "melatonina": ["melatonina", "sono", "sleep", "hormônio do sono", "regular sono"],
    "whey": ["whey", "proteína", "protein", "suplemento proteico", "proteina", "whey protein", "concentrado proteico"],
    "bcaa": ["bcaa", "aminoácidos", "aminoacidos", "aminoácidos essenciais", "recuperação muscular"],
    "creatina": ["creatina", "creatin", "creapure", "monohidratada", "kre-alkalyn", "força muscular"],
    "cafeína": ["cafeína", "caffeine", "cafein", "estimulante", "energia", "disposição", "café", "coffee"],
    "taurina": ["taurina", "taurine", "energia", "aminoácido", "pré-treino"],
    "colágeno": ["colágeno", "colageno", "collagen", "peptídeos de colágeno", "articulações", "pele"],
    "termogênico": ["termogênico", "termogenico", "thermo", "queimador de gordura", "metabolismo", "emagrecedor"],
    "glutamina": ["glutamina", "glutamine", "recuperação", "l-glutamina", "imunidade"],
    "omega 3": ["omega 3", "ômega 3", "omega três", "óleo de peixe", "dha", "epa", "gordura boa"],
    "proteína vegana": ["proteína vegana", "proteina vegana", "suplemento vegano", "plant based", "plant protein"],
    "magnésio": ["magnésio", "magnesium", "mg", "magnésio quelato", "cãibra", "caimbra", "relaxamento muscular"],
    "zinco": ["zinco", "zinc", "zn", "imunidade", "sistema imune", "hormônios"],
    "vitamina c": ["vitamina c", "ácido ascórbico", "vitamina c", "imunidade", "antioxidante"],
    "vitamina d": ["vitamina d", "vitamina d3", "colecalciferol", "sol", "ossos", "imunidade"],
    "probiótico": ["probiótico", "probiotico", "probiotic", "gut health", "intestino", "digestão", "flora intestinal"],
    "ashwagandha": ["ashwagandha", "adaptógeno", "estresse", "ansiedade", "energia"],
    "maca peruana": ["maca peruana", "maca", "peruvian maca", "libido", "energia", "disposição"],
    "própolis": ["própolis", "propolis", "imunidade", "anti-inflamatório", "garganta"],
    "spirulina": ["spirulina", "espirulina", "alga", "desintoxicação", "proteína vegetal", "antioxidante"],
    "fibras": ["fibras", "fibra", "fiber", "intestino", "digestão", "saciedade"]
}

# Modelos Pydantic para validação de dados
class Product(BaseModel):
    id: int
    product_name: str
    product_description: str
    categoria: Optional[str]
    product_price: float
    ingredients: List[str] = []

class ChatResponse(BaseModel):
    resposta: str
    produtos_relacionados: List[Product]
    total: Optional[float] = None  # Adicionando campo para o total

class RecommendedProduct(BaseModel):
    name: str
    categoria: str
    description: str
    price: float
    benefit: str
    usage: str
    ingredients: List[str] = []
    confidence: float = 0.0

# Cache de produtos
class ProductCache:
    def __init__(self):
        self.ingredients_to_products: Dict[str, List[Product]] = {}
        self.product_name_to_data: Dict[str, Product] = {}
        self.frequently_asked_products: Set[str] = set()

    def add_product(self, product: Product, ingredients: List[str]):
        if product.product_name:
            self.product_name_to_data[product.product_name.lower()] = product
            for ingredient in ingredients:
                if ingredient not in self.ingredients_to_products:
                    self.ingredients_to_products[ingredient] = []
                self.ingredients_to_products[ingredient].append(product)

    def get_products_by_ingredient(self, ingredient: str) -> List[Product]:
        if ingredient is None:
            return []
        normalized_ingredient = ingredient.lower()
        for main_ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
            if normalized_ingredient in synonyms:
                normalized_ingredient = main_ingredient
                break
        return self.ingredients_to_products.get(normalized_ingredient, [])

    def get_product_by_name(self, name: str) -> Optional[Product]:
        if name is None:
            return None
        return self.product_name_to_data.get(name.lower())

# Inicializar cache global
product_cache = ProductCache()

# Conexão com o banco de dados
def connect_mysql():
    try:
        db = mysql.connector.connect(
            host="autorack.proxy.rlwy.net",
            user="root",
            password="AGWseadASVhFzAaAlxmLBoYBzgvBQhVT",
            database="railway",
            port=16717
        )
        return db
    except mysql.connector.Error as err:
        print(f"Erro ao conectar ao banco de dados: {err}")
        raise HTTPException(status_code=500, detail="Erro ao conectar ao banco de dados")

# Buscar produtos com cache
@lru_cache(maxsize=1)
def get_products():
    try:
        db = connect_mysql()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM produto")
        products = cursor.fetchall()
        cursor.close()
        db.close()
        return products
    except Exception as e:
        print(f"Erro ao buscar produtos: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar produtos")

# Criar índice FAISS otimizado
def create_faiss_index():
    products = get_products()
    if not products:
        return None, []

    texts = [
        f"{p.get('product_name', '')} {p.get('product_description', '')}" 
        for p in products
    ]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Usar IndexIVFFlat para melhor desempenho
    nlist = 100  # Número de clusters
    quantizer = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
    index.train(embeddings)
    index.add(embeddings)

    return index, products

# Extrair lista de itens da pergunta do usuário
def extract_item_list(question: str) -> List[str]:
    # Padrão para identificar listas de itens
    pattern = r"(?:quero|vou querer|preciso de|gostaria de)\s+([\w\s,]+)"
    matches = re.findall(pattern, question.lower())
    if matches:
        items = matches[0].split(",")
        return [item.strip() for item in items]
    return []

# Calcular o total dos produtos solicitados
def calculate_total(items: List[str], products: List[Dict]) -> float:
    total = 0.0
    for item in items:
        for product in products:
            if item.lower() in product["product_name"].lower():
                total += product["product_price"]
                break
    return total

# Extrair informações do cliente
def extract_client_info(question: str) -> Dict[str, Optional[str]]:
    client_info = {
        "age": None,
        "gender": None,
        "health_conditions": [],
        "fitness_goals": []
    }

    # Extrair idade
    age_match = re.search(r"\b(\d+)\s*anos\b", question)
    if age_match:
        client_info["age"] = int(age_match.group(1))

    # Extrair gênero
    if re.search(r"\b(homem|masculino)\b", question, re.IGNORECASE):
        client_info["gender"] = "masculino"
    elif re.search(r"\b(mulher|feminino)\b", question, re.IGNORECASE):
        client_info["gender"] = "feminino"

    # Extrair condições de saúde
    health_conditions = [
        "dor nas articulações", "dor muscular", "fadiga", "insônia", "ansiedade", "estresse",
        "recuperação muscular", "ganho de massa", "perda de peso", "saúde cardiovascular",
        "imunidade", "digestão", "energia", "foco", "performance atlética", "memória",
        "menopausa", "tpm", "tensão pré-menstrual", "regulação hormonal", "colesterol",
        "pressão alta", "diabetes", "glicemia", "inflamação", "alergias", "retenção de líquido",
        "celulite", "pele", "cabelo", "unhas", "saúde óssea", "menopausa", "andropausa",
        "libido", "disposição", "cansaço", "fraqueza", "imunidade baixa", "intestino",
        "flora intestinal", "ressaca", "dor de cabeça", "enxaqueca", "sinusite", "congestão"
    ]
    for condition in health_conditions:
        if re.search(fr"\b{re.escape(condition)}\b", question.lower()):
            client_info["health_conditions"].append(condition)

    # Extrair objetivos de fitness
    fitness_goals = [
        "ganho de massa", "hipertrofia", "definição muscular", "perda de peso", "emagrecimento",
        "resistência", "desempenho", "performance", "força", "recovery", "recuperação",
        "energia", "disposição", "foco", "concentração", "bem-estar", "qualidade de vida",
        "saúde", "vitalidade", "tonificação", "queima de gordura", "metabolismo"
    ]
    for goal in fitness_goals:
        if re.search(fr"\b{re.escape(goal)}\b", question.lower()):
            client_info["fitness_goals"].append(goal)

    return client_info

# Endpoint principal
@app.get("/chat", response_model=ChatResponse)
async def chat_endpoint(question: str = Query(..., min_length=3)):
    try:
        # Buscar produtos e criar índice FAISS
        index, products = create_faiss_index()
        if not index or not products:
            raise HTTPException(status_code=500, detail="Erro ao carregar produtos")

        # Extrair lista de itens da pergunta
        items = extract_item_list(question)
        total = calculate_total(items, products) if items else None

        # Extrair informações do cliente
        client_info = extract_client_info(question)

        # Buscar produtos relevantes
        relevant_products = get_relevant_products(question, index, products)

        # Preparar recomendações
        recommended_products = prepare_recommended_products(relevant_products[:5], question, client_info)

        # Gerar resposta com Gemini API
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Pergunta do cliente: {question}
        
        Produtos solicitados: {", ".join(items) if items else "Nenhum"}
        Total: R$ {total:.2f}
        
        Por favor, gere uma resposta amigável e profissional que:
        1. Responda à pergunta do cliente
        2. Mencione os produtos recomendados de forma natural
        3. Explique brevemente por que cada produto é adequado
        4. Inclua informações sobre uso quando relevante
        5. Mantenha um tom consultivo e não muito comercial
        """
        response = model.generate_content(prompt)

        return ChatResponse(
            resposta=response.text,
            produtos_relacionados=recommended_products[:5],
            total=total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Executar o servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)













