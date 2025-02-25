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
from functools import lru_cache
import logging

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

# Mapeamento de sinônimos para ingredientes e produtos específicos
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

HEALTH_CONDITIONS = [
    "dor nas articulações", "dor muscular", "fadiga", "insônia", "ansiedade", "estresse",
    "recuperação muscular", "ganho de massa", "perda de peso", "saúde cardiovascular",
    "imunidade", "digestão", "energia", "foco", "performance atlética"
]

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

class ProductCache:
    def __init__(self):
        self.ingredients_to_products: Dict[str, List[Product]] = {}
        self.product_name_to_data: Dict[str, Product] = {}
        self.frequently_asked_products: Set[str] = set()
        
    def add_product(self, product: Product, ingredients: List[str]):
        self.product_name_to_data[product.product_name.lower()] = product
        for ingredient in ingredients:
            if ingredient not in self.ingredients_to_products:
                self.ingredients_to_products[ingredient] = []
            self.ingredients_to_products[ingredient].append(product)
    
    def get_products_by_ingredient(self, ingredient: str) -> List[Product]:
        normalized_ingredient = ingredient.lower()
        for main_ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
            if normalized_ingredient in synonyms:
                normalized_ingredient = main_ingredient
                break
        return self.ingredients_to_products.get(normalized_ingredient, [])
    
    def get_product_by_name(self, name: str) -> Optional[Product]:
        return self.product_name_to_data.get(name.lower())
    
    def add_frequently_asked_product(self, product_key: str):
        self.frequently_asked_products.add(product_key.lower())
    
    def is_frequently_asked(self, product_key: str) -> bool:
        return product_key.lower() in self.frequently_asked_products

# Inicializar o cache global
product_cache = ProductCache()

def connect_mysql():
    return mysql.connector.connect(
        host="autorack.proxy.rlwy.net",
        user="root",
        password="AGWseadASVhFzAaAlxmLBoYBzgvBQhVT",
        database="railway",
        port=16717
    )

@lru_cache(maxsize=1)
def get_products():
    db = connect_mysql()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT 
            p.id, 
            p.product_name, 
            p.product_description,
            c.category_name as categoria,
            p.product_price
        FROM produto p
        LEFT JOIN categoria c ON p.category_id = c.id
    """)
    products = cursor.fetchall()
    for product in products:
        product['ingredients'] = extract_ingredients(product)
    cursor.close()
    db.close()
    update_product_cache(products)
    return products

def extract_ingredients(product):
    ingredients = []
    text = f"{product['product_name']} {product['product_description']}".lower()
    for ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
        if any(syn in text for syn in synonyms):
            ingredients.append(ingredient)
    return ingredients

def update_product_cache(products):
    for product in products:
        product_obj = Product(
            id=product['id'],
            product_name=product['product_name'],
            product_description=product['product_description'],
            categoria=product['categoria'],
            product_price=product['product_price'],
            ingredients=product.get('ingredients', [])
        )
        product_cache.add_product(product_obj, product.get('ingredients', []))

def create_faiss_index():
    products = get_products()
    if not products:
        return None, []
    texts = [
        f"{p['product_name']} {p['product_name']} {p['product_name']} - {p['categoria']} {p['categoria']} - {' '.join(p.get('ingredients', []))} - {p['product_description']}" 
        for p in products
    ]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, products

def extract_client_info(question):
    age_match = re.search(r'\b(\d+)\s*anos\b', question)
    age = int(age_match.group(1)) if age_match else None
    health_conditions = [cond for cond in HEALTH_CONDITIONS if cond in question.lower()]
    return age, health_conditions

def extract_ingredient_queries(question):
    ingredient_queries = []
    patterns = [
        r"(?:tem|vende[m]?|dispõe[m]? de|possui|disponível)\s+(\w+)",
        r"produtos com\s+(\w+)",
        r"algum.+(?:com|contendo)\s+(\w+)",
        r"(\w+).+disponível",
        r"(\w+).+em estoque"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, question.lower())
        ingredient_queries.extend(matches)
    for ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
        if any(syn in question.lower() for syn in synonyms):
            ingredient_queries.append(ingredient)
    return list(set(ingredient_queries))

def get_relevant_products(question, index, products, k=20):
    ingredient_queries = extract_ingredient_queries(question)
    if ingredient_queries:
        ingredient_products = []
        for ingredient in ingredient_queries:
            ingredient_products.extend(product_cache.get_products_by_ingredient(ingredient))
        if ingredient_products:
            formatted_products = []
            for p in ingredient_products:
                formatted_product = {
                    'id': p.id,
                    'product_name': p.product_name,
                    'product_description': p.product_description,
                    'categoria': p.categoria,
                    'product_price': p.product_price,
                    'ingredients': p.ingredients
                }
                formatted_products.append(formatted_product)
            if len(formatted_products) < k:
                semantic_products = get_semantic_products(question, index, products, k - len(formatted_products))
                product_ids = {p['id'] for p in formatted_products}
                for p in semantic_products:
                    if p['id'] not in product_ids:
                        formatted_products.append(p)
                        product_ids.add(p['id'])
            return formatted_products
    return get_semantic_products(question, index, products, k)

def get_semantic_products(question, index, products, k=20):
    question_emb = model.encode([question])
    question_emb = np.array(question_emb, dtype=np.float32)
    _, indices = index.search(question_emb, k=k)
    relevant_products = [products[i] for i in indices[0] if i < len(products)]
    question_categories = []
    for cat, keywords in CATEGORIES.items():
        if any(keyword.lower() in question.lower() for keyword in keywords):
            question_categories.append(cat)
    if question_categories:
        filtered_products = [p for p in relevant_products if p['categoria'] in question_categories]
        return filtered_products if filtered_products else relevant_products
    return relevant_products

def generate_response(question, relevant_products):
    age, health_conditions = extract_client_info(question)
    client_info = f"Idade: {age if age else 'Não informada'}"
    client_info += f"\nObjetivos: {', '.join(health_conditions) if health_conditions else 'Não especificados'}"
    
    # Extrair consultas sobre ingredientes específicos
    ingredient_queries = extract_ingredient_queries(question)
    ingredient_info = ""
    
    # Verificar disponibilidade de ingredientes específicos
    available_ingredients = []
    unavailable_ingredients = []
    for ingredient in ingredient_queries:
        products = product_cache.get_products_by_ingredient(ingredient)
        if products:
            available_ingredients.append(ingredient)
        else:
            unavailable_ingredients.append(ingredient)
    
    if available_ingredients:
        ingredient_info += f"\nIngredientes disponíveis: {', '.join(available_ingredients)}"
    if unavailable_ingredients:
        ingredient_info += f"\nIngredientes não disponíveis: {', '.join(unavailable_ingredients)}"
    
    # Filtrar produtos não desejados
    unwanted_brands = extract_unwanted_brands(question)
    filtered_products = [p for p in relevant_products if not any(brand.lower() in p['product_name'].lower() for brand in unwanted_brands)]
    
    # Melhorar o contexto de produtos incluindo informações de ingredientes
    products_context = "\n".join([
        f"- {p['product_name']} ({p['categoria']}): {p['product_description'][:100]}... - Ingredientes: {', '.join(p.get('ingredients', []))} - R$ {p['product_price']:.2f}"
        for p in filtered_products[:5]
    ])
    
    # Prompt aprimorado para o Gemini
    prompt = f"""
    Você é um especialista em suplementação nutricional. Forneça uma recomendação precisa e relevante baseada APENAS nos produtos listados abaixo.

    PERGUNTA DO CLIENTE: {question}

    PERFIL DO CLIENTE:
    {client_info}
    {ingredient_info}

    PRODUTOS DISPONÍVEIS:
    {products_context}

    INSTRUÇÕES:
    1. Analise cuidadosamente a pergunta do cliente e os produtos disponíveis na lista acima.
    2. Recomende APENAS produtos que estejam na lista fornecida e que sejam relevantes para a pergunta do cliente.
    3. Se não houver produtos adequados na lista para a necessidade específica do cliente, informe isso claramente.
    4. IMPORTANTE: Se o cliente pergunta sobre ingredientes específicos (como Ômega 3, Colágeno ou Moringa), verifique se esses ingredientes estão explicitamente listados nos produtos acima.
    5. Para cada produto recomendado (se houver), forneça:
       - Nome exato do produto
       - Benefício principal relacionado à necessidade do cliente (uma frase)
       - Modo de uso básico (uma frase)
       - Preço
    6. Se não houver produtos adequados, sugira que o cliente procure outras opções ou consulte um profissional.

    FORMATO DA RESPOSTA:
    - Introdução: Uma frase abordando a necessidade específica do cliente.
    - Recomendações: Liste os produtos recomendados com suas informações (ou informe que não há produtos adequados).
    - Precaução: Uma frase sobre cuidados importantes, se aplicável.
    - Conclusão: Uma frase de encerramento ou sugestão adicional.

    IMPORTANTE: 
    - Responda APENAS com base nos produtos listados acima.
    - Não invente ou sugira produtos que não estejam na lista.
    - Seja honesto se não houver opções adequadas para a necessidade do cliente.
    - Mantenha a resposta objetiva, relevante e direta.
    - Use no máximo 4 parágrafos curtos.
    """
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    # Atualizar histórico de consultas com ingredientes específicos
    if ingredient_queries:
        update_query_history(ingredient_queries, filtered_products)
    
    return response.text

def extract_unwanted_brands(question):
    unwanted_keywords = ["não seja", "não quero", "exceto", "menos"]
    for keyword in unwanted_keywords:
        if keyword in question.lower():
            unwanted = re.findall(fr"{keyword}\s+(.*?)(?:\.|,|$)", question.lower())
            if unwanted:
                return unwanted[0].split()
    return []

# Gerenciamento de histórico de consultas
query_history = {}

def update_query_history(ingredient_queries, available_products):
    for ingredient in ingredient_queries:
        normalized_ingredient = ingredient.lower()
        for main_ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
            if normalized_ingredient in synonyms:
                normalized_ingredient = main_ingredient
                break
        ingredient_products = [p for p in available_products if 'ingredients' in p and normalized_ingredient in [i.lower() for i in p['ingredients']]]
        query_history[normalized_ingredient] = {
            "available": len(ingredient_products) > 0,
            "products": [
                {
                    "name": p['product_name'],
                    "price": p['product_price']
                } for p in ingredient_products[:3]
            ]
        }

def get_relevant_query_history(ingredient_queries):
    if not ingredient_queries:
        return "Nenhum histórico relevante."
    history_entries = []
    for ingredient in ingredient_queries:
        normalized_ingredient = ingredient.lower()
        for main_ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
            if normalized_ingredient in synonyms:
                normalized_ingredient = main_ingredient
                break
        if normalized_ingredient in query_history:
            entry = query_history[normalized_ingredient]
            if entry["available"]:
                products_info = ", ".join([f"{p['name']} (R$ {p['price']:.2f})" for p in entry["products"]])
                history_entries.append(f"Já informado que temos produtos com {normalized_ingredient}: {products_info}")
            else:
                history_entries.append(f"Já informado que NÃO temos produtos com {normalized_ingredient}")
    if history_entries:
        return "\n".join(history_entries)
    else:
        return "Nenhum histórico relevante para esses ingredientes."

# Create FAISS index and load products on startup
index, products = create_faiss_index()

@app.get("/chat", response_model=ChatResponse)
async def answer_question(question: str = Query(..., min_length=1, description="The question to ask about products")):
    if index is None:
        raise HTTPException(status_code=500, detail="O índice FAISS ainda não foi criado.")
    relevant_products = get_relevant_products(question, index, products, k=20)
    generated_response = generate_response(question, relevant_products)
    return ChatResponse(
        resposta=generated_response,
        produtos_relacionados=[Product(**p) for p in relevant_products[:3]]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)