import mysql.connector
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
import google.generativeai as genai
import re
from fastapi.middleware.cors import CORSMiddleware
import json
from functools import lru_cache

# Initialize FastAPI
app = FastAPI()

# Adicionar configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://emporioverdegraos.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API configuration
genai.configure(api_key="AIzaSyBWpMyp95TwrPPUxHBp0OWQGYNcs-UlS8M")

# Embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Categories for context
CATEGORIES = {
    "Suplementos": ["Whey", "BCAA", "Glutamina", "Proteína", "Aminoácidos", "Suplementação"],
    "Barrinhas": ["Protein Bar", "Top Whey Bar", "Crisp", "Barra de Proteína", "Snack Proteico"],
    "Amendoins": ["Pasta de Amendoim", "Amendoim", "Manteiga de Amendoim", "Nuts"],
    "Pré-treino": ["Nuclear Rush", "Évora PW", "Horus", "C4", "Pré-Workout", "Energia"],
    "Creatina": ["Creatina", "Creapure", "Monohidratada", "Força"],
    "Vitaminas/Minerais": ["ZMA", "Vitamina", "Multivitamínico", "Mineral", "Micronutrientes"],
    "Encapsulados": ["Cafeína", "Thermo", "Termogênico", "Emagrecedor", "Metabolismo", "Valeriana", "Passiflora"],
    "Óleos": ["Óleo", "Ômega 3", "Óleo de Coco", "Óleo de Peixe", "Gorduras Boas"],
    "Cereais": ["Cereal", "Grãos", "Aveia", "Granola", "Fibras"],
    "Energéticos": ["Energel", "Carb Up", "Gel Energético", "Maltodextrina", "Carboidratos"],
    "Temperos": ["Tempero", "Condimento", "Especiarias", "Sabor"],
    "Chá": ["Chá", "Infusão", "Erva", "Bebida Funcional"],
    "Chips": ["Chips", "Snack", "Petisco", "Lanche Saudável"]
}

# Mapeamento de sinônimos para ingredientes e produtos específicos
INGREDIENTS_SYNONYMS = {
    "valeriana": ["valeriana", "valerian"],
    "passiflora": ["passiflora", "maracujá", "passion flower"],
    "ansiedade": ["ansiedade", "anxiety", "estresse", "stress", "nervoso", "insônia"],
    "melatonina": ["melatonina", "sono", "sleep"],
    "whey": ["whey", "proteína", "protein", "suplemento proteico"],
    # Adicione mais sinônimos conforme necessário
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
    ingredients: List[str] = []  # Nova propriedade para ingredientes

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
        
        # Verificar sinônimos
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
    
    # Extrair ingredientes do nome e descrição dos produtos
    for product in products:
        product['ingredients'] = extract_ingredients(product)
        
    cursor.close()
    db.close()
    
    # Inicializar o cache de produtos
    update_product_cache(products)
    
    return products

def extract_ingredients(product):
    """Extrai ingredientes do nome e descrição do produto"""
    ingredients = []
    
    # Combinar nome e descrição para análise
    text = f"{product['product_name']} {product['product_description']}".lower()
    
    # Verificar ingredientes específicos
    for ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
        if any(syn in text for syn in synonyms):
            ingredients.append(ingredient)
    
    return ingredients

def update_product_cache(products):
    """Atualiza o cache de produtos com os dados mais recentes"""
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

    # Melhorar o texto para embeddings com mais contexto
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
    """Extrai consultas sobre ingredientes específicos da pergunta"""
    ingredient_queries = []
    
    # Padrões de perguntas comuns sobre ingredientes
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
    
    # Verificar ingredientes específicos na pergunta
    for ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
        if any(syn in question.lower() for syn in synonyms):
            ingredient_queries.append(ingredient)
    
    return list(set(ingredient_queries))  # Remover duplicatas

def get_relevant_products(question, index, products, k=20):
    # Extrair ingredientes específicos da pergunta
    ingredient_queries = extract_ingredient_queries(question)
    
    # Se houver perguntas específicas sobre ingredientes, priorizar esses produtos
    if ingredient_queries:
        ingredient_products = []
        for ingredient in ingredient_queries:
            ingredient_products.extend(product_cache.get_products_by_ingredient(ingredient))
            
        # Se encontrou produtos com esses ingredientes, retorná-los primeiro
        if ingredient_products:
            # Converter para formato compatível
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
            
            # Se ainda precisar de mais produtos, adicionar da busca semântica
            if len(formatted_products) < k:
                semantic_products = get_semantic_products(question, index, products, k - len(formatted_products))
                
                # Adicionar apenas produtos que não estão na lista de ingredientes
                product_ids = {p['id'] for p in formatted_products}
                for p in semantic_products:
                    if p['id'] not in product_ids:
                        formatted_products.append(p)
                        product_ids.add(p['id'])
                
            return formatted_products
    
    # Caso contrário, usar busca semântica normal
    return get_semantic_products(question, index, products, k)

def get_semantic_products(question, index, products, k=20):
    """Busca semântica usando FAISS"""
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
    if ingredient_queries:
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

    prompt = f"""
    Você é um especialista em suplementação nutricional. Forneça uma recomendação precisa e relevante baseada APENAS nos produtos listados abaixo.

    PERGUNTA DO CLIENTE: {question}

    PERFIL DO CLIENTE:
    {client_info}
    {ingredient_info}

    PRODUTOS DISPONÍVEIS:
    {products_context}

    HISTÓRICO DE CONSULTAS RELEVANTES:
    {get_relevant_query_history(ingredient_queries)}

    INSTRUÇÕES:
    1. Analise cuidadosamente a pergunta do cliente e os produtos disponíveis na lista acima.
    2. Recomende APENAS produtos que estejam na lista fornecida e que sejam relevantes para a pergunta do cliente.
    3. Se não houver produtos adequados na lista para a necessidade específica do cliente, informe isso claramente.
    4. IMPORTANTE: Se o cliente pergunta sobre ingredientes específicos (como Valeriana ou Passiflora), verifique se esses ingredientes estão explicitamente listados nos produtos acima.
    5. Seja consistente com o histórico de consultas. Se um produto foi informado como disponível anteriormente, mantenha essa consistência.
    6. Para cada produto recomendado (se houver), forneça:
       - Nome exato do produto
       - Benefício principal relacionado à necessidade do cliente (uma frase)
       - Modo de uso básico (uma frase)
       - Preço
    7. Se não houver produtos adequados, sugira que o cliente procure outras opções ou consulte um profissional.

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
    # Function to extract unwanted brands from the question
    unwanted_keywords = ["não seja", "não quero", "exceto", "menos"]
    for keyword in unwanted_keywords:
        if keyword in question.lower():
            # Extract the words after the keyword until the end of the sentence or punctuation
            unwanted = re.findall(fr"{keyword}\s+(.*?)(?:\.|,|$)", question.lower())
            if unwanted:
                return unwanted[0].split()
    return []

# Gerenciamento de histórico de consultas
query_history = {}

def update_query_history(ingredient_queries, available_products):
    """Atualiza o histórico de consultas com informações sobre disponibilidade de ingredientes"""
    for ingredient in ingredient_queries:
        # Normalizar o ingrediente
        normalized_ingredient = ingredient.lower()
        
        # Verificar sinônimos
        for main_ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
            if normalized_ingredient in synonyms:
                normalized_ingredient = main_ingredient
                break
        
        # Verificar se há produtos disponíveis com esse ingrediente
        ingredient_products = [p for p in available_products if 'ingredients' in p and normalized_ingredient in [i.lower() for i in p['ingredients']]]
        
        query_history[normalized_ingredient] = {
            "available": len(ingredient_products) > 0,
            "products": [
                {
                    "name": p['product_name'],
                    "price": p['product_price']
                } for p in ingredient_products[:3]  # Limitar a 3 produtos para o histórico
            ]
        }

def get_relevant_query_history(ingredient_queries):
    """Recupera o histórico de consultas relevante para os ingredientes da consulta atual"""
    if not ingredient_queries:
        return "Nenhum histórico relevante."
    
    history_entries = []
    
    for ingredient in ingredient_queries:
        # Normalizar o ingrediente
        normalized_ingredient = ingredient.lower()
        
        # Verificar sinônimos
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