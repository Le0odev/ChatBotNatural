import mysql.connector
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google import genai
import re
from fastapi.middleware.cors import CORSMiddleware

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
client = genai.Client(api_key="AIzaSyBWpMyp95TwrPPUxHBp0OWQGYNcs-UlS8M")

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
    "Encapsulados": ["Cafeína", "Thermo", "Termogênico", "Emagrecedor", "Metabolismo"],
    "Óleos": ["Óleo", "Ômega 3", "Óleo de Coco", "Óleo de Peixe", "Gorduras Boas"],
    "Cereais": ["Cereal", "Grãos", "Aveia", "Granola", "Fibras"],
    "Energéticos": ["Energel", "Carb Up", "Gel Energético", "Maltodextrina", "Carboidratos"],
    "Temperos": ["Tempero", "Condimento", "Especiarias", "Sabor"],
    "Chá": ["Chá", "Infusão", "Erva", "Bebida Funcional"],
    "Chips": ["Chips", "Snack", "Petisco", "Lanche Saudável"]
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

class ChatResponse(BaseModel):
    resposta: str
    produtos_relacionados: List[Product]

def connect_mysql():
    return mysql.connector.connect(
        host="autorack.proxy.rlwy.net",
        user="root",
        password="AGWseadASVhFzAaAlxmLBoYBzgvBQhVT",
        database="railway",
        port=16717
    )

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
    cursor.close()
    db.close()
    return products

def create_faiss_index():
    products = get_products()

    if not products:
        return None, []

    texts = [
        f"{p['product_name']} {p['product_name']} {p['product_name']} - {p['categoria']} {p['categoria']} - {p['product_description']}" 
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

def get_relevant_products(question, index, products, k=20):
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

    # Filtrar produtos não desejados
    unwanted_brands = extract_unwanted_brands(question)
    filtered_products = [p for p in relevant_products if not any(brand.lower() in p['product_name'].lower() for brand in unwanted_brands)]

    products_context = "\n".join([
        f"- {p['product_name']} ({p['categoria']}): {p['product_description'][:100]}... - R$ {p['product_price']:.2f}"
        for p in filtered_products[:5]
    ])

    prompt = f"""
    Você é um especialista em suplementação nutricional. Forneça uma recomendação precisa e relevante baseada APENAS nos produtos listados abaixo.

    PERGUNTA DO CLIENTE: {question}

    PERFIL DO CLIENTE:
    {client_info}

    PRODUTOS DISPONÍVEIS:
    {products_context}

    INSTRUÇÕES:
    1. Analise cuidadosamente a pergunta do cliente e os produtos disponíveis na lista acima.
    2. Recomende APENAS produtos que estejam na lista fornecida e que sejam relevantes para a pergunta do cliente.
    3. Se não houver produtos adequados na lista para a necessidade específica do cliente, informe isso claramente.
    4. Para cada produto recomendado (se houver), forneça:
       - Nome exato do produto
       - Benefício principal relacionado à necessidade do cliente (uma frase)
       - Modo de uso básico (uma frase)
       - Preço
    5. Se não houver produtos adequados, sugira que o cliente procure outras opções ou consulte um profissional.

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

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

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

