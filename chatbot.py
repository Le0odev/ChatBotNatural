import mysql.connector
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Set, Any
import google.generativeai as genai
import re
from functools import lru_cache
import logging
import jellyfish
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(title="Emporio Verde Graos API", 
              description="API para recomendação de produtos e chatbot inteligente",
              version="2.0.0")

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
    "maca peruana": ["maca peruana", "maca", "peruvian maca", "libido", "energia", "disposição", "desempenho sexual"],
    "própolis": ["própolis", "propolis", "imunidade", "anti-inflamatório", "garganta"],
    "spirulina": ["spirulina", "espirulina", "alga", "desintoxicação", "proteína vegetal", "antioxidante"],
    "fibras": ["fibras", "fibra", "fiber", "intestino", "digestão", "saciedade"],
    "colágeno tipo 2": ["colágeno tipo 2", "colageno tipo 2", "colágeno tipo ii", "colageno tipo ii", "colágeno para articulações"],
    "colágeno tipo 1": ["colágeno tipo 1", "colageno tipo 1", "colágeno tipo i", "colageno tipo i", "colágeno para pele"],
    "long jack": ["brocha, sexual, vaso dilatador, libido, saude sexual", "libido", "disposição", "disfunção", "impotencia", "vaso dilatação", "vaso dilatador"]
}

# Palavras-chave para identificar intenções do usuário
USER_INTENTIONS = {
    "compra": ["comprar", "adquirir", "quero", "preciso", "necessito", "onde encontro", "tem à venda", "disponível", "em estoque"],
    "informação": ["como funciona", "para que serve", "benefícios", "efeitos", "composição", "ingredientes", "o que é", "informações sobre"],
    "preço": ["quanto custa", "preço", "valor", "promoção", "desconto", "mais barato", "econômico", "custo-benefício"],
    "comparação": ["comparar", "diferença", "melhor", "versus", "ou", "vs", "qual escolher", "recomenda"],
    "recomendação": ["recomendar", "indicar", "sugerir", "melhor para", "ideal para", "bom para", "adequado para"],
    "modo_de_uso": ["como usar", "como tomar", "dosagem", "quantidade", "frequência", "administração", "modo de uso", "instruções"]
}

HEALTH_CONDITIONS = [
    "dor nas articulações", "dor muscular", "fadiga", "insônia", "ansiedade", "estresse",
    "recuperação muscular", "ganho de massa", "perda de peso", "saúde cardiovascular",
    "imunidade", "digestão", "energia", "foco", "performance atlética"
]

# Modelos de dados
class Product(BaseModel):
    id: int
    product_name: str
    product_description: str
    categoria: Optional[str]
    product_price: float
    ingredients: List[str] = []
    rating: Optional[float] = None
    
class FeedbackRequest(BaseModel):
    question: str
    response: str
    rating: int  # 1-5
    comment: Optional[str] = None
    
class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    resposta: str
    produtos_relacionados: List[Product]
    intent_detected: Optional[str] = None
    
class UserProfile(BaseModel):
    user_id: str
    preferences: Dict[str, Any] = {}
    health_conditions: List[str] = []
    purchase_history: List[Dict[str, Any]] = []
    query_history: List[Dict[str, Any]] = []
    
# Classe para gerenciar o cache de produtos
class ProductCache:
    def __init__(self):
        self.ingredients_to_products: Dict[str, List[Product]] = {}
        self.product_name_to_data: Dict[str, Product] = {}
        self.frequently_asked_products: Set[str] = set()
        self.product_name_variations: Dict[str, str] = {}  # Para correção ortográfica
        
    def add_product(self, product: Product, ingredients: List[str]):
        self.product_name_to_data[product.product_name.lower()] = product
        
        # Adicionar variações do nome do produto para correção ortográfica
        self._add_name_variations(product.product_name)
        
        for ingredient in ingredients:
            if ingredient not in self.ingredients_to_products:
                self.ingredients_to_products[ingredient] = []
            self.ingredients_to_products[ingredient].append(product)
    
    def _add_name_variations(self, product_name: str):
        # Adiciona o nome original
        self.product_name_variations[product_name.lower()] = product_name
        
        # Adiciona variações com erros comuns
        words = product_name.lower().split()
        for i, word in enumerate(words):
            if len(word) > 3:  # Apenas para palavras maiores
                # Variação com uma letra trocada
                for j in range(len(word)):
                    if j < len(word) - 1:
                        variation = word[:j] + word[j+1] + word[j] + word[j+2:]
                        variation_full = ' '.join(words[:i] + [variation] + words[i+1:])
                        self.product_name_variations[variation_full] = product_name
    
    def get_products_by_ingredient(self, ingredient: str) -> List[Product]:
        normalized_ingredient = ingredient.lower()
        for main_ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
            if normalized_ingredient in synonyms:
                normalized_ingredient = main_ingredient
                break
        return self.ingredients_to_products.get(normalized_ingredient, [])
    
    def get_product_by_name(self, name: str) -> Optional[Product]:
        # Tenta encontrar o produto diretamente
        product = self.product_name_to_data.get(name.lower())
        if product:
            return product
            
        # Se não encontrar, tenta correção ortográfica
        corrected_name = self.correct_product_name(name)
        if corrected_name and corrected_name != name:
            return self.product_name_to_data.get(corrected_name.lower())
            
        return None
    
    def correct_product_name(self, name: str) -> Optional[str]:
        """Corrige erros ortográficos no nome do produto"""
        name_lower = name.lower()
        
        # Verifica se é uma variação conhecida
        if name_lower in self.product_name_variations:
            return self.product_name_variations[name_lower]
            
        # Tenta encontrar o produto mais similar
        best_match = None
        best_score = 0
        
        for product_name in self.product_name_to_data.keys():
            score = jellyfish.jaro_winkler_similarity(name_lower, product_name)
            if score > 0.85 and score > best_score:  # Threshold de similaridade
                best_score = score
                best_match = product_name
                
        return self.product_name_to_data[best_match].product_name if best_match else None
    
    def add_frequently_asked_product(self, product_key: str):
        self.frequently_asked_products.add(product_key.lower())
    
    def is_frequently_asked(self, product_key: str) -> bool:
        return product_key.lower() in self.frequently_asked_products
        
    def get_similar_products(self, product: Product, max_count: int = 3) -> List[Product]:
        """Retorna produtos similares baseados na categoria e ingredientes"""
        if not product:
            return []
            
        category_products = [p for p in self.product_name_to_data.values() 
                            if p.categoria == product.categoria and p.id != product.id]
        
        # Ordena por similaridade de ingredientes
        scored_products = []
        for p in category_products:
            common_ingredients = set(p.ingredients).intersection(set(product.ingredients))
            score = len(common_ingredients) / max(len(product.ingredients), 1)
            scored_products.append((p, score))
            
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored_products[:max_count]]

# Inicializar o cache global
product_cache = ProductCache()

# Classe para gerenciar perfis de usuário e histórico
class UserProfileManager:
    def __init__(self):
        self.profiles: Dict[str, UserProfile] = {}
        self.feedback_history: List[Dict[str, Any]] = []
        
    def get_profile(self, user_id: str) -> UserProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        return self.profiles[user_id]
        
    def update_query_history(self, user_id: str, question: str, detected_intent: str, 
                            ingredients: List[str], products: List[Product]):
        if not user_id:
            return
            
        profile = self.get_profile(user_id)
        profile.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "intent": detected_intent,
            "ingredients": ingredients,
            "products_shown": [p.id for p in products[:3]]
        })
        
        # Atualiza preferências baseadas nas consultas
        for ingredient in ingredients:
            if ingredient in profile.preferences:
                profile.preferences[ingredient] += 1
            else:
                profile.preferences[ingredient] = 1
                
    def add_feedback(self, feedback: FeedbackRequest, user_id: Optional[str] = None):
        feedback_entry = feedback.dict()
        feedback_entry["timestamp"] = datetime.now().isoformat()
        feedback_entry["user_id"] = user_id
        
        self.feedback_history.append(feedback_entry)
        
        # Usar feedback para melhorar o sistema
        if feedback.rating <= 2:  # Feedback negativo
            logger.info(f"Feedback negativo recebido: {feedback.comment}")
            # Aqui poderia implementar lógica para ajustar o sistema com base no feedback negativo
            
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        if not user_id or user_id not in self.profiles:
            return {}
            
        profile = self.profiles[user_id]
        
        # Encontra os ingredientes mais consultados
        top_ingredients = sorted(
            profile.preferences.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "top_ingredients": [item[0] for item in top_ingredients],
            "health_conditions": profile.health_conditions,
            "query_count": len(profile.query_history)
        }

# Inicializar o gerenciador de perfis
user_profile_manager = UserProfileManager()

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
            ingredients=product.get('ingredients', []),
            rating=product.get('rating', 4.0)
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
    
    # Extrai condições de saúde mencionadas
    health_conditions = [cond for cond in HEALTH_CONDITIONS if cond in question.lower()]
    
    # Extrai informações de gênero
    gender = None
    if re.search(r'\b(homem|masculino|cara|rapaz|garoto|menino)\b', question.lower()):
        gender = "masculino"
    elif re.search(r'\b(mulher|feminino|moça|garota|menina)\b', question.lower()):
        gender = "feminino"
        
    # Extrai informações de atividade física
    activity_level = None
    if re.search(r'\b(sedentário|não pratico|não faço exercício)\b', question.lower()):
        activity_level = "sedentário"
    elif re.search(r'\b(treino|academia|exercício|musculação|corrida|esporte)\b', question.lower()):
        activity_level = "ativo"
        
    return {
        "age": age,
        "gender": gender,
        "activity_level": activity_level,
        "health_conditions": health_conditions
    }

def extract_ingredient_queries(question):
    ingredient_queries = []
    
    # Padrões para identificar consultas sobre ingredientes
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
        
    # Verifica se algum ingrediente conhecido está na pergunta
    for ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
        if any(syn in question.lower() for syn in synonyms):
            ingredient_queries.append(ingredient)
            
    return list(set(ingredient_queries))

def detect_user_intention(question):
    """Detecta a intenção principal do usuário na pergunta"""
    question_lower = question.lower()
    
    # Conta ocorrências de palavras-chave para cada intenção
    intention_scores = {}
    for intention, keywords in USER_INTENTIONS.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        intention_scores[intention] = score
        
    # Determina a intenção com maior pontuação
    max_score = max(intention_scores.values()) if intention_scores else 0
    if max_score == 0:
        return "informação"  # Intenção padrão
        
    # Em caso de empate, prioriza certas intenções
    max_intentions = [i for i, s in intention_scores.items() if s == max_score]
    if len(max_intentions) > 1:
        for priority in ["compra", "preço", "comparação", "recomendação"]:
            if priority in max_intentions:
                return priority
                
    return max(intention_scores, key=intention_scores.get)

def get_relevant_products(question, index, products, k=20, user_id=None):
    # Detecta a intenção do usuário
    user_intention = detect_user_intention(question)
    
    # Extrai consultas sobre ingredientes específicos
    ingredient_queries = extract_ingredient_queries(question)
    
    # Incorpora preferências do usuário se disponíveis
    user_preferences = {}
    if user_id:
        user_preferences = user_profile_manager.get_user_preferences(user_id)
    
    # Se a pergunta é sobre ingredientes específicos
    if ingredient_queries:
        ingredient_products = []
        for ingredient in ingredient_queries:
            ingredient_products.extend(product_cache.get_products_by_ingredient(ingredient))
            
        if ingredient_products:
            # Formata os produtos encontrados
            formatted_products = []
            for p in ingredient_products:
                if isinstance(p, Product):  # Check if it's already a Product instance
                    formatted_product = {
                        'id': p.id,
                        'product_name': p.product_name,
                        'product_description': p.product_description,
                        'categoria': p.categoria,
                        'product_price': p.product_price,
                        'ingredients': p.ingredients,
                    }
                else:  # If it's a dict
                    formatted_product = {
                        'id': p['id'],
                        'product_name': p['product_name'],
                        'product_description': p['product_description'],
                        'categoria': p['categoria'],
                        'product_price': p['product_price'],
                        'ingredients': p.get('ingredients', []),
                    }
                formatted_products.append(formatted_product)
                
            # Se não encontrou produtos suficientes, complementa com busca semântica
            if len(formatted_products) < k:
                semantic_products, _ = get_semantic_products(question, index, products, k - len(formatted_products))
                product_ids = {p['id'] for p in formatted_products}
                for p in semantic_products:
                    if p['id'] not in product_ids:
                        formatted_products.append(p)
                        product_ids.add(p['id'])
                        
            # Reordena produtos com base na intenção do usuário
            if user_intention == "preço":
                formatted_products.sort(key=lambda p: p['product_price'])
            elif user_intention == "recomendação" and user_preferences:
                # Prioriza produtos com ingredientes preferidos pelo usuário
                def preference_score(product):
                    return sum(1 for ing in product['ingredients'] 
                              if ing in user_preferences.get('top_ingredients', []))
                formatted_products.sort(key=preference_score, reverse=True)
                
            return formatted_products, user_intention
            
    # Busca semântica padrão
    semantic_products, _ = get_semantic_products(question, index, products, k)
    
    # Reordena produtos com base na intenção do usuário
    if user_intention == "preço":
        semantic_products.sort(key=lambda p: p['product_price'])
    elif user_intention == "recomendação" and user_preferences:
        def preference_score(product):
            return sum(1 for ing in product['ingredients'] 
                      if ing in user_preferences.get('top_ingredients', []))
        semantic_products.sort(key=preference_score, reverse=True)
        
    return semantic_products, user_intention

def get_semantic_products(question, index, products, k=20):
    # Codifica a pergunta para busca vetorial
    question_emb = model.encode([question])
    question_emb = np.array(question_emb, dtype=np.float32)
    
    # Busca os produtos mais similares
    _, indices = index.search(question_emb, k=k)
    
    # Ensure we only get valid indices and format products correctly
    relevant_products = []
    for i in indices[0]:
        if i < len(products):
            product = products[i]
            # Ensure the product has all required fields
            formatted_product = {
                'id': product['id'],
                'product_name': product['product_name'],
                'product_description': product['product_description'],
                'categoria': product['categoria'],
                'product_price': product['product_price'],
                'ingredients': product.get('ingredients', [])
            }
            relevant_products.append(formatted_product)
    
    # Identifica categorias mencionadas na pergunta
    question_categories = []
    for cat, keywords in CATEGORIES.items():
        if any(keyword.lower() in question.lower() for keyword in keywords):
            question_categories.append(cat)
            
    # Filtra por categoria se mencionada
    if question_categories:
        filtered_products = [p for p in relevant_products if p['categoria'] in question_categories]
        return filtered_products if filtered_products else relevant_products, question_categories
        
    return relevant_products, []

def extract_unwanted_brands(question):
    """Extrai marcas ou produtos que o usuário não deseja"""
    unwanted_keywords = ["não seja", "não quero", "exceto", "menos", "tirando", "fora"]
    for keyword in unwanted_keywords:
        if keyword in question.lower():
            unwanted = re.findall(fr"{keyword}\s+(.*?)(?:\.|,|$)", question.lower())
            if unwanted:
                return unwanted[0].split()
    return []

def compare_products(products):
    """Gera uma comparação estruturada entre produtos similares"""
    if not products or len(products) < 2:
        return ""
        
    # Agrupa produtos por categoria
    category_products = {}
    for product in products:
        category = product['categoria']
        if category not in category_products:
            category_products[category] = []
        category_products[category].append(product)
        
    # Gera comparação para cada categoria com múltiplos produtos
    comparisons = []
    for category, prods in category_products.items():
        if len(prods) < 2:
            continue
            
        # Ordena por preço
        prods.sort(key=lambda p: p['product_price'])
        
        comparison = f"**Comparação de {category}:**\n"
        for i, p in enumerate(prods[:3]):  # Limita a 3 produtos
            comparison += f"- **{p['product_name']}**: R$ {p['product_price']:.2f}"
            
            # Adiciona diferenciadores
            if i == 0:
                comparison += " (Melhor custo-benefício)"
            elif i == len(prods) - 1:
                comparison += " (Premium)"
                
            # Adiciona ingredientes exclusivos
            exclusive_ingredients = set(p['ingredients'])
            for other_p in prods:
                if other_p['id'] != p['id']:
                    exclusive_ingredients -= set(other_p['ingredients'])
                    
            if exclusive_ingredients:
                comparison += f" - Exclusivo: {', '.join(exclusive_ingredients)}"
                
            comparison += "\n"
            
        comparisons.append(comparison)
        
    return "\n".join(comparisons)

def generate_response(question, relevant_products, detected_intent, user_id=None):
    # Extrai informações do cliente
    client_info = extract_client_info(question)
    
    # Formata as informações do cliente
    client_info_text = f"Idade: {client_info['age'] if client_info['age'] else 'Não informada'}"
    if client_info['gender']:
        client_info_text += f"\nGênero: {client_info['gender']}"
    if client_info['activity_level']:
        client_info_text += f"\nNível de atividade: {client_info['activity_level']}"
    if client_info['health_conditions']:
        client_info_text += f"\nCondições de saúde: {', '.join(client_info['health_conditions'])}"
    
    # Extrai consultas sobre ingredientes específicos
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
    
    # Obter preferências do usuário se disponíveis
    user_preferences = {}
    if user_id:
        user_preferences = user_profile_manager.get_user_preferences(user_id)
        
    # Gerar comparação de produtos se a intenção for comparação
    comparison_text = ""
    if detected_intent == "comparação" and len(filtered_products) >= 2:
        comparison_text = compare_products(filtered_products)
    
    # Melhorar o contexto de produtos incluindo informações de ingredientes e disponibilidade
    products_context = "\n".join([
        f"- {p['product_name']} ({p['categoria']}): {p['product_description'][:100]}... - Ingredientes: {', '.join(p.get('ingredients', []))} - R$ {p['product_price']:.2f} - Avaliação: {p.get('rating', 'N/A')}/5"
        for p in filtered_products[:5]
    ])
    
    # Adapta o prompt com base na intenção detectada
    intent_specific_instructions = ""
    if detected_intent == "compra":
        intent_specific_instructions = """
        Foque em disponibilidade, preço e como adquirir os produtos.
        Destaque informações de formas de pagamento se mencionadas.
        """
    elif detected_intent == "preço":
        intent_specific_instructions = """
        Organize os produtos por preço, do mais barato ao mais caro.
        Destaque o melhor custo-benefício e eventuais promoções.
        """
    elif detected_intent == "comparação":
        intent_specific_instructions = """
        Compare diretamente os produtos, destacando diferenças em:
        - Composição e ingredientes
        - Preço e custo-benefício
        - Benefícios específicos
        - Público-alvo ideal
        """
    elif detected_intent == "recomendação":
        intent_specific_instructions = """
        Recomende produtos específicos para as necessidades do cliente.
        Explique por que cada produto é adequado para o caso específico.
        Considere idade, gênero, condições de saúde e objetivos mencionados.
        """
    elif detected_intent == "modo_de_uso":
        intent_specific_instructions = """
        Foque nas instruções de uso, dosagem e melhores momentos para consumo.
        Mencione precauções importantes e possíveis efeitos colaterais.
        """
    
    # Incorpora preferências do usuário no prompt se disponíveis
    user_preferences_text = ""
    if user_preferences and user_preferences.get('top_ingredients'):
        user_preferences_text = f"\nPreferências do usuário: {', '.join(user_preferences.get('top_ingredients', []))}"
    
    # Prompt aprimorado para o Gemini
    prompt = f"""
    Você é um especialista em suplementação nutricional da loja Emporio Verde Graos. 
    Forneça uma recomendação precisa e relevante baseada APENAS nos produtos listados abaixo.

    PERGUNTA DO CLIENTE: {question}

    INTENÇÃO DETECTADA: {detected_intent}

    PERFIL DO CLIENTE:
    {client_info_text}
    {ingredient_info}
    {user_preferences_text}

    PRODUTOS DISPONÍVEIS:
    {products_context}

    {comparison_text}

    INSTRUÇÕES:
    1. Analise a pergunta do cliente e responda diretamente à intenção detectada: {detected_intent}.
    2. Recomende APENAS produtos que estejam na lista fornecida e que sejam relevantes para a pergunta.
    3. Se não houver produtos adequados na lista para a necessidade específica do cliente, informe isso claramente.
    4. Para cada produto recomendado, forneça:
       - Nome exato do produto
       - Benefício principal relacionado à necessidade do cliente
       - Modo de uso básico
       - Preço exato
    5. Se o cliente pergunta sobre ingredientes específicos, verifique se esses ingredientes estão explicitamente listados nos produtos acima.
    
    {intent_specific_instructions}

    FORMATO DA RESPOSTA:
    - Saudação personalizada
    - Resposta direta à pergunta principal
    - Recomendações estruturadas em tópicos com marcadores
    - Informação sobre preço e modo de uso
    - Conclusão com sugestão de próximos passos

    Mantenha a resposta objetiva, precisa e focada nas necessidades do cliente.
    Use formatação com marcadores para facilitar a leitura.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name
        response = model.generate_content(prompt)
        
        # Atualizar histórico de consultas com ingredientes específicos
        if user_id and ingredient_queries:
            user_profile_manager.update_query_history(
                user_id, 
                question, 
                detected_intent, 
                ingredient_queries, 
                [Product(**p) for p in filtered_products[:3]]
            )
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Return a fallback response
        return "Desculpe, estamos enfrentando dificuldades técnicas. Por favor, tente novamente mais tarde ou entre em contato com nosso suporte."

# Create FAISS index and load products on startup
index, products = create_faiss_index()

# POST endpoint for chat
@app.post("/chat", response_model=ChatResponse)
async def chat_post(request: ChatRequest):
    if index is None:
        raise HTTPException(status_code=500, detail="Sistema não inicializado corretamente.")
    
    relevant_products, detected_intent = get_relevant_products(
        request.question, 
        index, 
        products, 
        k=20, 
        user_id=request.user_id
    )
    
    generated_response = generate_response(
        request.question, 
        relevant_products, 
        detected_intent, 
        user_id=request.user_id
    )
    
    return ChatResponse(
        resposta=generated_response,
        produtos_relacionados=[Product(**p) for p in relevant_products[:3]],
        intent_detected=detected_intent
    )

# GET endpoint for chat
@app.get("/chat", response_model=ChatResponse)
async def chat_get(
    question: str = Query(..., description="Pergunta do usuário"),
    user_id: Optional[str] = Query(None, description="ID do usuário (opcional)"),
    session_id: Optional[str] = Query(None, description="ID da sessão (opcional)")
):
    if index is None:
        raise HTTPException(status_code=500, detail="Sistema não inicializado corretamente.")
    
    relevant_products, detected_intent = get_relevant_products(
        question, 
        index, 
        products, 
        k=20, 
        user_id=user_id
    )
    
    generated_response = generate_response(
        question,
        relevant_products,
        detected_intent,
        user_id=user_id
    )
    
    return ChatResponse(
        resposta=generated_response,
        produtos_relacionados=[Product(**p) for p in relevant_products[:3]],
        intent_detected=detected_intent
    )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, user_id: Optional[str] = None):
    user_profile_manager.add_feedback(feedback, user_id)
    return {"status": "success", "message": "Feedback recebido com sucesso"}

@app.get("/product/correct")
async def correct_product_name(name: str = Query(..., description="Nome do produto com possíveis erros ortográficos")):
    corrected = product_cache.correct_product_name(name)
    return {"original": name, "corrected": corrected}

@app.get("/product/similar/{product_id}")
async def get_similar_products(product_id: int):
    # Encontra o produto pelo ID
    product = None
    for p in products:
        if p['id'] == product_id:
            product = Product(**p)
            break
            
    if not product:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
        
    # Obtém produtos similares
    similar_products = product_cache.get_similar_products(product)
    
    return {"product": product, "similar_products": similar_products}

@app.get("/user/profile/{user_id}")
async def get_user_profile(user_id: str):
    profile = user_profile_manager.get_profile(user_id)
    return profile

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)