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
              version="2.1.0")

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

# Modelo de embeddings
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
    "pré treino": ["pré treino", "pre treino", "pre-treino", "pré-treino", "horus", "égide", "c4"],
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
    "long jack": ["long jack", "sexual", "vaso dilatador", "libido", "saude sexual", "disposição", "disfunção", "impotencia"]
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
    preferences: Dict[str, int] = {}
    health_conditions: List[str] = []
    purchase_history: List[Dict[str, Any]] = []
    query_history: List[Dict[str, Any]] = []

# Classe para gerenciar o cache de produtos
class ProductCache:
    def __init__(self):
        self.ingredients_to_products: Dict[str, List[Product]] = {}
        self.product_name_to_data: Dict[str, Product] = {}
        self.frequently_asked_products: Set[str] = set()
        self.product_name_variations: Dict[str, str] = {}
        
    def add_product(self, product: Product, ingredients: List[str]):
        self.product_name_to_data[product.product_name.lower()] = product
        self._add_name_variations(product.product_name)
        
        for ingredient in ingredients:
            if ingredient not in self.ingredients_to_products:
                self.ingredients_to_products[ingredient] = []
            self.ingredients_to_products[ingredient].append(product)
    
    def _add_name_variations(self, product_name: str):
        self.product_name_variations[product_name.lower()] = product_name
        words = product_name.lower().split()
        for i, word in enumerate(words):
            if len(word) > 3:
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
        product = self.product_name_to_data.get(name.lower())
        if product:
            return product
        corrected_name = self.correct_product_name(name)
        if corrected_name and corrected_name != name:
            return self.product_name_to_data.get(corrected_name.lower())
        return None
    
    def correct_product_name(self, name: str) -> Optional[str]:
        name_lower = name.lower()
        if name_lower in self.product_name_variations:
            return self.product_name_variations[name_lower]
        best_match = None
        best_score = 0
        for product_name in self.product_name_to_data.keys():
            score = jellyfish.jaro_winkler_similarity(name_lower, product_name)
            if score > 0.9 and score > best_score:  # Aumentado para 0.9 para maior precisão
                best_score = score
                best_match = product_name
        return self.product_name_to_data[best_match].product_name if best_match else None
    
    def add_frequently_asked_product(self, product_key: str):
        self.frequently_asked_products.add(product_key.lower())
    
    def is_frequently_asked(self, product_key: str) -> bool:
        return product_key.lower() in self.frequently_asked_products
        
    def get_similar_products(self, product: Product, max_count: int = 3) -> List[Product]:
        if not product:
            return []
        category_products = [p for p in self.product_name_to_data.values() 
                            if p.categoria == product.categoria and p.id != product.id]
        scored_products = []
        for p in category_products:
            common_ingredients = set(p.ingredients).intersection(set(product.ingredients))
            score = len(common_ingredients) / max(len(product.ingredients), 1)
            if p.rating:
                score += p.rating * 0.1  # Bonus por rating
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
        for ingredient in ingredients:
            profile.preferences[ingredient] = profile.preferences.get(ingredient, 0) + 1
                
    def add_feedback(self, feedback: FeedbackRequest, user_id: Optional[str] = None):
        feedback_entry = feedback.dict()
        feedback_entry["timestamp"] = datetime.now().isoformat()
        feedback_entry["user_id"] = user_id
        self.feedback_history.append(feedback_entry)
        if feedback.rating <= 2:
            logger.info(f"Feedback negativo recebido: {feedback.comment}")
            if user_id:
                profile = self.get_profile(user_id)
                profile.preferences["negative_feedback_count"] = profile.preferences.get("negative_feedback_count", 0) + 1
                
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        if not user_id or user_id not in self.profiles:
            return {}
        profile = self.profiles[user_id]
        top_ingredients = sorted(
            [(k, v) for k, v in profile.preferences.items() if k != "negative_feedback_count"], 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        return {
            "top_ingredients": [item[0] for item in top_ingredients],
            "health_conditions": profile.health_conditions,
            "query_count": len(profile.query_history),
            "negative_feedback_count": profile.preferences.get("negative_feedback_count", 0)
        }

# Inicializar o gerenciador de perfis
user_profile_manager = UserProfileManager()

def connect_mysql():
    try:
        return mysql.connector.connect(
            host="autorack.proxy.rlwy.net",
            user="root",
            password="AGWseadASVhFzAaAlxmLBoYBzgvBQhVT",
            database="railway",
            port=16717
        )
    except mysql.connector.Error as e:
        logger.error(f"Erro ao conectar ao MySQL: {e}")
        raise

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
        logger.warning("Nenhum produto encontrado ao criar índice FAISS.")
        return None, []
    texts = [
        f"{p['product_name']} {p['categoria']} {' '.join(p.get('ingredients', []))} {p['product_description']}" 
        for p in products
    ]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info(f"Índice FAISS criado com {len(products)} produtos.")
    return index, products

def extract_client_info(question):
    age_match = re.search(r'\b(\d+)\s*anos\b', question, re.IGNORECASE)
    age = int(age_match.group(1)) if age_match else None
    health_conditions = [cond for cond in HEALTH_CONDITIONS if cond in question.lower()]
    gender = None
    if re.search(r'\b(homem|masculino|cara|rapaz|garoto|menino)\b', question.lower()):
        gender = "masculino"
    elif re.search(r'\b(mulher|feminino|moça|garota|menina)\b', question.lower()):
        gender = "feminino"
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

def extract_product_names(question):
    product_names = []
    question_lower = question.lower()
    for product_name in product_cache.product_name_to_data.keys():
        if product_name.lower() in question_lower:
            product_names.append(product_name)
    return product_names

def extract_ingredient_queries(question):
    ingredient_queries = []
    patterns = [
        r"(?:tem|vende[m]?|dispõe[m]? de|possui|disponível)\s+([\w\s]+)",
        r"produtos com\s+([\w\s]+)",
        r"algum.+(?:com|contendo)\s+([\w\s]+)",
        r"([\w\s]+).+disponível",
        r"([\w\s]+).+em estoque"
    ]
    question_lower = question.lower()
    for pattern in patterns:
        matches = re.findall(pattern, question_lower)
        ingredient_queries.extend([match.strip() for match in matches])
    for ingredient, synonyms in INGREDIENTS_SYNONYMS.items():
        if any(syn in question_lower for syn in synonyms):
            ingredient_queries.append(ingredient)
    return list(set(ingredient_queries))

def detect_user_intention(question):
    question_lower = question.lower()
    intention_scores = {intent: 0 for intent in USER_INTENTIONS}
    for intention, keywords in USER_INTENTIONS.items():
        intention_scores[intention] = sum(1 for keyword in keywords if keyword in question_lower)
    
    product_count = sum(1 for product_name in product_cache.product_name_to_data.keys() if product_name.lower() in question_lower)
    category_count = sum(1 for cat, keywords in CATEGORIES.items() if any(kw in question_lower for kw in keywords))
    ingredient_count = len(extract_ingredient_queries(question))
    
    max_score = max(intention_scores.values()) if intention_scores else 0
    if max_score == 0:
        return "informação" if ingredient_count == 0 else "recomendação"
    
    max_intentions = [i for i, s in intention_scores.items() if s == max_score]
    if (product_count > 1 or category_count > 1 or ingredient_count > 1) and "comparação" not in max_intentions:
        return "multi_produto"
    if len(max_intentions) > 1:
        for priority in ["compra", "preço", "comparação", "recomendação", "modo_de_uso"]:
            if priority in max_intentions:
                return priority
    return max(intention_scores, key=intention_scores.get)

def get_semantic_products(question, index, products, k=20):
    question_emb = model.encode([question])
    question_emb = np.array(question_emb, dtype=np.float32)
    distances, indices = index.search(question_emb, k=k * 2)  # Busca mais candidatos
    relevant_products = []
    seen_categories = set()
    category_priority = []

    # Detectar categorias mencionadas
    question_lower = question.lower()
    for cat, keywords in CATEGORIES.items():
        if any(keyword.lower() in question_lower for keyword in keywords):
            category_priority.append(cat)

    # Priorizar produtos das categorias mencionadas
    for i in indices[0]:
        if i < len(products):
            product = products[i]
            formatted_product = Product(
                id=product['id'],
                product_name=product['product_name'],
                product_description=product['product_description'],
                categoria=product['categoria'],
                product_price=product['product_price'],
                ingredients=product.get('ingredients', []),
                rating=product.get('rating', None)
            )
            if formatted_product.categoria in category_priority and formatted_product.categoria not in seen_categories:
                relevant_products.append(formatted_product)
                seen_categories.add(formatted_product.categoria)
            elif formatted_product.categoria not in seen_categories and len(relevant_products) < k:
                relevant_products.append(formatted_product)

    # Completar com os melhores matches semânticos
    for i in indices[0]:
        if i < len(products) and len(relevant_products) < k:
            product = products[i]
            formatted_product = Product(**{key: product.get(key) for key in Product.__fields__.keys()})
            if formatted_product not in relevant_products:
                relevant_products.append(formatted_product)

    return relevant_products[:k], category_priority

def get_relevant_products(question, index, products, k=20, user_id=None):
    user_intention = detect_user_intention(question)
    ingredient_queries = extract_ingredient_queries(question)
    product_names = extract_product_names(question)
    
    relevant_products = []
    seen_ids = set()

    # Priorizar produtos mencionados explicitamente
    if product_names:
        for product_name in product_names:
            product = product_cache.get_product_by_name(product_name)
            if product and product.id not in seen_ids:
                relevant_products.append(product)
                seen_ids.add(product.id)
    
    # Adicionar produtos por ingredientes mencionados
    for ingredient in ingredient_queries:
        prods = product_cache.get_products_by_ingredient(ingredient)
        for p in prods:
            if p.id not in seen_ids and len(relevant_products) < k:
                relevant_products.append(p)
                seen_ids.add(p.id)

    # Completar com produtos semânticos
    if len(relevant_products) < k:
        semantic_products, _ = get_semantic_products(question, index, products, k - len(relevant_products))
        relevant_products.extend([p for p in semantic_products if p.id not in seen_ids][:k - len(relevant_products)])

    # Ordenar com base na intenção e preferências do usuário
    if user_intention == "preço":
        relevant_products.sort(key=lambda p: p.product_price)
    elif user_intention == "recomendação" and user_id:
        user_preferences = user_profile_manager.get_user_preferences(user_id)
        def preference_score(product):
            score = sum(1 for ing in product.ingredients if ing in user_preferences.get('top_ingredients', []))
            return score + (product.rating or 4.0) * 0.2  # Peso extra para rating
        relevant_products.sort(key=preference_score, reverse=True)
    elif user_intention in ["multi_produto", "comparação"]:
        relevant_products.sort(key=lambda p: (p.categoria, --(p.rating or 4.0)))  # Agrupar por categoria, ordenar por rating descendente

    return relevant_products[:k], user_intention

def extract_unwanted_brands(question):
    unwanted_keywords = ["não seja", "não quero", "exceto", "menos", "tirando", "fora"]
    question_lower = question.lower()
    for keyword in unwanted_keywords:
        if keyword in question_lower:
            unwanted = re.findall(fr"{keyword}\s+([\w\s]+?)(?:\.|,|$)", question_lower)
            if unwanted:
                return [u.strip() for u in unwanted[0].split()]
    return []

def compare_products(products):
    if not products or len(products) < 2:
        return "Não há produtos suficientes para comparação."
    category_products = {}
    for product in products:
        category = product.categoria or "Outros"
        if category not in category_products:
            category_products[category] = []
        category_products[category].append(product)
    
    comparisons = []
    for category, prods in category_products.items():
        if len(prods) < 1:
            continue
        prods.sort(key=lambda p: p.product_price)
        comparison = f"### Comparação em {category}\n\n| Produto | Preço | Ingredientes Exclusivos | Avaliação |\n|---------|-------|-------------------------|-----------|\n"
        for p in prods[:3]:
            exclusive_ingredients = set(p.ingredients)
            for other_p in prods:
                if other_p.id != p.id:
                    exclusive_ingredients -= set(other_p.ingredients)
            comparison += f"| **{p.product_name}** | R$ {p.product_price:.2f} | {', '.join(exclusive_ingredients) or 'Nenhum'} | {p.rating or 'N/A'}/5 |\n"
        comparisons.append(comparison)
    return "\n".join(comparisons)

def generate_response(question, relevant_products, detected_intent, user_id=None):
    client_info = extract_client_info(question)
    client_info_text = f"- **Idade**: {client_info['age'] if client_info['age'] else 'Não informada'}\n" \
                      f"- **Gênero**: {client_info['gender'] or 'Não informado'}\n" \
                      f"- **Nível de atividade**: {client_info['activity_level'] or 'Não informado'}\n" \
                      f"- **Condições de saúde**: {', '.join(client_info['health_conditions']) or 'Nenhuma mencionada'}"
    
    ingredient_queries = extract_ingredient_queries(question)
    ingredient_info = ""
    available_ingredients = []
    unavailable_ingredients = []
    for ingredient in ingredient_queries:
        products = product_cache.get_products_by_ingredient(ingredient)
        if products:
            available_ingredients.append(ingredient)
        else:
            unavailable_ingredients.append(ingredient)
    if available_ingredients:
        ingredient_info += f"- **Ingredientes disponíveis**: {', '.join(available_ingredients)}\n"
    if unavailable_ingredients:
        ingredient_info += f"- **Ingredientes não encontrados**: {', '.join(unavailable_ingredients)}\n"
    
    unwanted_brands = extract_unwanted_brands(question)
    filtered_products = [p for p in relevant_products if not any(brand.lower() in p.product_name.lower() for brand in unwanted_brands)]
    
    user_preferences = user_profile_manager.get_user_preferences(user_id) if user_id else {}
    comparison_text = compare_products(filtered_products) if detected_intent == "comparação" and len(filtered_products) >= 1 else ""
    
    products_context = "\n".join([
        f"- **{p.product_name}** ({p.categoria}):\n  - Descrição: {p.product_description[:100]}...\n  - Ingredientes: {', '.join(p.ingredients) or 'Não especificado'}\n  - Preço: R$ {p.product_price:.2f}\n  - Avaliação: {p.rating or 'N/A'}/5"
        for p in filtered_products[:5]
    ]) or "Nenhum produto relevante encontrado."

    intent_specific_instructions = {
        "compra": "Foque em disponibilidade e preço. Inclua instruções claras sobre como comprar (ex.: 'Disponível em nossa loja online').",
        "preço": "Liste os produtos em ordem de preço crescente, destacando opções econômicas e custo-benefício.",
        "comparação": "Use a tabela de comparação fornecida para destacar diferenças entre produtos.",
        "recomendação": "Recomende produtos com base no perfil do cliente e explique os benefícios específicos para suas necessidades.",
        "modo_de_uso": "Forneça instruções detalhadas de uso, incluindo dosagem, horário ideal e possíveis precauções.",
        "multi_produto": "Agrupe os produtos por categoria ou nome mencionado, com uma descrição curta e relevante para cada."
    }.get(detected_intent, "Responda de forma informativa e geral, priorizando clareza.")
    
    user_preferences_text = f"- **Preferências do usuário**: {', '.join(user_preferences.get('top_ingredients', [])) or 'Nenhuma registrada'}" if user_id else ""
    
    prompt = f"""
    Você é um especialista em suplementação nutricional da loja Empório Verde Grãos, com tom profissional, amigável e preciso. 
    Responda à pergunta do cliente com base APENAS nos produtos listados abaixo, oferecendo uma resposta clara, bem formatada e adaptada à intenção detectada.

    ### Pergunta do Cliente
    {question}

    ### Intenção Detectada
    {detected_intent}

    ### Perfil do Cliente
    {client_info_text}
    {ingredient_info}
    {user_preferences_text}

    ### Produtos Disponíveis
    {products_context}

    ### Comparação (se aplicável)
    {comparison_text}

    ### Instruções
    1. Responda diretamente à pergunta do cliente conforme a intenção: {detected_intent}.
    2. Use APENAS os produtos listados acima, sem inventar informações.
    3. Se for 'multi_produto', agrupe os produtos por categoria ou nome com subtítulos (ex.: '## Suplementos').
    4. Se não houver produtos adequados, informe de forma educada e sugira alternativas genéricas.
    5. Para cada recomendação, inclua: nome exato, benefício principal, modo de uso básico e preço.
    6. Use Markdown para formatação clara: títulos (##, ###), listas (-), tabelas (|), negrito (**).
    {intent_specific_instructions}

    ### Formato da Resposta
    - **Olá!** (saudação personalizada se possível)
    - Resposta direta à pergunta
    - ## Recomendações (ou seção relevante ao intent)
      - Detalhes estruturados dos produtos
    - ## Próximos Passos
      - Sugestão clara (ex.: "Visite nossa loja online para comprar!")

    Mantenha a resposta concisa, precisa e fácil de ler.
    """
    
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Modelo atualizado
        response = gemini_model.generate_content(prompt)
        if user_id and ingredient_queries:
            user_profile_manager.update_query_history(
                user_id, question, detected_intent, ingredient_queries, filtered_products[:3]
            )
        return response.text
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}")
        try:
            available_models = [m.name for m in genai.list_models()]
            logger.info(f"Modelos disponíveis: {available_models}")
        except Exception as list_error:
            logger.error(f"Erro ao listar modelos: {list_error}")
        return """**Olá!**  
Desculpe, estamos enfrentando um problema técnico. Tente novamente em breve ou entre em contato com nosso suporte pelo e-mail suporte@emporioverdegraos.com.br."""

# Criar índice FAISS e carregar produtos no início
index, products = create_faiss_index()

# Endpoint POST para chat
@app.post("/chat", response_model=ChatResponse)
async def chat_post(request: ChatRequest):
    if index is None:
        raise HTTPException(status_code=500, detail="Sistema não inicializado corretamente.")
    relevant_products, detected_intent = get_relevant_products(
        request.question, index, products, k=20, user_id=request.user_id
    )
    generated_response = generate_response(
        request.question, relevant_products, detected_intent, user_id=request.user_id
    )
    return ChatResponse(
        resposta=generated_response,
        produtos_relacionados=relevant_products[:3],
        intent_detected=detected_intent
    )

# Endpoint GET para chat
@app.get("/chat", response_model=ChatResponse)
async def chat_get(
    question: str = Query(..., description="Pergunta do usuário"),
    user_id: Optional[str] = Query(None, description="ID do usuário (opcional)"),
    session_id: Optional[str] = Query(None, description="ID da sessão (opcional)")
):
    if index is None:
        raise HTTPException(status_code=500, detail="Sistema não inicializado corretamente.")
    relevant_products, detected_intent = get_relevant_products(
        question, index, products, k=20, user_id=user_id
    )
    generated_response = generate_response(
        question, relevant_products, detected_intent, user_id=user_id
    )
    return ChatResponse(
        resposta=generated_response,
        produtos_relacionados=relevant_products[:3],
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
    product = next((p for p in product_cache.product_name_to_data.values() if p.id == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    similar_products = product_cache.get_similar_products(product)
    return {"product": product, "similar_products": similar_products}

@app.get("/user/profile/{user_id}")
async def get_user_profile(user_id: str):
    profile = user_profile_manager.get_profile(user_id)
    return profile

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)