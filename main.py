import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json
from datetime import datetime
from typing import List, Tuple, Optional

# (كل مكتبات RAG الخاصة بك)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# --- (1) تحميل الإعدادات والنماذج (مرة واحدة عند بدء التشغيل) ---
load_dotenv()

# configuration
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "example_collection" 

print("--- [INFO] Loading models and vector store... ---")
# (تم تعديله ليطابق ingest_database.py)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# initiate the model
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)
print("--- [INFO] Models and vector store loaded successfully. ---")


# --- (2) تعريف نماذج المدخلات والمخرجات (Data Models) ---

# (شكل الـ JSON الذي سيستقبله الـ API)
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Tuple[str, str]]] = [] 

# (شكل الـ JSON الذي سيرجعه الـ API)
class ChatResponse(BaseModel):
    answer: str
    search_query: str # (السؤال الذي تم البحث به)
    retrieved_docs_count: int # (عدد المستندات المسترجعة)


# --- (3) إنشاء تطبيق FastAPI ---
app = FastAPI(
    title=" RAG API للمعهد السعودي المتخصص",
    description="API للرد على استفسارات الطلاب باستخدام RAG والذاكرة.",
    version="1.0.0"
)


# --- (4) تعريف "الدالة المنطقية" (RAG Logic) ---
# (هذه هي دالة stream_response الخاصة بك، ولكن معدلة لترجع بيانات)
def get_rag_response(message: str, history: List[Tuple[str, str]]) -> dict:
    
    search_query = message # (الافتراضي هو السؤال الأصلي)

    # --- (الميزة 2: الذاكرة وإعادة الصياغة للأسئلة التالية) ---
    if history: # (هل يوجد تاريخ للمحادثة؟)
        print("\n--- DEBUG: History found. Attempting to rephrase query. ---")
        
        # (تنسيق بسيط للـ history)
        formatted_history = "\n".join([f"User: {turn[0]}\nAssistant: {turn[1]}" for turn in history])
        
        rephrase_prompt = f"""
        بالنظر إلى تاريخ المحادثة التالي والسؤال الجديد من المستخدم، 
        أعد صياغة السؤال الجديد ليكون "سؤالاً مستقلاً بذاته" (standalone question) يمكن استخدامه للبحث في قاعدة بيانات.
        إذا كان السؤال الجديد مستقلاً بالفعل، أعده كما هو.
        لا تجب على السؤال، فقط أعد صياغته.

        تاريخ المحادثة:
        {formatted_history}

        السؤال الجديد: {message}

        السؤال المستقل:
        """
        
        try:
            # (استدعاء الـ LLM فقط لإعادة الصياغة)
            rephrase_response = llm.invoke(rephrase_prompt)
            search_query = rephrase_response.content.strip()
            print(f"--- DEBUG: Original Query: '{message}' ---")
            print(f"--- DEBUG: Rephrased Query: '{search_query}' ---")
        except Exception as e:
            print(f"--- ERROR in rephrasing: {e} ---")
            search_query = message # (في حالة حدوث خطأ، استخدم السؤال الأصلي)
    else:
        # (هذا هو ما طلبته: لا يوجد تصحيح إملائي للسؤال الأول)
        print("\n--- DEBUG: No history. Using original query for search. ---")
    
    # --- (2. البحث والاسترجاع - نستخدم "search_query" المعاد صياغته) ---
    print("--- DEBUG: Searching ChromaDB ---")
    results_with_scores = vector_store.similarity_search_with_score(search_query, k=5) 

    if not results_with_scores:
        print("Database found NO results.")
    else:
        for i, (doc, score) in enumerate(results_with_scores):
            print(f"Result {i+1} [Score: {score:.4f}]: {doc.page_content[:100]}...")
    
    good_docs = [doc for doc, score in results_with_scores if score < 1.5]
    
    if not good_docs:
        print("DEBUG: No results passed the filter (Score too high or no results).")
    print("--------------------------------------\n")

    knowledge = ""
    retrieved_context_for_log = [] 

    for doc in good_docs:
        knowledge += doc.page_content + "\n\n"
        retrieved_context_for_log.append(doc.page_content) 


    # --- (3. بناء الـ Prompt والاتصال بـ LLM للإجابة النهائية) ---
    print("--- PROMPT BEING SENT TO LLM (Final Answer) ---")
    
    rag_prompt = f"""
    "# هويتك وقدراتك",
    "- أنت مساعد طلاب ومتدربين المعهد السعودي المتخصص العالي  للتدريب",
    "- مهمتك الرئيسية هي تقديم معلومات دقيقة عن برامج المعهد ودوراته ودبلوماته",
    "- عليك توليد الرد بنفس لغة استفسار المستخدم",

    استخدم "تاريخ المحادثة" التالي و "المعرفة المسترجعة" للإجابة على "سؤال المستخدم الأخير".
    
    تاريخ المحادثة:
    {history}

    المعرفة المسترجعة (من قاعدة البيانات):
    {knowledge}
    
    سؤال المستخدم الأخير: {message}
    
    الإجابة:
    """
    
    # (نستخدم "invoke" للحصول على إجابة كاملة، بدلاً من "stream")
    final_response = llm.invoke(rag_prompt)
    final_answer = final_response.content.strip()

    # --- (4. كود التسجيل - الميزة 3) ---
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_query": message,                    # (السؤال الأصلي)
        "search_query": search_query,             # (السؤال المُعاد صياغته)
        "chat_history": history,
        "retrieved_knowledge": retrieved_context_for_log,
        "full_prompt": rag_prompt,
        "bot_answer": final_answer
    }

    try:
        with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print("--- INFO: Chat log saved successfully. ---")
    except Exception as e:
        print(f"--- ERROR: Failed to write to log file: {e} ---")

    # (إرجاع البيانات للـ API)
    return {
        "answer": final_answer,
        "search_query": search_query,
        "retrieved_docs_count": len(good_docs)
    }


# --- (5) تعريف "نقطة النهاية" (API Endpoint) ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    نقطة النهاية الرئيسية للدردشة.
    - تستقبل `message` و `history`.
    - ترجع `answer` ومعلومات إضافية.
    """
    print(f"\n--- [INFO] Received new request: {request.message} ---")
    
    # (استدعاء الدالة المنطقية)
    response_data = get_rag_response(request.message, request.history)
    
    # (إرجاع الـ JSON النهائي)
    return ChatResponse(
        answer=response_data["answer"],
        search_query=response_data["search_query"],
        retrieved_docs_count=response_data["retrieved_docs_count"]
    )

# (نقطة نهاية للتأكد أن الخادم يعمل)
@app.get("/")
def read_root():
    return {"status": "RAG API is running!"}


# (هذا الكود لتشغيل الخادم مباشرة من بايثون - اختياري)
if __name__ == "__main__":
    print("--- [INFO] Starting Uvicorn server... ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)