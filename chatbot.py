from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import json
from datetime import datetime

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "example_collection" 

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

# call this function for every message added to the chatbot
def stream_response(message, history):
    
    # --- (1. خطوة جديدة: إعادة صياغة السؤال بناءً على الذاكرة) ---
    search_query = message # (الافتراضي هو السؤال الأصلي)

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
    if message is not None:

        partial_message = ""

        # (مهم: الـ Prompt الآن يحتوي على الـ history بشكل صريح)
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

        print("--- PROMPT BEING SENT TO LLM (Final Answer) ---")
        # (طباعة جزء صغير من الـ prompt لتجنب ازدحام الـ terminal)
        print(rag_prompt[:500] + "...") 
        print("----------------------------------")

        # (Stream الإجابة إلى واجهة Gradio)
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
        
        # --- (4. كود التسجيل - لا يتغير) ---
        final_answer = partial_message.strip() 

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

# --- (5. تشغيل الواجهة الرسومية - هذا هو الجزء الناقص) ---
print("Starting Gradio Interface...")

demo = gr.ChatInterface(
    fn=stream_response,
    title=" مساعد المعهد السعودي المتخصص العالي",
    description="""
    أهلاً بك. اسألني عن الدبلومات المتاحة، الشروط، الرسوم، أو أي استفسار آخر يخص المعهد.
    (مثال: "ما هو دبلوم التمريض؟" ثم "وما هي مدته؟")
    """,
    examples=[
        "ما هي الدبلومات المعتمدة؟",
        "ما هو دبلوم التمريض؟",
        "كم رسوم دبلوم الحاسب الآلي؟",
        "ما هي شروط القبول؟",
        "هل التسجيل متاح الآن؟"
    ],
    theme="soft",
    concurrency_limit=10
)

if __name__ == "__main__":
    demo.launch(share=True) # (يمكنك تغيير share=True إلى False إذا أردت)