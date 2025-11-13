from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import json                     # <-- (1) إضافة جديدة
from datetime import datetime   # <-- (2) إضافة جديدة

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
# (تأكد أن هذا مطابق 100% لملف ingest_database.py)
COLLECTION_NAME = "example_collection" 

# (تم تعديله ليطابق ingest_database.py)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# initiate the model
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name=COLLECTION_NAME, # (تأكد من إضافة هذا السطر)
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# call this function for every message added to the chatbot
def stream_response(message, history):
    
    # --- (1. البحث والاسترجاع) ---
    print("\n--- DEBUG: Searching ChromaDB ---")
    search_query = message
    # (البحث عن 5 نتائج لرؤية الخيارات)
    results_with_scores = vector_store.similarity_search_with_score(search_query, k=5) 

    if not results_with_scores:
        print("Database found NO results.")
    else:
        for i, (doc, score) in enumerate(results_with_scores):
            print(f"Result {i+1} [Score: {score:.4f}]: {doc.page_content[:100]}...")
    
    # (فلترة النتائج - نقبل فقط النتائج ذات درجة تطابق جيدة)
    # (score هو L2 distance، كلما قل كان أفضل)
    good_docs = [doc for doc, score in results_with_scores if score < 1.5]
    
    if not good_docs:
        print("DEBUG: No results passed the filter (Score too high or no results).")
    print("--------------------------------------\n")

    # (بناء المعرفة "knowledge" والتحضير للتسجيل)
    knowledge = ""
    retrieved_context_for_log = [] # (للتسجيل)

    for doc in good_docs:
        knowledge += doc.page_content + "\n\n"
        retrieved_context_for_log.append(doc.page_content) # (للتسجيل)


    # --- (2. بناء الـ Prompt والاتصال بـ LLM) ---
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        "# هويتك وقدراتك",
        "- أنت مساعد طلاب ومتدربين المعهد السعودي المتخصص العالي  للتدريب",
        "- مهمتك الرئيسية هي تقديم معلومات دقيقة عن برامج المعهد ودوراته ودبلوماته",
        "- عليك توليد الرد بنفس لغة استفسار المستخدم",
        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}
        """

        print("--- PROMPT BEING SENT TO LLM ---")
        print(rag_prompt)
        print("----------------------------------")

        # (Stream الإجابة إلى واجهة Gradio)
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
        
        # --- (3. بداية كود التسجيل - إضافة جديدة) ---
        # (بعد انتهاء الـ stream، "partial_message" يحتوي على الإجابة الكاملة)
        final_answer = partial_message.strip() 

        # (تجهيز البيانات للتسجيل)
        log_entry = {
            "timestamp": datetime.now().isoformat(),  # وقت السؤال
            "user_query": message,                    # سؤال المستخدم الأصلي
            "search_query": search_query,             # السؤال الذي تم البحث به
            "chat_history": history,                  # تاريخ المحادثة
            "retrieved_knowledge": retrieved_context_for_log, # المستندات المسترجعة
            "full_prompt": rag_prompt,                # الـ Prompt الكامل
            "bot_answer": final_answer                # إجابة البوت النهائية
        }

        # (كتابة السجل في ملف "chat_logs.jsonl")
        try:
            # 'a' تعني (append) أي "إضافة" في نهاية الملف
            # 'encoding="utf-8"' ضروري جداً للغة العربية
            with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print("--- INFO: Chat log saved successfully. ---")
        except Exception as e:
            print(f"--- ERROR: Failed to write to log file: {e} ---")
        # --- (نهاية كود التسجيل) ---

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch(share=True)