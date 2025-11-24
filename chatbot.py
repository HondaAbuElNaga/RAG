import os
import json
from datetime import datetime
import gradio as gr

# استيراد مكتبات LangChain و Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# تحميل متغيرات البيئة (.env)
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "example_collection" 

# إعداد نموذج Embeddings (مطابق لما تم استخدامه في ingest)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# إعداد نموذج الـ LLM
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

# الاتصال بقاعدة البيانات Chroma
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# تحديد حجم الذاكرة (عدد المحادثات السابقة التي يتم تذكرها)
MEMORY_WINDOW_SIZE = 3 

def stream_response(message, history):
    """
    الدالة الرئيسية التي تعالج رسالة المستخدم، تبحث في قاعدة البيانات،
    وتولد الإجابة ثم تحفظ السجل.
    """
    
    # 1. معالجة الذاكرة (تحديد الذاكرة المحدودة)
    limited_history = history[-MEMORY_WINDOW_SIZE:]
    search_query = message 

    # 2. إعادة صياغة السؤال إذا وجد تاريخ للمحادثة (مفعلة كما طلبت)
    if history:
        print(f"\n--- DEBUG: History found. Using last {len(limited_history)} turns for rephrasing. ---")
        formatted_history = "\n".join([f"User: {turn[0]}\nAssistant: {turn[1]}" for turn in limited_history])
        
        rephrase_prompt = f"""
        بالنظر إلى تاريخ المحادثة التالي (آخر {len(limited_history)} محادثات)، والسؤال الجديد من المستخدم، 
        أعد صياغة السؤال الجديد ليكون "سؤالاً مستقلاً بذاته" (standalone question).

        تاريخ المحادثة:
        {formatted_history}

        السؤال الجديد: {message}

        السؤال المستقل:
        """
        
        try:
            rephrase_response = llm.invoke(rephrase_prompt)
            search_query = rephrase_response.content.strip()
            print(f"--- DEBUG: Original Query: '{message}' ---")
            print(f"--- DEBUG: Rephrased Query: '{search_query}' ---")
        except Exception as e:
            print(f"--- ERROR in rephrasing: {e} ---")
            search_query = message 
    else:
        print("\n--- DEBUG: No history. Using original query for search. ---")
    
    # 3. البحث والاسترجاع من ChromaDB
    print("--- DEBUG: Searching ChromaDB ---")
    results_with_scores = vector_store.similarity_search_with_score(search_query, k=5) 

    if not results_with_scores:
        print("Database found NO results.")
    else:
        for i, (doc, score) in enumerate(results_with_scores):
            print(f"Result {i+1} [Score: {score:.4f}]: {doc.page_content[:100]}...")
    
    # تصفية النتائج (Score Filter)
    good_docs = [doc for doc, score in results_with_scores if score < 1.5]
    
    if not good_docs:
        print("DEBUG: No results passed the filter (Score too high or no results).")
    print("--------------------------------------\n")

    knowledge = ""
    retrieved_context_for_log = [] 

    for doc in good_docs:
        knowledge += doc.page_content + "\n\n"
        retrieved_context_for_log.append(doc.page_content) 


    # 4. بناء الـ Prompt والاتصال بـ LLM
    partial_message = ""
    if message is not None:
        rag_prompt = f"""
        "# هويتك وقدراتك",
        "- أنت مساعد طلاب ومتدربين المعهد السعودي المتخصص العالي  للتدريب",
        "- مهمتك الرئيسية هي تقديم معلومات دقيقة عن برامج المعهد ودوراته ودبلوماته",
        "- عليك توليد الرد بنفس لغة استفسار المستخدم",

        استخدم "تاريخ المحادثة" التالي و "المعرفة المسترجعة" للإجابة على "سؤال المستخدم الأخير".
        
        تاريخ المحادثة:
        {limited_history}
        
        المعرفة المسترجعة (من قاعدة البيانات):
        {knowledge}
        
        سؤال المستخدم الأخير: {message}
        
        الإجابة:
        """

        print("--- PROMPT BEING SENT TO LLM (Final Answer) ---")
        
        # Stream الإجابة إلى واجهة Gradio
        for response in llm.stream(rag_prompt):
            chunk = response.content
            partial_message += chunk
            yield partial_message
        
        # 5. حفظ السجلات (Logs) - التعديل الجديد بالتاريخ والمسار المطلق
        final_answer = partial_message.strip() 

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "search_query": search_query, # سيحفظ السؤال المعاد صياغته هنا
            "chat_history": history, 
            "retrieved_knowledge": retrieved_context_for_log,
            "bot_answer": final_answer
        }

        try:
            # 1. نحدد مجلد الكود الحالي بدقة (لضمان مكان الحفظ)
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. اسم الملف يعتمد على تاريخ اليوم (ملف واحد لكل يوم)
            # مثال: chat_logs_2023-10-25.jsonl
            timestamp_str = datetime.now().strftime("%Y-%m-%d") 
            log_filename = f"chat_logs_{timestamp_str}.jsonl"
            
            # دمج المسار
            abs_path = os.path.join(current_script_dir, log_filename)
            
            print(f"--- DEBUG: Writing log to: {abs_path} ---")

            with open(abs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f.flush()           # دفع البيانات فوراً
                os.fsync(f.fileno()) # التأكد من كتابتها فعلياً على القرص الصلب
            
            print(f"--- INFO: Chat log saved successfully. File size: {os.path.getsize(abs_path)} bytes ---")
        except Exception as e:
            print(f"--- ERROR: Failed to write to log file: {e} ---")

# --- تشغيل الواجهة الرسومية ---
print("Starting Gradio Interface...")

demo = gr.ChatInterface(
    fn=stream_response,
    title="🤖 مساعد المعهد السعودي المتخصص العالي (نسخة تجريبية)",
    description="""
    أهلاً بك. اسألني عن الدبلومات المتاحة، الشروط، أو أي استفسار آخر يخص المعهد.
    """,
    examples=[
        "ما هو دبلوم  إدارة التمريض؟",
        "ما هي شروط القبول؟",
    ],
    theme="soft",
    concurrency_limit=10
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)