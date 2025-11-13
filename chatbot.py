from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# initiate the model
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# !! (لقد حذفنا السطر القديم "retriever =" من هنا لأننا سنقوم بالبحث يدوياً) !!

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # --- (بداية التعديل) ---
    # 1. البحث يدوياً في قاعدة البيانات مع إظهار "الدرجة"
    # (score هنا هو L2 distance، كلما قل كان أفضل)
    # سنبحث عن 5 نتائج لنرى الخيارات المتاحة
    print("\n--- DEBUG: Searching ChromaDB ---")
    search_query = message # يمكنك لاحقاً جعل هذا أكثر تعقيداً (مثل إضافة المحادثات السابقة)
    results = vector_store.similarity_search_with_score(search_query, k=5)

    # 2. طباعة النتائج التي تم العثور عليها (قبل الفلترة)
    if not results:
        print("Database found NO results.")
    else:
        for i, (doc, score) in enumerate(results):
            # نطبع أول 100 حرف من كل نتيجة مع الدرجة
            print(f"Result {i+1} [Score: {score:.4f}]: {doc.page_content[:100]}...")
            
    # 3. فلترة النتائج (نحن نقبل فقط النتائج "الجيدة")
    # (لقد رفعنا الحد إلى 1.5 لجعله أقل صرامة مؤقتاً)
    good_docs = [doc for doc, score in results if score < 1.5]
    
    if not good_docs:
        print("DEBUG: No results passed the filter (Score too high or no results).")
    print("--------------------------------------\n")
    # --- (نهاية التعديل) ---


    # 4. بناء "المعرفة" (Knowledge) فقط من النتائج الجيدة
    knowledge = ""
    for doc in good_docs: # (ملاحظة: نستخدم good_docs بدلاً من docs القديمة)
        knowledge += doc.page_content+"\n\n"


    # 5. make the call to the LLM (including prompt)
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

        # هذا السطر مهم جداً، سيطبع لنا المعرفة التي سيراها الـ LLM
        print("--- PROMPT BEING SENT TO LLM ---")
        print(rag_prompt)
        print("----------------------------------")

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
# chatbot.launch(server_name="192.168.50.73", server_port=7860)
chatbot.launch(share=True)