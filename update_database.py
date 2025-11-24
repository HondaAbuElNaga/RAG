import os
import time
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document

load_dotenv()

# --- Config ---
# لاحظ: هنا نضع اسم الملف الذي يحتوي على الأسئلة الجديدة فقط
NEW_DATA_FILE = "new_data.jsonl" 
DB_PATH = "chroma_db"
COLLECTION_NAME = "example_collection" # يجب أن يكون نفس الاسم المستخدم في الإنشاء
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 100
WAIT_TIME_SECONDS = 30

api_key = os.getenv("OPENAI_API_KEY")

# دالة قراءة الملف (نفس المنطق السابق لكن للملف الجديد)
def load_and_process_jsonl(file_path):
    print(f"--- [INFO] Loading NEW data from {file_path} ---")
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    question = data.get("task")
                    answer_data = data.get("response", {})
                    answer = None
                    if isinstance(answer_data, dict):
                        answer = answer_data.get("answer")

                    if question and answer:
                        content = f"السؤال: {question}\nالجواب: {answer}"
                        doc = Document(
                            page_content=content,
                            metadata={"source": f"{file_path} (line {line_number})", "type": "new_update"}
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"--- [WARN] Skipping line {line_number}: {e} ---")
        return documents
    except FileNotFoundError:
        print(f"--- [ERROR] File {file_path} not found. Make sure you created it. ---")
        return []

def main():
    if not api_key:
        print("--- [ERROR] OPENAI_API_KEY missing. ---")
        return

    # 1. تحميل المستندات الجديدة فقط
    documents = load_and_process_jsonl(NEW_DATA_FILE)
    if not documents:
        print("--- [INFO] No new documents found to add. Exiting. ---")
        return

    # 2. تقسيم النصوص (Splitting)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"--- [INFO] Prepared {len(splits)} new splits to add. ---")

    # 3. تجهيز Embeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    # 4. الاتصال بقاعدة البيانات الموجودة (بدون حذفها)
    print(f"--- [INFO] Connecting to existing Chroma DB at {DB_PATH}... ---")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME
    )

    # 5. إضافة البيانات الجديدة (Batch Processing)
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        print(f"--- [INFO] Adding batch {current_batch_num} / {total_batches}... ---")
        
        try:
            # استخدام add_documents بدلاً من from_documents
            vectorstore.add_documents(batch)
            print(f"--- [SUCCESS] Batch {current_batch_num} added. ---")
        except Exception as e:
            print(f"--- [ERROR] Failed to add batch {current_batch_num}: {e} ---")

        if i + BATCH_SIZE < len(splits):
            print(f"--- [INFO] Waiting {WAIT_TIME_SECONDS}s... ---")
            time.sleep(WAIT_TIME_SECONDS)

    print("--- [DONE] New data successfully added to ChromaDB. ---")

if __name__ == "__main__":
    main()