import os
import shutil
import json
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ##### 1. تم تصحيح المسار (import) ليتوافق مع التحديثات الجديدة #####
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document

load_dotenv()

# --- Config ---
FILE_PATH = "knowledge.jsonl" 
DB_PATH = "chroma_db"
# ##### 2. تم إضافة اسم "المجموعة" (Collection) الذي نبحث عنه #####
COLLECTION_NAME = "example_collection" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
api_key = os.getenv("OPENAI_API_KEY")

# --- 2. تعريف إعدادات الدُفعات ---
BATCH_SIZE = 100 # عدد المستندات في كل دفعة
WAIT_TIME_SECONDS = 30 # الانتظار 30 ثانية بين كل دفعة

# Function to clear old database
def clear_database():
    if os.path.exists(DB_PATH):
        print(f"--- [INFO] Deleting old database: {DB_PATH} ---")
        shutil.rmtree(DB_PATH)
        print("--- [INFO] Old database deleted. ---")
    else:
        print("--- [INFO] No old database found to delete. ---")

# Function to load and process the complex JSONL
def load_and_process_jsonl(file_path):
    print(f"--- [INFO] Loading data from {file_path} ---")
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
                        
                        # إنشاء Document وإضافة المصدر (اسم الملف ورقم السطر)
                        doc = Document(
                            page_content=content,
                            metadata={"source": f"{file_path} (line {line_number})"}
                        )
                        documents.append(doc)

                except json.JSONDecodeError:
                    print(f"--- [WARN] Skipping line {line_number}: Invalid JSON format. ---")
                except Exception as e:
                    print(f"--- [WARN] Skipping line {line_number} due to processing error: {e} ---")

        print(f"--- [INFO] Successfully loaded and combined {len(documents)} Q&A pairs. ---")
        return documents

    except FileNotFoundError:
        print(f"--- [FATAL ERROR] Input file not found: {file_path} ---")
        return []
    except Exception as e:
        print(f"--- [FATAL ERROR] An error occurred while reading file: {e} ---")
        return []


def main():
    if not api_key:
        print("--- [FATAL ERROR] OPENAI_API_KEY not found. Please set it in your .env file. ---")
        return
        
    # 1. Clear old DB
    clear_database()

    # 2. Load documents using our custom function
    documents = load_and_process_jsonl(FILE_PATH)

    if not documents:
        print("--- [FATAL ERROR] No documents were loaded. Exiting. ---")
        return

    # 3. Split documents
    print("--- [INFO] Splitting documents... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"--- [INFO] Created {len(splits)} document splits. ---")

    # 4. Create Embeddings model
    print("--- [INFO] Initializing Embeddings Model (this requires API key)... ---")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    # --- 5. تعديل خطوة الإنشاء لتعمل بالدُفعات ---
    print(f"--- [INFO] Starting batch processing of {len(splits)} splits... ---")
    
    # حساب عدد الدفعات الكلي
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    
    vectorstore = None
    
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        print(f"--- [INFO] Processing batch {current_batch_num} / {total_batches} (docs {i} to {i+len(batch)-1})... ---")
        
        try:
            if i == 0:
                # الدفعة الأولى: قم بإنشاء قاعدة البيانات
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings_model,
                    persist_directory=DB_PATH,
                    # ##### 3. هذا هو السطر الأهم الذي تم إصلاحه #####
                    collection_name=COLLECTION_NAME 
                )
            else:
                # الدفعات التالية: أضف إليها
                vectorstore.add_documents(batch)
            
            print(f"--- [INFO] Batch {current_batch_num} processed successfully. ---")

        except Exception as e:
            print(f"--- [ERROR] Failed to process batch {current_batch_num}. Error: {e} ---")
            # يمكنك اختيار إيقاف التشغيل أو المتابعة
            # return 
        
        # لا تنتظر بعد آخر دفعة
        if i + BATCH_SIZE < len(splits):
            print(f"--- [INFO] Waiting for {WAIT_TIME_SECONDS}s to avoid rate limit... ---")
            time.sleep(WAIT_TIME_SECONDS)

    print("--- [SUCCESS] All batches processed. Vector store created successfully. ---")

if __name__ == "__main__":
    main()