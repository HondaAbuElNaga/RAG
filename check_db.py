import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# تحميل متغيرات البيئة (مثل مفتاح OpenAI)
load_dotenv()

# --- (تأكد أن هذه الإعدادات مطابقة 100% لملفاتك الأخرى) ---
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "example_collection"
# ----------------------------------------------------

print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
print(f"Looking for collection: {COLLECTION_NAME}")

# التحقق من وجود المجلد أصلاً
if not os.path.exists(CHROMA_PATH):
    print(f"\n--- ERROR ---")
    print(f"Error: Directory not found: {CHROMA_PATH}")
    print("This means 'load_data.py' never ran or it failed very early.")
    print("Please run 'python load_data.py' first.")
    exit()

try:
    # تهيئة نفس دالة التضمين المستخدمة في التحميل والبحث
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # الاتصال بقاعدة البيانات الموجودة
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

    # (هذه هي أهم دالة)
    # جلب عدد المستندات في المجموعة
    count = vector_store._collection.count()
    
    print("\n--- RESULT ---")
    print(f"Total documents found in the database: {count}")
    print("----------------")

    if count == 0:
        print("\nWARNING: The database is EMPTY!")
        print("This is 100% the reason for the problem.")
        print("Please re-run 'python load_data.py' and carefully check its output for any errors.")
    else:
        print(f"\nSuccess: The database contains {count} documents.")
        print("This is good. If you still have problems, the issue is more complex.")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An error occurred while trying to connect or count:")
    print(f"{e}")
    print("Please check your CHROMA_PATH, COLLECTION_NAME, and ensure 'load_data.py' ran successfully.")