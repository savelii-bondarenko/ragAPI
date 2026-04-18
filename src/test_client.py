import requests
import os

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {
    "X-API-Key": "secret_key"
}


def test_upload_and_query():
    print("1. Creating test files...")

    with open("test_doc_1.txt", "w", encoding="utf-8") as f:
        f.write("The secret code for the director's safe is: 1234.")

    with open("test_doc_2.txt", "w", encoding="utf-8") as f:
        f.write("The fine for being late to work is 5000 rubles. It is doubled for repeated violations.")

    print("2. Sending a BATCH of files to the server...")

    files_to_upload = [
        ("files", ("test_doc_1.txt", open("test_doc_1.txt", "rb"), "text/plain")),
        ("files", ("test_doc_2.txt", open("test_doc_2.txt", "rb"), "text/plain"))
    ]

    try:
        upload_res = requests.post(f"{BASE_URL}/index", headers=HEADERS, files=files_to_upload)

        if upload_res.status_code != 200:
            print(f"❌ Upload Error: {upload_res.text}")
            return

        session_id = upload_res.json().get("index_id")
        print(f"✅ Files successfully uploaded! Session ID: {session_id}\n")

        print("3. Testing AI query (across both documents)...")
        query_payload = {
            "index_id": session_id,
            "message": "What is the safe code and what is the late arrival fine? Find information in both documents.",
            "message_history": []
        }

        query_res = requests.post(f"{BASE_URL}/query", headers=HEADERS, json=query_payload)

        if query_res.status_code != 200:
            print(f"❌ Query Error: {query_res.text}")
            return

        print(f"✅ LLM Response:\n{query_res.json().get('answer')}")

    finally:
        print("\n4. Cleaning up temporary files...")
        os.remove("test_doc_1.txt")
        os.remove("test_doc_2.txt")


if __name__ == "__main__":
    test_upload_and_query()