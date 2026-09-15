from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

app = FastAPI()

# Static files (frontend)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Models / DB init
CHROMA_DIR = "./chroma_db"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
embed_model = SentenceTransformer(EMB_MODEL_NAME)

# ensure collection
collection = None
try:
    collection = client.get_collection(name="docs")
except Exception:
    collection = client.create_collection(name="docs")

# If collection empty, add sample docs
if collection.count() == 0:
    sample_docs = [
        {"id": "finance1", "text": "Vergi beyannamesi hazırlama ve vergi oranları hakkında genel bilgiler.", "meta": {"domain": "finance"}},
        {"id": "finance2", "text": "Bireysel yatırım ve portföy çeşitlendirmesi temel ilkeleri.", "meta": {"domain": "finance"}},
        {"id": "health1", "text": "Grip ve soğuk algınlığı semptomları ve tedavi önerileri. Acil durumlarda doktora başvurun.", "meta": {"domain": "health"}},
        {"id": "health2", "text": "Temel hijyen, aşı takvimi ve kronik hastalık yönetimi hakkında rehber.", "meta": {"domain": "health"}},
        {"id": "law1", "text": "Basit sözleşme maddeleri ve tüketici hakları hakkında genel bilgiler. Hukuki danışmanlık yerine geçmez.", "meta": {"domain": "law"}},
        {"id": "law2", "text": "İş hukuku ve işten çıkarma süreçleri hakkında genel bilgi. Kesin bilgi için avukata danışın.", "meta": {"domain": "law"}}
    ]
    texts = [d["text"] for d in sample_docs]
    ids = [d["id"] for d in sample_docs]
    metadatas = [d["meta"] for d in sample_docs]
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    client.persist()

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse('./static/index.html')

@app.post("/chat")
async def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz")

    # Basit domain classifier
    low = text.lower()
    domain = "general"
    if any(k in low for k in ["vergi", "banka", "yatırım", "faiz"]):
        domain = "finance"
    elif any(k in low for k in ["hastalık", "doktor", "ilaç", "ateş"]):
        domain = "health"
    elif any(k in low for k in ["hukuk", "avukat", "sözleşme", "dava"]):
        domain = "law"

    # Retrieve docs from Chroma
    query_emb = embed_model.encode([text])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=3, where={})
    retrieved_texts = []
    for doc_list in results.get('documents', [[]]):
        for d in doc_list:
            retrieved_texts.append(d)

    # Build prompt (Turkish)
    system = f"Sen Türkçe konuşan yardımcı asistsan. Kullanıcı sorusuna net, kısa ve kaynak göstererek cevap ver. Domain: {domain}. Eğer soru kritik tıbbi/hukuki/finansal tavsiye gerektiriyorsa kullanıcıyı uzmana yönlendir."
    context = "\n---\n".join(retrieved_texts)
    user_prompt = f"Kullanıcı: {text}\n\nElde edilen belgeler:\n{context}\n\nCevabı kısa ve Türkçe ver. Kaynakları belirt."

    if openai.api_key:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                max_tokens=600,
                temperature=0.2
            )
            answer = resp['choices'][0]['message']['content'].strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Demo fallback
        answer = "(API anahtarı eklenmedi) Örnek cevap: Bu bir demo yanıttır. Lütfen .env içine OPENAI_API_KEY ekleyin.\n\nKısaca: " + (retrieved_texts[0] if retrieved_texts else "Özür dilerim, yeterli veri yok.")

    # Add generic disclaimer for critical domains
    if domain in ["health", "law", "finance"]:
        answer = answer + "\n\nNot: Bu bilgi genel amaçlıdır ve profesyonel tavsiye yerine geçmez."

    return {"answer": answer, "domain": domain, "sources": retrieved_texts}
