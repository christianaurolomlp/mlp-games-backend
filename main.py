"""
MLP Games Backend - Carrera Criptos
FastAPI backend for YouTube live chat collection + Gemini AI parsing
"""

import os
import uuid
import asyncio
import logging
import json
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Optional imports with graceful fallback
try:
    import pytchat
    PYTCHAT_AVAILABLE = True
except ImportError:
    PYTCHAT_AVAILABLE = False
    logging.warning("pytchat not available - YouTube chat collection disabled")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic SDK not available - AI parsing disabled")

# ─── Config ───────────────────────────────────────────────────────
app = FastAPI(title="MLP Games API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlp-games")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── In-memory session store ─────────────────────────────────────
sessions: dict = {}

# ─── Valid crypto symbols ─────────────────────────────────────────
VALID_CRYPTOS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK',
    'MATIC', 'UNI', 'SHIB', 'LTC', 'TRX', 'ATOM', 'XMR', 'ETC', 'XLM', 'NEAR',
    'APT', 'ARB', 'OP', 'SUI', 'SEI', 'INJ', 'TIA', 'FET', 'RNDR', 'RENDER',
    'WLD', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'AAVE', 'MKR', 'CRV', 'LDO', 'SNX',
    'RUNE', 'IMX', 'FTM', 'SAND', 'MANA', 'AXS', 'GALA', 'ENJ', 'FLOW', 'CHZ',
    'ENS', 'DYDX', 'GMX', 'STX', 'ICP', 'FIL', 'GRT', 'QNT', 'HBAR', 'VET',
    'ALGO', 'EOS', 'XTZ', 'THETA', 'EGLD', 'KAVA', 'ROSE', 'ZIL', 'ONE', 'HOT',
    'ONDO', 'BIO', 'PIPPIN', 'ASTER', 'PUMP', 'RED', 'PENGU', 'JASMY', 'BLUR',
    'TAO', 'FTT', 'KAS', 'TON', 'NOT', 'DOGS', 'HMSTR', 'CATI', 'EIGEN',
    'TRUMP', 'MELANIA', 'ACT', 'USUAL', 'MOVE', 'ME', 'ENA', 'W', 'BOME',
    'PNUT', 'GOAT', 'MOODENG', 'NEIRO', 'TURBO', 'PRIME', 'BEAM', 'RONIN',
    'METIS', 'ZRO', 'ETHFI', 'REZ', 'LISTA', 'BB', 'OMNI', 'SAGA',
    'ANIME', 'CGPT', 'GPS', 'PAWS', 'SOLV', 'COOKIE', 'VANA', 'DRIFT',
    'TNSR', 'SAFE', 'COW', 'SCR', 'BERA', 'KAITO', 'LAYER', 'SHELL',
    'FORM', 'FUN', 'RAY', 'ORCA', 'PUFFER', 'KERNEL', 'PROMPT', 'SIGN',
    'INIT', 'HAEDAL', 'MUBARAK', 'BROCCOLI', 'TUT', 'D', 'VINE',
]

# ─── Models ───────────────────────────────────────────────────────
class RaceStartRequest(BaseModel):
    video_id: str
    max_participants: int = 15
    game_type: str = "race"
    asset: str = "BTC"

class RaceStopRequest(BaseModel):
    pass

class ChatMessage(BaseModel):
    author: str
    text: str

class ParseChatRequest(BaseModel):
    messages: list[ChatMessage]
    game_type: str = "race"
    asset: str = "BTC"


# ─── YouTube Chat Collector (background task) ────────────────────
async def collect_chat(session_id: str, video_id: str, max_participants: int):
    if not PYTCHAT_AVAILABLE:
        logger.error("pytchat not available")
        sessions[session_id]["error"] = "pytchat not installed"
        sessions[session_id]["collecting"] = False
        return

    try:
        loop = asyncio.get_event_loop()
        chat = await loop.run_in_executor(None, lambda: pytchat.create(video_id=video_id, interruptable=False))
        logger.info(f"[{session_id}] Started collecting from video {video_id}")

        while sessions.get(session_id, {}).get("collecting", False):
            if not chat.is_alive():
                logger.warning(f"[{session_id}] Chat stream ended")
                break
            items = await loop.run_in_executor(None, lambda: list(chat.get().sync_items()))
            for item in items:
                if not sessions.get(session_id, {}).get("collecting", False):
                    break
                sessions[session_id]["raw_messages"].append({
                    "author": item.author.name,
                    "text": item.message,
                    "timestamp": item.datetime,
                })
            await asyncio.sleep(1)

        chat.terminate()
        logger.info(f"[{session_id}] Stopped. {len(sessions[session_id]['raw_messages'])} messages.")

    except Exception as e:
        logger.error(f"[{session_id}] Chat collection error: {e}")
        sessions[session_id]["error"] = str(e)
        sessions[session_id]["collecting"] = False


# ─── Anthropic AI Parser ──────────────────────────────────────────
async def parse_with_ai(messages: list[dict], game_type: str, asset: str = "BTC") -> list[dict]:
    if not ANTHROPIC_AVAILABLE:
        raise HTTPException(status_code=500, detail="anthropic SDK not installed")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    chat_text = "\n".join([f"{m['author']}: {m['text']}" for m in messages])
    cryptos_list = ", ".join(VALID_CRYPTOS)

    if game_type == "race":
        prompt = f"""Eres un extractor de participantes de un chat de YouTube para una carrera de criptomonedas.

COMENTARIOS DEL CHAT:
{chat_text}

CRIPTOS VÁLIDAS: {cryptos_list}

REGLAS:
- Extrae el nombre de usuario y la cripto que eligió
- Sé MUY FLEXIBLE: "voy con btc", "yo BTC", "mi voto es ethereum", "ETH porfavor", "elijo solana", "SOL!!", "DOGE 🚀" — todos válidos
- Si mencionan nombre completo (bitcoin, ethereum, solana, dogecoin...) conviértelo al símbolo
- Si aparece varias veces el mismo usuario, usa su ÚLTIMO mensaje
- Ignora líneas del sistema de YouTube (Top Fans, Actividad del canal, Miembro X meses, Usuario X meses, Nuevo miembro, etc.)
- Si no hay cripto clara en el mensaje, omite ese usuario
- NO confundas nombres de usuario con criptos

Responde SOLO con JSON sin markdown:
[{{"name": "usuario1", "crypto": "BTC"}}, {{"name": "usuario2", "crypto": "ETH"}}]

Si no hay ninguno válido: []"""
    else:
        prompt = f"""Eres un extractor de predicciones de precio de {asset} de un chat de YouTube.

COMENTARIOS DEL CHAT:
{chat_text}

REGLAS:
- Extrae nombre de usuario y el número que predice como precio de {asset} en USD
- Sé MUY FLEXIBLE: "yo digo 72.500", "creo que 71k", "$69,800", "85000", "mi pred: 75K"
- Convierte K a miles: 71k=71000, 2.5k=2500
- Punto como separador de miles (español): 72.500=72500
- Si un usuario aparece varias veces, usa su ÚLTIMA predicción
- Ignora spam del sistema de YouTube (Miembro X meses, etc.)

Responde SOLO con JSON sin markdown:
[{{"name": "usuario1", "prediction": 100000}}, {{"name": "usuario2", "prediction": 95000}}]

Si no hay ninguno válido: []"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            results = json.loads(json_match.group(0))
            if game_type == "race":
                results = [r for r in results if r.get("crypto", "").upper() in VALID_CRYPTOS]
                for r in results:
                    r["crypto"] = r["crypto"].upper()
            return results
        return []
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid ANTHROPIC_API_KEY")
    except Exception as e:
        logger.error(f"AI parse error: {e}")
        raise HTTPException(status_code=500, detail=f"AI parsing error: {str(e)}")


# ─── API Endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "MLP Games API", "version": "2.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pytchat": PYTCHAT_AVAILABLE,
        "anthropic": ANTHROPIC_AVAILABLE,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "active_sessions": len([s for s in sessions.values() if s.get("collecting")]),
    }


@app.post("/api/race/start")
async def race_start(req: RaceStartRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "id": session_id,
        "video_id": req.video_id,
        "max_participants": req.max_participants,
        "game_type": req.game_type,
        "asset": req.asset.upper(),
        "collecting": True,
        "participants": [],
        "raw_messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
    }
    background_tasks.add_task(collect_chat, session_id, req.video_id, req.max_participants)
    return {
        "session_id": session_id,
        "status": "collecting",
        "video_id": req.video_id,
        "max_participants": req.max_participants,
        "game_type": req.game_type,
        "asset": req.asset.upper(),
    }


@app.get("/api/race/{session_id}/participants")
async def race_participants(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "participants": session["participants"],
        "count": len(session["participants"]),
        "max": session["max_participants"],
        "collecting": session["collecting"],
        "raw_messages_count": len(session["raw_messages"]),
        "error": session.get("error"),
    }


@app.post("/api/race/{session_id}/stop")
async def race_stop(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    session["collecting"] = False

    if session["raw_messages"]:
        try:
            seen = {}
            for msg in session["raw_messages"]:
                seen[msg["author"]] = msg
            messages_to_parse = list(seen.values())
            if len(messages_to_parse) > 200:
                messages_to_parse = messages_to_parse[-200:]

            parsed = await parse_with_ai(
                [{"author": m["author"], "text": m["text"]} for m in messages_to_parse],
                session.get("game_type", "race"),
                session.get("asset", "BTC")
            )
            session["participants"] = parsed[:session["max_participants"]]
        except Exception as e:
            logger.error(f"Error parsing messages: {e}")
            session["error"] = f"Parse error: {str(e)}"

    return {
        "participants": session["participants"],
        "count": len(session["participants"]),
        "final": True,
        "raw_messages_count": len(session["raw_messages"]),
    }


@app.post("/api/parse-chat")
async def parse_chat(req: ParseChatRequest):
    messages = [{"author": m.author, "text": m.text} for m in req.messages]
    results = await parse_with_ai(messages, req.game_type, req.asset)
    return {"participants": results, "count": len(results)}


@app.get("/api/race/{session_id}/result")
async def race_result(session_id: str, asset: str = "BTCUSDT"):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    participants = session.get("participants", [])
    if not participants:
        raise HTTPException(status_code=400, detail="No participants in session")

    if not asset.endswith("USDT"):
        asset = asset.upper() + "USDT"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={asset}",
                timeout=10.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Binance API error: {response.status_code}")
            current_price = float(response.json()["price"])
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Binance: {str(e)}")

    ranking = []
    for p in participants:
        prediction = p.get("prediction")
        if prediction is None:
            continue
        prediction = float(prediction)
        distance = abs(prediction - current_price)
        ranking.append({
            "name": p.get("name", "Unknown"),
            "prediction": prediction,
            "distance": round(distance, 2),
            "difference_pct": round((prediction - current_price) / current_price * 100, 2),
        })

    ranking.sort(key=lambda x: x["distance"])
    for i, r in enumerate(ranking):
        r["position"] = i + 1

    return {
        "asset": asset.replace("USDT", ""),
        "current_price": current_price,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
        "total_participants": len(ranking),
    }


@app.delete("/api/race/{session_id}")
async def race_delete(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions.pop(session_id)
    session["collecting"] = False
    return {"deleted": True}


@app.get("/api/sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "id": s["id"],
                "video_id": s["video_id"],
                "game_type": s.get("game_type", "race"),
                "asset": s.get("asset", ""),
                "collecting": s["collecting"],
                "participants": len(s["participants"]),
                "raw_messages": len(s["raw_messages"]),
                "created_at": s["created_at"],
            }
            for s in sessions.values()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
