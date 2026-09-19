import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from .config import get_settings
from .database import Database
from .vision import TrafficEngine

settings = get_settings()
subscribers = set()
db = Database(settings.database)

def on_event(event):
    db.insert_event(event)
    payload = event.model_dump()
    for ws in list(subscribers):
        try:
            asyncio.get_event_loop().create_task(ws.send_text(json.dumps(payload)))
        except Exception:
            pass

engine = TrafficEngine(event_callback=on_event)

@asynccontextmanager
async def lifespan(app):
    engine.start()
    yield
    engine.stop()

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><meta charset="utf-8"><title>RoadVision AI</title>
<style>
body{margin:0;background:#070a0f;color:#e8eef7;font-family:Inter,Arial,sans-serif}
header{padding:18px 28px;border-bottom:1px solid #202733;font-size:22px;font-weight:700}
main{padding:24px;max-width:1500px;margin:auto}.status{padding:14px 16px;border-radius:12px;background:#111720;border:1px solid #27303d;margin-bottom:16px}
.status b{margin-right:12px}.ok{color:#62e38b}.bad{color:#ff6b6b}
.cards{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin:16px 0}
.card{background:#111720;border:1px solid #27303d;padding:14px 18px;border-radius:12px}
.card b{display:block;font-size:12px;color:#9aabc0}.card span{font-size:24px}
.viewer{background:#020408;border:1px solid #27303d;border-radius:14px;overflow:hidden;min-height:360px;display:flex;align-items:center;justify-content:center}
.viewer img{width:100%;display:block}.hint{color:#8b9ab0;padding:30px;text-align:center}
</style></head><body><header>🚦 RoadVision AI — Traffic Intelligence</header><main>
<div class="status" id="status">Connecting to AI engine...</div>
<div class="cards" id="stats"></div>
<div class="viewer"><img id="video" src="/video" onerror="this.style.display='none';document.getElementById('hint').style.display='block'"><div id="hint" class="hint" style="display:none">No video stream available. Check the AI engine status above.</div></div>
<script>
async function update(){
 try{
  const h=await fetch('/api/health').then(r=>r.json());
  const el=document.getElementById('status');
  el.innerHTML=`<b class="${h.status==='RUNNING'?'ok':'bad'}">${h.status}</b> Source: ${h.source}${h.error?`<br><small>${h.error}</small>`:''}`;
  const s=await fetch('/api/stats').then(r=>r.json());
  document.getElementById('stats').innerHTML=Object.entries(s).map(([k,v])=>`<div class="card"><b>${k.replaceAll('_',' ').toUpperCase()}</b><span>${v}</span></div>`).join('');
 }catch(e){document.getElementById('status').innerHTML='<b class="bad">BACKEND OFFLINE</b>'}
}
setInterval(update,1000);update();
</script></main></body></html>"""
@app.get("/api/stats")
def stats():
    return engine.stats().model_dump()

@app.get("/api/health")
def health():
    return engine.health()

def frames():
    while True:
        frame = engine.jpeg()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            import time; time.sleep(.05)

@app.get("/video")
def video():
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/events")
async def events(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        subscribers.discard(ws)
