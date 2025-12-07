import subprocess, uuid, os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).parent.resolve()
RESULTS = ROOT / "examples" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
app = FastAPI()

@app.get("/")
def index():
    return HTMLResponse("""
      <html><body>
        <h3>Upload JPG/PNG to /upload/ (form below)</h3>
        <form action="/upload/" enctype="multipart/form-data" method="post">
          <input name="file" type="file"/>
          <input type="submit" value="Upload & Generate OBJ"/>
        </form>
      </body></html>
    """)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg",".jpeg",".png")):
        raise HTTPException(400, "image required (.jpg/.png)")
    uid = uuid.uuid4().hex[:8]
    in_path = ROOT / f"tmp_{uid}.jpg"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    # run demo.py to produce OBJ (demo writes to examples/results)
    cmd = ["python3", "demo.py", "-f", str(in_path), "-o", "obj"]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Inference timed out")
    if proc.returncode != 0:
        raise HTTPException(500, f"Inference failed: {proc.stderr[:400]}")
    # find newest obj in results
    objs = sorted(RESULTS.glob("*.obj"), key=os.path.getmtime, reverse=True)
    if not objs:
        raise HTTPException(500, "OBJ not produced")
    return {"obj_url": f"/results/{objs[0].name}"}

@app.get("/results/{name}")
def get_obj(name: str):
    p = RESULTS / name
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p)
