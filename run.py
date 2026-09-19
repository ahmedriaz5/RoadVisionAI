from apps.backend.main import app
import uvicorn
from apps.backend.config import get_settings
if __name__ == "__main__":
    s=get_settings()
    uvicorn.run(app, host=s.host, port=s.port)
