import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .conversation import cleanup_expired_histories
from .reactivation import send_reactivation_message
from .routes import router

_background_lock = threading.Lock()
_background_started = False


def start_background_threads() -> None:
    global _background_started  # pylint: disable=global-statement
    with _background_lock:
        if _background_started:
            return
        threading.Thread(target=cleanup_expired_histories, daemon=True).start()
        threading.Thread(target=send_reactivation_message, daemon=True).start()
        _background_started = True


def create_app() -> FastAPI:
    app = FastAPI(title="WhatsApp Transcription API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    def _startup() -> None:
        start_background_threads()

    return app
