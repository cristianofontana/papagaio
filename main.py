from app.application import create_app, start_background_threads

app = create_app()

if __name__ == "__main__":
    start_background_threads()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
