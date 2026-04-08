from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

def main():
    return app

if __name__ == "__main__":
    main()
