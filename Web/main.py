from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from services.classifier import classify_waste, formating

# Загружаем переменные из файла .env в окружение программы
load_dotenv()

# Создаём экземпляр приложения FastAPI
app = FastAPI()

# Подключаем статические файлы (логотипы и т.д.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Указываем папку "templates", где лежат файлы
templates = Jinja2Templates(directory="templates")


# / — главная страница
@app.get("/", response_class=HTMLResponse)  # обрабатываем GET-запрос на "/"
async def index(request: Request):
    # Рендерим шаблон index.html и отдаём как HTML-страницу
    return templates.TemplateResponse(request, "index.html")


# /scan — загрузка картинки для классификации мусора
@app.post("/scan")  # обрабатываем POST-запрос на "/scan"
async def scan(file: UploadFile = File(...)):
    try:
        # Читаем загруженный файл целиком в байты (картинка в памяти)
        image_bytes = await file.read()
        raw = classify_waste(image_bytes)
        result = formating(raw)
        # Отправляем клиенту JSON с результатом
        return JSONResponse(content=result)

    except Exception as e:
        # Если что-то пошло не так (битый файл, ошибка модели и т.д.):
        # выводим ошибки
        import traceback
        traceback.print_exc()
        # Отправляем клиенту JSON с текстом ошибки и HTTP-код 400 (Bad Request)
        return JSONResponse(content={"error": str(e)}, status_code=400)


# /health — проверка, жив ли сервер
@app.get("/health")  # обрабатываем GET-запрос на "/health"
async def health():
    # Просто возвращаем {"status": "ok"}
    return {"status": "ok"}