
import io
import json
import os

# для загрузки и использования модели
import joblib
import torch
from torchvision import models, transforms


from PIL import Image # для открытия фото
from openai import OpenAI # для использования апи

#Путь к модельке в сайте
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

#Создаем  модель для последующего внедрения моего файла
classes = joblib.load(os.path.join(MODELS_DIR, "class_names.pkl"))

#Созданние модельки
model = models.resnet50()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, len(classes)),
)
#Загрузка файла дампа с ИИ 
state_dict = torch.load(os.path.join(MODELS_DIR, "garbage_resnet18_final.pth"), map_location="cpu", weights_only=True)
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
#Нормализация
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#Функция предикта 
def classify_waste(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    waste_type = classes[pred_idx.item()]
    return {"waste_type": waste_type, "confidence": round(confidence.item(), 4)}


# форматирование ответа нашей модели с помощью апи ключа
def formating(result: dict) -> dict:
    client = OpenAI(
        api_key=os.getenv("api_key"),
        base_url="https://api.aitunnel.ru/v1/",
    )

    #системный промт который отправляется вместе с ответом нашей модели
    system_prompt = (
        "Ты — API для сортировки мусора. Отвечаешь ТОЛЬКО валидным JSON, без markdown, без пояснений.\n"
        "Формат ответа:\n"
        '{"waste_type": "человекочитаемое название на русском", "bin": "один из списка", "tip": "один конкретный совет 1-2 предложения", "confidence": число}\n'
        "bin СТРОГО один из:\n"
        "- Жёлтый бак — пластик\n"
        "- Синий бак — бумага\n"
        "- Зелёный бак — стекло\n"
        "- Серый бак — смешанные отходы\n"
        "- Красный бак — опасные отходы\n"
        "- Коричневый бак — органика\n"
        "Правила:\n"
        "- confidence бери из входных данных как есть, не меняй\n"
        "- tip должен быть конкретным и полезным для России\n"
        "- НЕ добавляй ничего кроме JSON\n"
        "- НЕ оборачивай в ```json```"
    )
    # это для отладки нашей модели
    print(result)
    # ответ нашей модели
    user_message = f"Тип отхода: {result['waste_type']}, уверенность: {result['confidence']}"

    # запрос к нейросети
    response = client.chat.completions.create(
        model="kimi-k2.5",
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    # ответ нейросети из апишника
    content = response.choices[0].message.content
    # если ответа нету соединение прервалось или другая ошибка то выводим стандартный ответ
    if not content:
        return {
            "waste_type": result["waste_type"],
            "bin": "Серый бак — смешанные отходы",
            "tip": "Не удалось определить точный тип, выбросьте в общий бак.",
            "confidence": result["confidence"],
        }
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # отдаем ответ
    return json.loads(content)
