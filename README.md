



Команда победителей v2 сиквел

Состав команды
Участник	Роль
Никишов Артём	ИИ
Габелев Иван	ИИ
Вихарев Виталий	Web
Гареев Тимур	Web
Сницерук Артём	Управленец


Технический стек
Frontend
HTML5
CSS3
JavaScript
(при необходимости React/Vue/другое)
Backend
Python
FastAPI (или Flask, если используется)
Искусственный интеллект
Python
OpenAI API / LLM
LangChain (если используется)
Прочее
Git
GitHub
Docker (если используется)
REST API


Навигация по репозиторию
.
├── backend/          # Серверная часть
├── frontend/         # Клиентская часть
├── ai/               # Модели ИИ и логика работы
├── docs/             # Документация
├── assets/           # Статические файлы
├── README.md         # Описание проекта
└── requirements.txt  # Зависимости Python


Запуск проекта
1. Клонирование репозитория
git clone <ссылка-на-репозиторий>
cd <название-папки>
2. Установка зависимостей
pip install -r requirements.txt
3. Запуск backend
python main.py

или

uvicorn main:app --reload
4. Запуск frontend

Если используется обычный HTML:

Откройте файл index.html в браузере.

Если используется Node.js:

npm install
npm run dev
5. Использование

После запуска приложение будет доступно по адресу:

http://localhost:3000

или

http://localhost:8000

в зависимости от конфигурации проекта.# Ochisheno-
Проект для хакатона.
