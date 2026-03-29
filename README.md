# News Forecast Pipeline

Автоматический пайплайн сбора, анализа и прогнозирования новостей пяти крупнейших российских СМИ с LLM-генерацией заголовков на дату **02.04.2026**.

## СМИ

| Slug | Название | Источники | Лиды |
|---|---|---|---|
| `rbc` | РБК | RSS + архив | ✅ |
| `kommersant` | Коммерсантъ | RSS + архив | ✅ |
| `vedomosti` | Ведомости | RSS (paywall) | ❌ |
| `lenta` | Лента.ру | RSS + архив | ✅ |
| `interfax` | Интерфакс | RSS + архив | ✅ |

## Архитектура

```
titles/
├── config.py           # пути, даты, параметры всех СМИ и OpenRouter (через .env)
├── scraper.py          # RSS + архив + Wayback Machine CDX fallback
├── etl.py              # очистка, дедупликация (TF-IDF cosine), фильтр мнений
├── analyzer.py         # TF-IDF + KMeans топики, NER (pymystem3), noise check
├── event_calendar.py   # события апреля 2026 из CSV
├── forecaster.py       # baseline inertia/frequency/calendar + OpenRouter LLM
├── backtester.py       # holdout-бэктест на последних 7 днях
├── metrics.py          # 5 метрик качества прогноза
├── main.py             # CLI точка входа
├── pipeline.ipynb      # интерактивный Jupyter-ноутбук
├── requirements.txt
└── data/
    ├── raw/            # сырые CSV после скрапинга
    ├── clean/          # очищенные CSV после ETL
    ├── forecasts/      # JSON/XLSX прогнозы и бэктест-отчёты
    └── events/
        └── events.csv  # календарь событий апреля 2026
```

## Быстрый старт

### 1. Установка зависимостей

```bash
uv pip install -r requirements.txt
```

Python 3.10+. Для Python 3.14 используйте версии пакетов из `requirements.txt` — они содержат pre-built wheels.

### 2. OpenRouter (LLM)

1. Создайте файл `.env` на основе `.env.example`:
   ```bash
   cp .env.example .env
   ```
2. Впишите ваш API-ключ в `.env` (`OPENROUTER_API_KEY=...`).
3. По желанию измените модель (`OPENROUTER_MODEL=...`).

Без LLM можно запустить с флагом `--no-llm`.

### 3. Запуск через CLI

```bash
# Полный пайплайн (сбор → ETL → анализ → бэктест → прогноз)
uv run python main.py --mode all --target 2026-04-02

# Отдельные шаги
uv run python main.py --mode scrape
uv run python main.py --mode etl
uv run python main.py --mode analyze
uv run python main.py --mode backtest
uv run python main.py --mode metrics
uv run python main.py --mode forecast --target 2026-04-02

# Только одно СМИ, без LLM
uv run python main.py --mode all --outlets rbc --no-llm
```

### 4. Запуск в Jupyter

Откройте `pipeline.ipynb` и последовательно выполните ячейки (0 → 7).

## Флаги CLI

| Флаг | Описание | По умолчанию |
|---|---|---|
| `--mode` | Этап пайплайна: `scrape / etl / analyze / backtest / metrics / forecast / all` | обязательный |
| `--outlets` | Список СМИ через запятую: `rbc,lenta,...` | все 5 |
| `--target` | Дата прогноза `YYYY-MM-DD` | `2026-04-02` |
| `--no-llm` | Пропустить генерацию через OpenRouter | выключено |
| `--enrich-leads` | Скачивать полные лиды (медленно, ~1-2 с/статья) | выключено |

## Методы прогнозирования

| Метод | Описание |
|---|---|
| **inertia** | Темы с наибольшей частотой за последние 7 дней |
| **frequency** | Темы с наибольшей частотой за скользящее окно 30 дней |
| **calendar** | События из `data/events/events.csv` в ±3 дня от целевой даты |
| **llm** | OpenRouter (Claude/Llama/Gemma) генерирует заголовки + лиды по промпту со стилем СМИ |

## Метрики качества (бэктест)

| Метрика | Цель | Метод |
|---|---|---|
| `topic_hit_rate` | ≥ 0.30 | Jaccard IoU токенов тем |
| `entity_match_f1` | ≥ 0.20 | F1 по персонам и организациям |
| `semantic_similarity` | ≥ 0.45 | Cosine по `paraphrase-multilingual-MiniLM-L12-v2` |
| `style_match` | ≥ 0.30 | TF-IDF cosine прогноза к корпусу СМИ |
| `diversity_score` | ≥ 0.70 | 1 − среднее попарное сходство заголовков |

## Выходные данные

| Файл | Содержимое |
|---|---|
| `data/clean/{slug}_clean.csv` | Очищенные статьи: `id, outlet, url, published_at, title, lead, rubric` |
| `data/forecasts/forecast_{date}.json` | Прогноз: все методы, все СМИ |
| `data/forecasts/forecast_{date}.xlsx` | Прогноз в Excel: листы по методам и СМИ |
| `data/forecasts/backtest_{slug}_{date}.json` | Бэктест-отчёт по одному СМИ |

## Требования

- Python ≥ 3.10
- API ключ OpenRouter
- RAM ≥ 8 ГБ (для `sentence-transformers`)
- Интернет для скрапинга RSS и архивов

