# News Forecast Pipeline

Репозиторий с воспроизводимым пайплайном для конкурса НИУ ВШЭ «Гонка кентавров ВШЭ — Конкурс краткосрочных прогнозов для команд “Человек + ИИ”».

Проект собирает новости, чистит корпус, выделяет темы, строит baseline-прогнозы и генерирует финальные заголовки и лиды через OpenRouter для даты `2026-04-02`.

## Задача конкурса

По описанию на странице конкурса нужно:

- спрогнозировать заголовки СМИ, которые выйдут `2 апреля 2026 года`;
- для каждого прогноза указать СМИ, ожидаемый заголовок и первый абзац новости;
- описать, как был получен прогноз: какая LLM использовалась и какой был промпт;
- если подается код, приложить техническое описание, `README` и ссылку на публичный GitHub-репозиторий.

В этом репозитории сделан упор на воспроизводимость: весь пайплайн запускается локально, а результаты можно пересчитать через CLI или ноутбук.

Источник описания задачи: [страница конкурса ВШЭ](https://mollab.hse.ru/centaur_hse).

## Что покрывает репозиторий

Сейчас пайплайн работает по трем российским СМИ:

| Slug | Название | Источники | Лиды |
|---|---|---|---|
| `kommersant` | Коммерсантъ | RSS + архив | ✅ |
| `lenta` | Лента.ру | RSS + архив | ✅ |
| `interfax` | Интерфакс | RSS + архив | ✅ |

## Архитектура

```text
titles_forecating/
├── config.py           # пути, даты, параметры источников и OpenRouter
├── scraper.py          # RSS + архивы + приведение к единому raw CSV
├── etl.py              # очистка текстов, title cleanup, dedup, audit-отчеты
├── analyzer.py         # TF-IDF + KMeans темы, эвристическое извлечение сущностей
├── event_calendar.py   # events.csv + recurring fallback-events
├── forecaster.py       # inertia / frequency / calendar / retrieval-grounded LLM
├── backtester.py       # holdout-бэктест на последних 7 днях
├── metrics.py          # topic/entity/semantic/style/diversity метрики
├── main.py             # CLI-точка входа
├── pipeline.ipynb      # Jupyter-ноутбук для пошагового запуска
└── data/
    ├── raw/            # сырые выгрузки
    ├── clean/          # очищенные корпуса и ETL audit
    ├── forecasts/      # прогнозы и backtest-артефакты
    └── events/         # календарные события
```

## Текущий статус

Проект сейчас лучше предсказывает уровень тем, чем конкретные финальные заголовки редакции.

- `frequency` остается самым сильным topic-level baseline;
- `llm` дает правдоподобные и стилистически похожие заголовки, но topic fidelity все еще ограничен;
- `calendar` работает как вспомогательный сигнал, а не как главный источник качества;
- пайплайн подходит для исследовательского прототипа и конкурсной submission с воспроизводимым кодом.

## Быстрый старт

### 1. Установка

```bash
uv sync
```

Если нужен альтернативный способ установки:

```bash
uv pip install -r requirements.txt
```

### 2. Настройка OpenRouter

Создайте `.env` и укажите ключ:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=anthropic/claude-3-haiku-20240307
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Без LLM можно запускать пайплайн с флагом `--no-llm`.

### 3. CLI

```bash
# Полный пайплайн
uv run python main.py --mode all --target 2026-04-02

# Отдельные этапы
uv run python main.py --mode scrape
uv run python main.py --mode etl
uv run python main.py --mode analyze
uv run python main.py --mode backtest
uv run python main.py --mode metrics
uv run python main.py --mode forecast --target 2026-04-02

# Только одно СМИ
uv run python main.py --mode forecast --outlets kommersant --target 2026-04-02
```

### 4. Jupyter

Откройте [pipeline.ipynb](/Users/justcomex/Documents/nstu/pet/titles_forecating/pipeline.ipynb) и выполните ячейки по порядку.

## Флаги CLI

| Флаг | Описание | По умолчанию |
|---|---|---|
| `--mode` | `scrape / etl / analyze / backtest / metrics / forecast / all` | обязательный |
| `--outlets` | список slug через запятую | все 3 |
| `--target` | дата прогноза `YYYY-MM-DD` | `2026-04-02` |
| `--no-llm` | пропустить OpenRouter-генерацию | выключено |
| `--enrich-leads` | скачивать полные лиды из article pages | выключено |

## Методы прогноза

| Метод | Что делает |
|---|---|
| `inertia` | переносит вчерашние темы на целевую дату |
| `frequency` | берет самые частые темы за окно `FREQ_WINDOW` |
| `calendar` | строит заголовки из `events.csv` и recurring событий |
| `llm` | генерирует заголовки и лиды через OpenRouter по retrieval-grounded prompt |

Для честного конкурсного артефакта `forecast`-режим экспортирует только `llm`-прогнозы. Baseline-методы остаются в пайплайне для backtest и сравнения качества.

## Оценка качества

Бэктест делается на последних `7` днях истории. Метрики разделены на topic-level и text-level:

| Метрика | Смысл |
|---|---|
| `topic_hit` | пересечение ключевых слов темы у прогноза и фактических заголовков |
| `entity_f1` | F1 по персонам и организациям |
| `semantic_similarity` | embedding-based похожесть прогноза на реальные новости |
| `style_match` | близость прогноза к стилю корпуса издания |
| `diversity` | разнообразие между сгенерированными заголовками |

Важно: для методов без текстовой генерации (`inertia`, `frequency`) text-level метрики показываются как `n/a`, а не как искусственные нули.

## Выходные файлы

| Файл | Содержимое |
|---|---|
| `data/raw/{slug}_raw.csv` | сырые статьи после скрапинга |
| `data/clean/{slug}_clean.csv` | очищенный корпус |
| `data/clean/{slug}_etl_audit.json` | сводка ETL по шагам |
| `data/clean/{slug}_etl_daily_counts.csv` | дневные counts по этапам ETL |
| `data/forecasts/backtest_{slug}_{YYYYMMDD}.json` | backtest по одному СМИ |
| `data/forecasts/forecast_{YYYY-MM-DD}.json` | итоговый LLM-only прогноз |
| `data/forecasts/forecast_{YYYY-MM-DD}.xlsx` | итоговый LLM-only прогноз в Excel |
| `data/forecasts/forecast_{YYYY-MM-DD}_shortlist.md` | ручная shortlist-выборка лучших LLM-прогнозов для submission |

## Для конкурсной подачи

Если submission идет вместе с кодом, этот репозиторий уже закрывает базовые требования к воспроизводимости:

- есть публичный `README`;
- пайплайн запускается одной CLI-командой;
- можно приложить описание используемой LLM и prompts;
- можно добавить ссылку на публичный репозиторий и финальные артефакты из `data/forecasts/`;
- для быстрой подачи можно использовать готовый shortlist-файл `data/forecasts/forecast_2026-04-02_shortlist.md`.

## Ограничения

- качество topic modeling чувствительно к шуму и слишком широким кластерам;
- `llm` пока лучше в правдоподобии текста, чем в точном угадывании редакционной повестки;
- `calendar` зависит от полноты `events.csv`;
- для повторяемости результатов нужен стабильный доступ к OpenRouter и источникам новостей.
