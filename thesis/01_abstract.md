# Abstract

**Topic:** Tree Detection for Astana — Deep Learning for Urban Green Space Mapping.

**Authors:** Anuar Totin, Rasul Aidarkhanov, Berik Sharipov. Educational program: 6B06103 — Information Technologies (IT-2304).

**Relevance.** Urban tree inventories are a basic input to municipal planning, environmental monitoring and climate-adaptation processes, but the manual field surveys that have historically supplied them are slow, expensive and quickly obsolete for a city of Astana's size. Modern deep-learning models for object detection and instance segmentation now operate on freely available satellite imagery at the per-tree level, opening a route to an automated alternative — but the published literature does not yet contain a single evaluation of these models on Central-Asian imagery.

**Aim.** To design, implement and evaluate an end-to-end software system that, given a satellite image of an area of Astana as input, automatically produces an inventory of the trees present in that area with per-tree polygon mask, confidence score and geographic coordinates.

**Object of research:** the urban green space of Astana, namely the population of individual trees observable from above in the visible spectrum. **Subject of research:** deep-learning models and software components for the detection, segmentation, geographic conversion and export of these trees from satellite imagery.

**Scientific novelty.** This is the first published evaluation of state-of-the-art deep-learning tree-detection models on Astana satellite imagery. The work proposes a three-model architecture combining YOLOv8 instance segmentation, fine-tuned DeepForest bounding-box detection and zero-shot SAM 2 mask refinement, combined through a Weighted-Box-Fusion ensemble. A custom-annotated Astana dataset of ≈ 77 source images and ≈ 8 000 polygon-level annotations has been built and is reusable for future regional research.

**Methods.** Literature analysis of 31 peer-reviewed publications of 2019 – 2025; deep-learning training and fine-tuning in PyTorch / Ultralytics / DeepForest / SAM 2; dataset engineering in CVAT with a model-in-the-loop pre-labelling tool; quantitative evaluation through standard object-detection metrics (Box mAP@50, mAP@50-95, mask AP, precision, recall) and qualitative inspection on representative Astana scenes.

**Structure.** The diploma project consists of three chapters across approximately 45 pages and contains 9 tables and 35 cited sources. Chapter 1 analyses the subject area, surveys the deep-learning state of the art and formulates the precise problem statement. Chapter 2 describes the proposed methodology — the three model branches, the data-preparation pipeline, the ensemble and the geographic conversion. Chapter 3 reports the experimental results: three YOLOv8x-seg checkpoints were trained on successive iterations of the Astana dataset (v1, v2-fromscratch, v2-finetune) and evaluated head-to-head on a common validation set; the best configuration — v2-finetune, which continues training from the v1 checkpoint on the new images only — reaches Box mAP@50 = **0.372** and Mask mAP@50 = **0.331**, a 40 % relative improvement over the v1 baseline (0.265 / 0.240) on the same data; the fine-tuned DeepForest detector, trained on a separate auxiliary annotation set, reaches precision ≈ 0.80 and recall ≈ 0.65; the integrated pipeline processes a 1 km × 1 km capture at zoom 18 in approximately 18 seconds on a single laptop GPU.

**Practical result.** A deployable prototype — FastAPI backend with a pluggable model-adapter interface, React 18 + Leaflet frontend, four-mode geographic conversion, in-browser ESRI tile capture and three exporters (GeoJSON, CSV, standalone HTML) — that delivers an Astana tree inventory in a single click and serves as a reusable template for any other Central-Asian city. The system meets all functional requirements set by *Zelenstroy* and establishes the empirical baseline against which any future model for the region will be compared.

\newpage

# Аннотация

**Тема:** Обнаружение деревьев в городе Астана — глубокое обучение для картирования городских зелёных насаждений.

**Авторы:** Тотин Ануар, Айдарханов Расул, Шарипов Берик. Образовательная программа: 6B06103 — Информационные технологии (IT-2304).

**Актуальность темы** дипломного проекта обусловлена потребностью городских служб Астаны (в первую очередь *Зеленстрой*) в быстром, повторяемом и недорогом способе получения инвентаризации городских деревьев — задачи, которая исторически решалась только трудоёмкими ручными полевыми обследованиями. Современные модели глубокого обучения позволяют автоматизировать эту задачу по бесплатно доступным спутниковым снимкам, однако ни одно опубликованное исследование к настоящему времени не оценивало эти модели на снимках центральноазиатских городов.

**Цель работы** — разработать программную систему, которая по спутниковому снимку любого района Астаны автоматически выдаёт инвентаризацию деревьев с полигональной маской кроны, уверенностью и географическими координатами для каждого дерева.

**Объект исследования** — городские зелёные насаждения Астаны, а именно совокупность отдельных деревьев, видимых сверху в видимом спектре. **Предмет исследования** — модели глубокого обучения и программные компоненты для их детектирования, сегментации, геокодирования и экспорта.

**Научная новизна.** Впервые опубликованы количественные оценки современных моделей обнаружения деревьев на спутниковых снимках Астаны. Предложена трёхмодельная архитектура (YOLOv8 для instance segmentation, DeepForest дообученный для bounding-box детектирования, SAM 2 как zero-shot mask refinement над DeepForest), объединённая через Weighted Box Fusion. Создан собственный размеченный датасет Астаны (≈ 77 изображений / ≈ 8 000 полигональных аннотаций).

**Методы** — анализ 31 публикации 2019–2025 годов, обучение моделей в PyTorch / Ultralytics / DeepForest / SAM 2, инжиниринг данных в CVAT с дообучением через model-in-the-loop, количественная оценка через mAP / precision / recall и качественный анализ на тестовых сценах Астаны.

**Структура.** Дипломный проект состоит из трёх глав, занимает приблизительно 45 страниц и включает 9 таблиц и 35 источников. В первой главе проведён анализ предметной области, обзор литературы и сформулирована постановка задачи. Во второй главе описана предложенная методология: три ветви моделей, конвейер подготовки данных, ансамбль и географическое преобразование. В третьей главе приведены результаты экспериментов: обучены три чекпойнта YOLOv8x-seg (v1, v2-fromscratch, v2-finetune) и проведено их сравнение «голова-в-голову» на общем валидационном наборе; лучшая конфигурация — v2-finetune, дообучение от v1 только на новых снимках — достигает Box mAP@50 = **0,372** и Mask mAP@50 = **0,331**, на 40 % относительно лучше v1 (0,265 / 0,240) на тех же данных; дообученный DeepForest (на отдельном вспомогательном наборе аннотаций) показывает precision ≈ 0,80, recall ≈ 0,65; интегрированный конвейер обрабатывает захват 1 км × 1 км на zoom 18 за ≈ 18 секунд на одной RTX 4060.

**Практический результат** — развёрнутый прототип (FastAPI backend, React + Leaflet frontend, четыре режима геопривязки, захват плиток ESRI прямо из браузера и три формата экспорта — GeoJSON, CSV, автономный HTML), который выдаёт инвентаризацию деревьев Астаны одним кликом и служит переиспользуемым шаблоном для любого другого города региона.

\newpage

# Аңдатпа

**Тақырып:** Астана қаласындағы ағаштарды анықтау — қалалық жасыл аумақтарды картаға түсіруге арналған терең оқыту.

**Авторлар:** Тотин Ануар, Айдарханов Расул, Шарипов Берик. Білім беру бағдарламасы: 6B06103 — Ақпараттық технологиялар (IT-2304).

**Дипломдық жобаның өзектілігі** Астана қаласының қалалық қызметтерінің (бірінші кезекте *Зеленстрой*) қалалық ағаштардың түгендеуін жылдам, қайталанатын және арзан түрде алу қажеттілігінен туындайды. Бұл мәселе тарихи түрде тек қолмен жасалатын далалық зерттеулер арқылы шешілді. Қазіргі заманғы терең оқыту модельдері тегін қол жетімді спутниктік суреттерден осы тапсырманы автоматтандыруға мүмкіндік береді, бірақ Орталық Азия қалаларының суреттеріне арналған бірде-бір жарияланған зерттеу жоқ.

**Зерттеудің мақсаты** — Астананың кез келген аумағының спутниктік суреті бойынша әр ағаштың полигондық масштабы, сенімділігі және географиялық координаталарымен бірге автоматты түрде түгендеу шығаратын бағдарламалық жасақтаманы әзірлеу.

**Зерттеудің нысаны** — Астана қаласының жасыл аумақтары, атап айтқанда жоғарыдан көрінетін жеке ағаштар жиынтығы. **Зерттеудің пәні** — оларды анықтауға, сегментациялауға, геокодтауға және экспорттауға арналған терең оқыту модельдері мен бағдарламалық компоненттер.

**Ғылыми жаңалығы.** Астана спутниктік суреттерінде заманауи терең оқыту модельдерінің сандық бағалаулары алғаш рет жарияланды. Үш модельді архитектура ұсынылды (YOLOv8 — даналық сегментация үшін, дайын DeepForest — шектеу шеңберлерін анықтау үшін, SAM 2 — DeepForest үстіндегі zero-shot маскалау үшін), Weighted Box Fusion арқылы біріктірілген. Астанаға арналған меншікті аннотацияланған деректер жиынтығы құрылды (≈ 77 сурет / ≈ 8 000 полигондық аннотация).

**Әдістер** — 2019–2025 жылдардағы 31 ғылыми жұмысты талдау; PyTorch / Ultralytics / DeepForest / SAM 2 арқылы модельдерді оқыту; CVAT-та деректерді инжинирингтеу; mAP / precision / recall арқылы сандық бағалау және Астана көріністеріндегі сапалық талдау.

**Құрылымы.** Дипломдық жоба үш тараудан тұрады, шамамен 45 бет көлемінде, 9 кесте және 35 дереккөзден тұрады. Бірінші тарауда пәндік сала талданып, әдебиетке шолу жасалып, тапсырманың қойылымы тұжырымдалған. Екінші тарауда ұсынылған әдіснама — үш модель тармағы, деректерді дайындау конвейері, ансамбль және географиялық түрлендіру — сипатталған. Үшінші тарауда эксперимент нәтижелері берілген: YOLOv8x-seg моделінің үш чекпойнты (v1, v2-fromscratch, v2-finetune) дайындалып, ортақ валидация жиынында «бетпе-бет» салыстырылды; ең үздік конфигурация — v2-finetune, тек жаңа суреттер бойынша v1-ден жалғастырылған оқыту — Box mAP@50 = **0,372** және Mask mAP@50 = **0,331** көрсетеді, бұл сол деректер бойынша v1 (0,265 / 0,240) моделінен 40 %-ға жақсырақ; жеке көмекші жинақта дайын DeepForest precision ≈ 0,80, recall ≈ 0,65 береді; біріктірілген конвейер 1 км × 1 км zoom 18 аумағын бір RTX 4060 видеокартасында ≈ 18 секундта өңдейді.

**Тәжірибелік нәтиже** — орналастырылған прототип (FastAPI backend, React + Leaflet frontend, географиялық байланыстырудың төрт режимі, браузерден тікелей ESRI плиткаларын алу және үш экспорт пішімі — GeoJSON, CSV, дербес HTML), ол Астана ағаштарының түгендеуін бір рет шерту арқылы шығарады және аймақтағы кез келген басқа қала үшін қайта пайдалануға болатын үлгі ретінде қызмет етеді.

\newpage
