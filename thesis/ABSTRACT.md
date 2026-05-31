# АННОТАЦИЯ

**Тема дипломной работы:** «Разработка модели глубокого обучения для автоматического распознавания деревьев и картографирования зелёных насаждений в городской среде».

**Студенты:** Тотин Ануар, Айдарханов Расул, Шарипов Берик — образовательная программа 6B06101 «Computer Science». **Научный руководитель:** Сындар Сатбаев.

Актуальность темы дипломной работы состоит в том, что инвентаризация городских деревьев необходима для городского планирования, экологического мониторинга и адаптации к изменению климата, однако традиционные полевые обследования медленны, дороги и быстро устаревают, тогда как в научной литературе отсутствует оценка современных моделей обнаружения деревьев на спутниковых снимках Центральной Азии. Цель работы — спроектировать, реализовать и оценить сквозную систему, которая по спутниковому снимку участка Астаны автоматически формирует инвентаризацию деревьев с полигональной маской, оценкой уверенности и географическими координатами. Объект исследования — городские зелёные насаждения Астаны, то есть отдельные деревья, наблюдаемые сверху в видимом спектре. Предмет исследования — модели глубокого обучения и программные компоненты для обнаружения деревьев, сегментации крон и преобразования координат. Научная новизна работы состоит в том, что впервые опубликована оценка современных моделей обнаружения деревьев на снимках Астаны (Box mAP@50 = 0,012 для готовой модели DeepForest), предложена четырёхмодельная архитектура с двумя ансамблями и собран собственный набор данных Астаны (≈ 100 снимков, ≈ 5 500 размеченных вручную крон, ≈ 8 700 после нарезки на тайлы). Для достижения цели использованы методы анализа литературы 2019–2025 гг., обучения и дообучения моделей (PyTorch, Ultralytics, torchvision, DeepForest, SAM 2), разметки в CVAT и количественной оценки по метрикам COCO.

Дипломная работа содержит 3 главы на 83 страницах и имеет 17 рисунков, а также 13 таблиц.

В первой главе отражены анализ предметной области, обзор литературы 2019–2025 гг. в табличной форме, выявление разрыва географической генерализации для Центральной Азии и постановка задачи.

Вторая глава посвящена методологии: архитектуре системы, четырём моделям (YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest + SAM 2), подготовке данных, ансамблям, географическому преобразованию, хранению и веб-приложению Canopy.

В третьей главе представлены эксперименты и результаты: серия из 23 экспериментов по подбору гиперпараметров, кросс-модельное сравнение на валидационном наборе M14, ансамбль cross-YOLO и сопоставление с опубликованными результатами.

В результате исследования получена развёрнутая система Canopy (FastAPI + React + SQLite): лучшая модель YOLOv8x-seg достигает Box mAP@50 = 0,315 (+140 % к базовой версии); участок 1 × 1 км обрабатывается примерно за 18 секунд на одном ноутбучном GPU с экспортом в GeoJSON, CSV и HTML.

\newpage

# ABSTRACT

**Diploma Project Topic:** "Development of a Deep Learning Model for Automated Tree Recognition and Green Space Mapping in Urban Environments".

**Students:** Totin Anuar, Aidarkhanov Rasul, Sharipov Berik — Educational Program 6B06101 "Computer Science". **Scientific Supervisor:** Syndar Satbayev.

The relevance of the diploma project topic lies in the fact that urban tree inventories are essential for city planning, environmental monitoring and climate adaptation, yet traditional field surveys are slow, expensive and quickly obsolete, while the scientific literature contains no evaluation of modern tree-detection models on Central-Asian satellite imagery. The aim of the study is to design, implement and evaluate an end-to-end system that, given a satellite image of an area of Astana, automatically produces a tree inventory with a polygon mask, confidence score and geographic coordinates. The object of the research is the urban green space of Astana — the individual trees observable from above in the visible spectrum. The subject of the research is the deep-learning models and software components capable of detecting trees, segmenting crowns and converting coordinates. The scientific novelty of the study consists in the fact that it is the first published evaluation of state-of-the-art tree-detection models on Astana imagery (Box mAP@50 = 0.012 for the off-the-shelf DeepForest checkpoint), a four-model architecture with two ensembles, and a custom Astana dataset (≈ 100 images, ≈ 5,500 hand-labelled crowns, ≈ 8,700 after tiling). To achieve the aim, the following methods were used: literature analysis of 2019–2025, deep-learning training and fine-tuning (PyTorch, Ultralytics, torchvision, DeepForest, SAM 2), annotation in CVAT, and quantitative evaluation with COCO metrics.

The diploma project consists of 3 chapters across 83 pages and includes 17 figures and 13 tables.

The first chapter presents the subject-area analysis, a tabular literature review of 2019–2025, the identification of the geographic-generalisation gap for Central Asia, and the problem statement.

The second chapter is devoted to the methodology: the system architecture, the four models (YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest + SAM 2), data preparation, the ensembles, geographic conversion, persistence and the Canopy web application.

The third chapter provides the experiments and results: a 23-experiment hyperparameter ablation, the cross-model comparison on the M14 validation set, the cross-YOLO ensemble, and a comparison with published results.

As a result of the research, a deployed Canopy system (FastAPI + React + SQLite) was obtained: the best YOLOv8x-seg model reaches Box mAP@50 = 0.315 (+140 % over the baseline); a 1 × 1 km area is processed in about 18 seconds on a single laptop GPU, with export to GeoJSON, CSV and HTML.

\newpage

# АҢДАТПА

**Дипломдық жұмыс тақырыбы:** «Қалалық ортада ағаштарды автоматты түрде тану және жасыл желектерді картографиялау үшін терең оқыту моделін әзірлеу».

**Студенттер:** Тотин Ануар, Айдарханов Расул, Шарипов Берік — 6B06101 «Computer Science» білім беру бағдарламасы. **Ғылыми жетекшісі:** Сындар Сатбаев.

Дипломдық жұмыс тақырыбының өзектілігі қалалық ағаштар түгендеуінің қала жоспарлауы, экологиялық мониторинг және климатқа бейімделу үшін қажет екендігінде, алайда дәстүрлі далалық зерттеулер баяу, қымбат және тез ескіреді, ал ғылыми әдебиетте Орталық Азияның спутниктік суреттеріндегі заманауи ағаш анықтау модельдерінің бағасы жоқ. Зерттеудің мақсаты — Астананың аумағының спутниктік суреті бойынша полигондық маскасы, сенімділік бағасы және географиялық координаттары бар ағаштар түгендеуін автоматты түрде құрайтын ұштан-ұшқа жүйені жобалау, іске асыру және бағалау. Зерттеудің нысаны — Астананың қалалық жасыл желектері, яғни жоғарыдан көрінетін жекелеген ағаштар. Зерттеудің пәні — ағаштарды анықтауға, бұтақтарды сегменттеуге және координаттарды түрлендіруге қабілетті терең оқыту модельдері мен бағдарламалық компоненттер. Зерттеудің ғылыми жаңалығы — Астана суреттеріндегі заманауи ағаш анықтау модельдерінің алғашқы жарияланған бағасы (дайын DeepForest моделі үшін Box mAP@50 = 0,012), екі ансамбльді қамтитын төрт модельді архитектура және Астананың меншікті деректер жинағы (≈ 100 сурет, қолмен белгіленген ≈ 5 500 бұтақ, тайлдардан кейін ≈ 8 700). Мақсатқа жету үшін келесі әдістер қолданылды: 2019–2025 жж. әдебиетті талдау, модельдерді оқыту және қосымша оқыту (PyTorch, Ultralytics, torchvision, DeepForest, SAM 2), CVAT-та белгілеу және COCO метрикалары бойынша сандық бағалау.

Дипломдық жұмыс 83 беттен тұратын 3 тараудан, сондай-ақ 17 сызба мен 13 кестеден тұрады.

Бірінші тарауда пәндік саланы талдау, 2019–2025 жж. әдебиетке кестелік шолу, Орталық Азия үшін географиялық жалпылау алшақтығын анықтау және есептің қойылымы көрсетілген.

Екінші тарау әдіснамаға арналған: жүйе архитектурасы, төрт модель (YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest + SAM 2), деректерді дайындау, ансамбльдер, географиялық түрлендіру, сақтау және Canopy веб-қосымшасы.

Үшінші тарауда эксперименттер мен нәтижелер ұсынылған: гиперпараметрлерді таңдаудың 23 эксперименті, M14 валидациялық жиынтығындағы кросс-модельдік салыстыру, cross-YOLO ансамблі және жарияланған нәтижелермен салыстыру.

Зерттеу нәтижесінде Canopy жүйесі (FastAPI + React + SQLite) алынды: үздік YOLOv8x-seg моделі Box mAP@50 = 0,315 көрсетеді (+140 % базалық нұсқаға); 1 × 1 км аумақ бір ноутбук GPU-да шамамен 18 секундта өңделеді, нәтиже GeoJSON, CSV және HTML форматтарына экспортталады.
