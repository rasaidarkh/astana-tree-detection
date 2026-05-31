# АННОТАЦИЯ

**Тема дипломной работы:** «Разработка модели глубокого обучения для автоматического распознавания деревьев и картографирования зелёных насаждений в городской среде».

**Студенты:** Тотин Ануар, Айдарханов Расул, Шарипов Берик — образовательная программа 6B06101 «Computer Science». **Научный руководитель:** Сындар Сатбаев.

Актуальность темы дипломной работы состоит в том, что инвентаризация городских деревьев необходима для городского планирования, экологического мониторинга и адаптации к изменению климата, однако традиционные полевые обследования медленны, дороги и быстро устаревают, тогда как в научной литературе отсутствует оценка современных моделей обнаружения деревьев на спутниковых снимках Центральной Азии. Цель работы — спроектировать, реализовать и оценить сквозную систему, которая по спутниковому снимку участка Астаны автоматически формирует инвентаризацию деревьев с полигональной маской, оценкой уверенности и географическими координатами. Объект исследования — городские зелёные насаждения Астаны, то есть отдельные деревья, наблюдаемые сверху в видимом спектре. Предмет исследования — модели глубокого обучения и программные компоненты для обнаружения деревьев, сегментации крон и преобразования координат. Научная новизна работы состоит в следующем: (i) впервые измерена величина разрыва географической генерализации для региона — Box mAP@50 = 0,012 для готовой модели DeepForest (предобучена на NEON) на снимках Астаны; (ii) предложена четырёхмодельная архитектура (YOLOv8-seg, Mask R-CNN, дообученная DeepForest, zero-shot SAM 2) с двумя ансамблями — Weighted-Box-Fusion и голосование cross-YOLO (оба реализованы в системе, их количественная оценка на M14 оставлена для будущей работы); (iii) собран собственный набор данных Астаны (≈ 100 исходных снимков, ≈ 5 500 размеченных вручную полигонов крон, ≈ 8 700 экземпляров полигонов после нарезки на тайлы скользящим окном). Для достижения цели использованы методы анализа литературы 2019–2025 гг., обучения и дообучения моделей (PyTorch, Ultralytics, torchvision, DeepForest, SAM 2), разметки в CVAT с предразметкой моделью в цикле и количественной оценки по метрикам COCO на едином кросс-модельном валидационном наборе из 14 снимков (M14, 702 полигона), который служит общей эталонной разметкой для всех ветвей.

Дипломная работа содержит 3 главы на 83 страницах и имеет 17 рисунков, а также 13 таблиц.

В первой главе отражены анализ предметной области, обзор литературы 2019–2025 гг. в табличной форме, выявление разрыва географической генерализации для Центральной Азии и постановка задачи.

Вторая глава посвящена методам: архитектуре системы, четырём моделям (YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest + SAM 2), подготовке данных, ансамблям, географическому преобразованию, хранению и веб-приложению Canopy.

В третьей главе представлены эксперименты и результаты: серия из 23 экспериментов по подбору гиперпараметров, кросс-модельное сравнение на валидационном наборе M14 (702 полигона) и сопоставление с опубликованными результатами.

В результате исследования получена развёрнутая система Canopy (FastAPI + React + SQLite): отбор по итогам 23 экспериментов выбрал рабочую конфигурацию — YOLOv8x-seg, обученную от весов COCO со стандартной аугментацией Ultralytics, которая достигает Box mAP@50 = 0,315 и Mask mAP@50 = 0,289 на M14 (+140 % к базовой версии YOLO v1); далее следуют Mask R-CNN (0,166 / 0,158) и DeepForest + SAM 2 (0,146 / 0,134). Участок 1 × 1 км при зуме 19 обрабатывается примерно за 18 секунд на одном ноутбучном GPU, а инвентаризация экспортируется в форматы GeoJSON, CSV и автономный HTML.

\newpage

# ABSTRACT

**Diploma Project Topic:** "Development of a Deep Learning Model for Automated Tree Recognition and Green Space Mapping in Urban Environments".

**Students:** Totin Anuar, Aidarkhanov Rasul, Sharipov Berik — Educational Program 6B06101 "Computer Science". **Scientific Supervisor:** Syndar Satbayev.

The relevance of the diploma project topic lies in the fact that urban tree inventories are essential for city planning, environmental monitoring and climate adaptation, yet traditional field surveys are slow, expensive and quickly obsolete, while the scientific literature contains no evaluation of modern tree-detection models on Central-Asian satellite imagery. The aim of the study is to design, implement and evaluate an end-to-end system that, given a satellite image of an area of Astana, automatically produces a tree inventory with a polygon mask, confidence score and geographic coordinates. The object of the research is the urban green space of Astana — the individual trees observable from above in the visible spectrum. The subject of the research is the deep-learning models and software components capable of detecting trees, segmenting crowns and converting coordinates. The scientific novelty of the study consists in the following: (i) the first measured magnitude of the geographic-generalisation gap for the region — Box mAP@50 = 0.012 for the off-the-shelf NEON-pretrained DeepForest checkpoint on Astana; (ii) a four-model architecture (YOLOv8-seg, Mask R-CNN, fine-tuned DeepForest, zero-shot SAM 2) with two ensembles — a Weighted-Box-Fusion ensemble and a cross-YOLO voting ensemble (both implemented in the system, their quantitative M14 evaluation left to future work); and (iii) a custom Astana dataset of ≈ 100 source images with ≈ 5,500 hand-labelled tree-crown polygons (≈ 8,700 polygon instances after sliding-window tiling). To achieve the aim, the following methods were used: literature analysis of 2019–2025, deep-learning training and fine-tuning (PyTorch, Ultralytics, torchvision, DeepForest, SAM 2), annotation in CVAT with model-in-the-loop pre-labelling, and quantitative evaluation with COCO metrics on a single 14-image cross-model validation set (M14, 702 polygons) shared as common ground truth by all branches.

The diploma project consists of 3 chapters across 83 pages and includes 17 figures and 13 tables.

The first chapter presents the subject-area analysis, a tabular literature review of 2019–2025, the identification of the geographic-generalisation gap for Central Asia, and the problem statement.

The second chapter is devoted to the methods: the system architecture, the four models (YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest + SAM 2), data preparation, the ensembles, geographic conversion, persistence and the Canopy web application.

The third chapter provides the experiments and results: a 23-experiment hyperparameter ablation, the cross-model comparison on the M14 validation set (702 polygons), and a comparison with published results.

As a result of the research, a deployed Canopy system (FastAPI + React + SQLite) was obtained: a 23-experiment ablation selected the production configuration — a YOLOv8x-seg checkpoint trained from public COCO weights with the Ultralytics default augmentation — reaching Box mAP@50 = 0.315 and Mask mAP@50 = 0.289 on M14 (+140 % over the YOLO v1 baseline), followed by Mask R-CNN (0.166 / 0.158) and DeepForest + SAM 2 (0.146 / 0.134). A 1 km × 1 km capture at zoom 19 is processed in about 18 seconds on a single laptop GPU, and the inventory is exported as GeoJSON, CSV and standalone HTML.

\newpage

# АҢДАТПА

**Дипломдық жұмыс тақырыбы:** «Қалалық ортада ағаштарды автоматты түрде тану және жасыл желектерді картографиялау үшін терең оқыту моделін әзірлеу».

**Студенттер:** Тотин Ануар, Айдарханов Расул, Шарипов Берік — 6B06101 «Computer Science» білім беру бағдарламасы. **Ғылыми жетекшісі:** Сындар Сатбаев.

Дипломдық жұмыс тақырыбының өзектілігі қалалық ағаштар түгендеуінің қала жоспарлауы, экологиялық мониторинг және климатқа бейімделу үшін қажет екендігінде, алайда дәстүрлі далалық зерттеулер баяу, қымбат және тез ескіреді, ал ғылыми әдебиетте Орталық Азияның спутниктік суреттеріндегі заманауи ағаш анықтау модельдерінің бағасы жоқ. Зерттеудің мақсаты — Астананың аумағының спутниктік суреті бойынша полигондық маскасы, сенімділік бағасы және географиялық координаттары бар ағаштар түгендеуін автоматты түрде құрайтын ұштан-ұшқа жүйені жобалау, іске асыру және бағалау. Зерттеудің нысаны — Астананың қалалық жасыл желектері, яғни жоғарыдан көрінетін жекелеген ағаштар. Зерттеудің пәні — ағаштарды анықтауға, ағаш бастарын сегменттеуге және координаттарды түрлендіруге қабілетті терең оқыту модельдері мен бағдарламалық компоненттер. Зерттеудің ғылыми жаңалығы мынада: (i) аймақ үшін географиялық жалпылау алшақтығының шамасы алғаш рет өлшенді — Астана суреттерінде дайын DeepForest моделі (NEON-да алдын ала оқытылған) үшін Box mAP@50 = 0,012; (ii) екі ансамбльді қамтитын төрт модельді архитектура ұсынылды (YOLOv8-seg, Mask R-CNN, қосымша оқытылған DeepForest, zero-shot SAM 2) — Weighted-Box-Fusion және cross-YOLO дауыс беру ансамблі (екеуі де жүйеде іске асырылған, олардың M14 бойынша сандық бағасы болашақ жұмысқа қалдырылған); (iii) Астананың меншікті деректер жинағы жиналды (≈ 100 бастапқы сурет, қолмен белгіленген ≈ 5 500 ағаш басы полигоны, жылжымалы тереземен тайлдарға бөлгеннен кейін ≈ 8 700 полигон данасы). Мақсатқа жету үшін келесі әдістер қолданылды: 2019–2025 жж. әдебиетті талдау, модельдерді оқыту және қосымша оқыту (PyTorch, Ultralytics, torchvision, DeepForest, SAM 2), модельмен алдын ала белгілеу циклін қолдана отырып CVAT-та белгілеу және барлық тармақтар үшін ортақ эталондық белгілеу ретіндегі 14 суреттен тұратын бірыңғай кросс-модельдік валидациялық жиынтықта (M14, 702 полигон) COCO метрикалары бойынша сандық бағалау.

Дипломдық жұмыс 83 беттен тұратын 3 тараудан, сондай-ақ 17 сызба мен 13 кестеден тұрады.

Бірінші тарауда пәндік саланы талдау, 2019–2025 жж. әдебиетке кестелік шолу, Орталық Азия үшін географиялық жалпылау алшақтығын анықтау және есептің қойылымы көрсетілген.

Екінші тарау әдістерге арналған: жүйе архитектурасы, төрт модель (YOLOv8-seg, Mask R-CNN, DeepForest, DeepForest + SAM 2), деректерді дайындау, ансамбльдер, географиялық түрлендіру, сақтау және Canopy веб-қосымшасы.

Үшінші тарауда эксперименттер мен нәтижелер ұсынылған: гиперпараметрлерді таңдаудың 23 эксперименті, M14 валидациялық жиынтығындағы (702 полигон) кросс-модельдік салыстыру және жарияланған нәтижелермен салыстыру.

Зерттеу нәтижесінде Canopy жүйесі (FastAPI + React + SQLite) алынды: 23 эксперименттің қорытындысы бойынша жұмыс конфигурациясы таңдалды — COCO салмақтарынан Ultralytics стандартты аугментациясымен оқытылған YOLOv8x-seg, ол M14-те Box mAP@50 = 0,315 және Mask mAP@50 = 0,289 көрсетеді (YOLO v1 базалық нұсқасына +140 %); одан кейін Mask R-CNN (0,166 / 0,158) және DeepForest + SAM 2 (0,146 / 0,134) орналасады. 1 × 1 км аумақ 19-зум кезінде бір ноутбук GPU-да шамамен 18 секундта өңделеді, ал түгендеу GeoJSON, CSV және дербес HTML форматтарына экспортталады.
