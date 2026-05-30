# Abstract

**Topic:** Development of a Deep Learning Model for Automated Tree Recognition and Green Space Mapping in Urban Environments. **Authors:** Anuar Totin, Rasul Aidarkhanov, Berik Sharipov. Educational program: 6B06101 — Computer Science. Scientific supervisor: Syndar Satbayev.

**Relevance.** Urban tree inventories are a basic input to municipal planning, environmental monitoring and climate adaptation, but the manual field surveys that traditionally supply them are slow, expensive and quickly become obsolete for a city of Astana's size. Deep-learning detectors now work on freely-available satellite imagery at the per-tree level — yet the published literature does not contain a single evaluation of these models on Central-Asian imagery.

**Aim.** To design, implement and evaluate an end-to-end system that, given a satellite image of an area of Astana, automatically produces an inventory of its trees with a per-tree polygon mask, confidence score and geographic coordinates.

**Scientific novelty.** This is the first published evaluation of state-of-the-art deep-learning tree-detection models on Astana satellite imagery, and on any major Central-Asian capital. The work delivers (i) the first measured magnitude of the geographic-generalisation gap for the region — Box mAP@50 = **0.012** for the off-the-shelf NEON-pretrained DeepForest checkpoint on Astana; (ii) a four-model architecture combining YOLOv8-seg, Mask R-CNN, fine-tuned DeepForest and zero-shot SAM 2 mask refinement, combined through a Weighted-Box-Fusion ensemble and a novel cross-YOLO voting ensemble; and (iii) a custom Astana dataset of ≈ 100 source images with ≈ 5 500 hand-labelled tree-crown polygons (≈ 8 700 polygon instances after sliding-window tiling).

**Methods.** Analysis of peer-reviewed publications of 2019–2025; deep-learning training and fine-tuning in PyTorch, Ultralytics, torchvision, DeepForest and SAM 2; dataset engineering in CVAT with model-in-the-loop pre-labelling; and quantitative evaluation with standard COCO metrics on a single 14-image cross-model validation set (M14, 702 polygons) ensuring an apples-to-apples comparison between all branches.

**Results.** A 23-experiment hyperparameter ablation selected the production configuration: a YOLOv8x-seg checkpoint trained from public COCO weights with the Ultralytics default augmentation pipeline, reaching Box mAP@50 = **0.315** and Mask mAP@50 = **0.289** on M14 — a +140 % relative improvement over the YOLO v1 baseline. Mask R-CNN (0.166 / 0.158) and DeepForest + SAM 2 (0.146 / 0.134) follow. The deployed prototype — *Canopy*, a FastAPI + React + SQLite web application — processes a 1 km × 1 km capture at zoom 19 in about 18 seconds on a single laptop GPU and exports the inventory as GeoJSON, CSV and standalone HTML.

**Keywords:** Astana, tree detection, YOLO, Mask R-CNN, DeepForest, SAM 2, deep learning, remote sensing, urban forestry, instance segmentation, geographic-generalisation gap, ensemble.

\newpage

# Аннотация

**Тема:** Разработка модели глубокого обучения для автоматического распознавания деревьев и картографирования зелёных насаждений в городской среде. **Авторы:** Тотин Ануар, Айдарханов Расул, Шарипов Берик. Образовательная программа: 6B06101 — Computer Science. Научный руководитель: Сындар Сатбаев.

**Актуальность.** Инвентаризация городских деревьев — базовый элемент городского планирования, экологического мониторинга и адаптации к изменению климата, однако традиционные полевые обследования медленны, дороги и быстро устаревают для города размером с Астану. Современные модели глубокого обучения уже способны обнаруживать отдельные деревья на общедоступных спутниковых снимках — но в литературе нет ни одной оценки этих моделей на снимках Центральной Азии.

**Цель.** Спроектировать, реализовать и оценить сквозную систему, которая по спутниковому снимку участка Астаны автоматически формирует инвентаризацию деревьев с полигональной маской, оценкой уверенности и географическими координатами для каждого дерева.

**Научная новизна.** Это первая опубликованная оценка современных моделей обнаружения деревьев на спутниковых снимках Астаны и любой крупной столицы Центральной Азии. Работа даёт: (i) первое измерение величины разрыва географической генерализации для региона — Box mAP@50 = **0,012** для готовой модели DeepForest (NEON) на Астане; (ii) четырёхмодельную архитектуру — YOLOv8-seg, Mask R-CNN, дообученный DeepForest и уточнение масок SAM 2 без обучения — объединённую через ансамбли Weighted Box Fusion и оригинальное голосование cross-YOLO; (iii) собственный набор данных Астаны: ≈ 100 исходных снимков с ≈ 5 500 размеченными вручную полигонами крон (≈ 8 700 экземпляров после нарезки на тайлы).

**Методы.** Анализ рецензируемых публикаций 2019–2025 гг.; обучение и дообучение моделей в PyTorch, Ultralytics, torchvision, DeepForest и SAM 2; разметка в CVAT с предразметкой «модель в цикле»; количественная оценка по стандартным метрикам COCO на едином валидационном наборе из 14 снимков (M14, 702 полигона), обеспечивающем сопоставимость всех ветвей.

**Результаты.** Серия из 23 экспериментов по подбору гиперпараметров определила рабочую конфигурацию: YOLOv8x-seg, обученная с весов COCO с аугментацией Ultralytics по умолчанию, достигает Box mAP@50 = **0,315** и Mask mAP@50 = **0,289** на M14 — относительный прирост +140 % к базовой версии v1. Далее следуют Mask R-CNN (0,166 / 0,158) и DeepForest + SAM 2 (0,146 / 0,134). Развёрнутый прототип *Canopy* (веб-приложение FastAPI + React + SQLite) обрабатывает участок 1 × 1 км на зуме 19 примерно за 18 секунд на одном ноутбучном GPU и экспортирует результат в GeoJSON, CSV и автономный HTML.

**Ключевые слова:** Астана, обнаружение деревьев, YOLO, Mask R-CNN, DeepForest, SAM 2, глубокое обучение, дистанционное зондирование, городское озеленение, сегментация экземпляров, разрыв географической генерализации, ансамбль.

\newpage

# Аңдатпа

**Тақырыбы:** Қалалық ортада ағаштарды автоматты түрде тану және жасыл желектерді картографиялау үшін терең оқыту моделін әзірлеу. **Авторлар:** Тотин Ануар, Айдарханов Расул, Шарипов Берік. Білім беру бағдарламасы: 6B06101 — Computer Science. Ғылыми жетекшісі: Сындар Сатбаев.

**Өзектілігі.** Қалалық ағаштар түгендеуі қала жоспарлауының, экологиялық мониторингтің және климатқа бейімделудің негізгі құрамдасы болып табылады, бірақ дәстүрлі далалық зерттеулер баяу, қымбат және Астана көлеміндегі қала үшін тез ескіреді. Қазіргі терең оқыту модельдері қолжетімді спутниктік суреттерден жекелеген ағаштарды анықтай алады — алайда әдебиетте бұл модельдердің Орталық Азия суреттеріндегі бірде-бір бағасы жоқ.

**Мақсаты.** Астананың аумағының спутниктік суреті бойынша әр ағаштың полигондық маскасымен, сенімділік бағасымен және географиялық координаттарымен ағаштар түгендеуін автоматты түрде құрайтын ұштан-ұшқа жүйені жобалау, іске асыру және бағалау.

**Ғылыми жаңалығы.** Бұл — Астананың спутниктік суреттеріндегі және кез келген ірі Орталық Азия астанасындағы заманауи ағаш анықтау модельдерінің алғашқы жарияланған бағасы. Жұмыс мыналарды ұсынады: (i) аймақ үшін географиялық жалпылау алшақтығының алғашқы өлшемі — Астанада дайын DeepForest (NEON) моделі үшін Box mAP@50 = **0,012**; (ii) YOLOv8-seg, Mask R-CNN, қосымша оқытылған DeepForest және SAM 2 маскаларын нақтылауды біріктіретін, Weighted Box Fusion және өзіндік cross-YOLO дауыс беру ансамбльдері арқылы біріктірілген төрт модельді архитектура; (iii) Астананың меншікті деректер жинағы: ≈ 100 бастапқы сурет, қолмен белгіленген ≈ 5 500 ағаш бұтағының полигоны (тайлдарға бөлгеннен кейін ≈ 8 700 дана).

**Әдістері.** 2019–2025 жж. рецензияланған басылымдарды талдау; PyTorch, Ultralytics, torchvision, DeepForest және SAM 2 ішінде модельдерді оқыту мен қосымша оқыту; CVAT-та «циклдегі модель» алдын ала белгілеуімен деректерді дайындау; барлық тармақтарды салыстыруға мүмкіндік беретін 14 суреттен тұратын біртұтас валидациялық жиынтықта (M14, 702 полигон) стандартты COCO метрикаларымен сандық бағалау.

**Нәтижелері.** Гиперпараметрлерді таңдаудың 23 эксперименті өндірістік конфигурацияны анықтады: COCO салмақтарынан Ultralytics әдепкі аугментациясымен оқытылған YOLOv8x-seg моделі M14-те Box mAP@50 = **0,315** және Mask mAP@50 = **0,289** көрсетеді — v1 базалық нұсқасына қатысты +140 % салыстырмалы өсім. Одан кейін Mask R-CNN (0,166 / 0,158) және DeepForest + SAM 2 (0,146 / 0,134). Енгізілген *Canopy* прототипі (FastAPI + React + SQLite веб-қосымшасы) 19-зумдағы 1 × 1 км аумақты бір ноутбук GPU-да шамамен 18 секундта өңдейді және нәтижені GeoJSON, CSV және дербес HTML форматтарында экспорттайды.

**Түйін сөздер:** Астана, ағаштарды анықтау, YOLO, Mask R-CNN, DeepForest, SAM 2, терең оқыту, қашықтықтан зондтау, қалалық көгалдандыру, даналарды сегменттеу, географиялық жалпылау алшақтығы, ансамбль.

\newpage
