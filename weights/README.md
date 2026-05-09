# Model weights

Файлы весов в git **не коммитятся** (gitignored). Сюда нужно положить:

| Файл | Источник | Размер | Назначение |
|---|---|---|---|
| `yolo_satellite.pt` | `pipeline/yolov8seg/runs/segment/train/weights/best.pt` | ~50 MB | YOLOAdapter → backend |
| `deepforest_astana.pl` | `deepforest/models/astana_trees_v4_10epochs.pl` | ~600 MB | DeepForestAdapter (опционально) |

Если `deepforest_astana.pl` нет — DeepForestAdapter автоматически загрузит pretrained `weecology/deepforest-tree`.

## Команды копирования (Windows PowerShell)

```powershell
# С существующих результатов 70%-защиты
Copy-Item "C:\Users\Rasul\DeepLearning\pipeline\yolov8seg\runs\segment\train\weights\best.pt" `
          ".\weights\yolo_satellite.pt"

# Опционально: дообученный DeepForest
Copy-Item "C:\Users\Rasul\DeepLearning\deepforest\models\astana_trees_v4_10epochs.pl" `
          ".\weights\deepforest_astana.pl"
```

После копирования перезапусти бэкенд — он подхватит веса автоматически.

## После новой тренировки

После `python ml/train_yolo.py ...` лучшие веса оказываются в `runs/segment/<name>/weights/best.pt`.
Скопируй их сюда:

```powershell
Copy-Item ".\runs\segment\astana_v2\weights\best.pt" ".\weights\yolo_satellite.pt"
```

Backend подхватит при следующем `/api/predict`.
