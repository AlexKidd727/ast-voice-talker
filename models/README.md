# Папка models

Используемые артефакты:

- **t-one/** — ASR (распознавание речи): `model.onnx`, `kenlm.bin`. Используются при `ASR_MODEL_PATH` (в Docker: `./models/t-one` → `/app/asr-models`). Не класть сюда файлы из корня — пайплайн читает только эту папку.
- **v3_1_ru.pt, v5_ru.pt, v5_cis_base_nostress.pt** — модели Silero TTS. TTS-микросервис ищет их здесь или в `/app/models` (Docker), затем в кэше PyTorch.

Дубликаты: раньше в корне лежала копия `kenlm.bin` и файлы репозитория T-one (config, model.safetensors, tokenizer) — они не используются пайплайном (нужны только `t-one/model.onnx` и `t-one/kenlm.bin`), поэтому были удалены.
