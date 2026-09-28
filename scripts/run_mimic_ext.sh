AMA_DATA_PATH=""

PYTHONPATH=src python3 -m ama.importers.mimic_cdm \
  --source {$AMA_DATA_PATH}/raw/mimic-iv-ext-clinical-decision-making/ \
  --out {$AMA_DATA_PATH}/processed/mimic-iv-ext-clinical-decision-making/

ama run datasets/mimic_cdm/mimic_cdm_interactive --model glm-5.3-flash --episode 20000602 \
  --tools src/ama/importers/mimic_cdm_tools.py --max-calls 8 --runs-root runs