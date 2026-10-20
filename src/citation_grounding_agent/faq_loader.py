from pathlib import Path

from citation_grounding_agent.models import FAQDataset


def load_FAQDataset(json_path: Path) -> FAQDataset:

   read_json = json_path.read_text(encoding="utf-8")

   return FAQDataset.model_validate_json(read_json)



