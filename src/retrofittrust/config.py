"""Project paths and constants."""

from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
SQLITE_PATH = DATA_PROCESSED / "twin_state.db"
LEDGER_PATH = DATA_PROCESSED / "ledger.json"

# Birmingham local authority name as it appears in EPC bulk data
BIRMINGHAM_LA = "Birmingham"

# Default composite target weights (see CURSOR_BUILD_SPEC open items)
EPC_GAP_WEIGHT = 0.6
IMD_INCOME_WEIGHT = 0.4

# Demo cohort size for integration loop
DEMO_COHORT_LSOA_COUNT = 10
