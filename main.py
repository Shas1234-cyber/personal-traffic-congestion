"""
main.py
-------
Master pipeline – runs all four steps in sequence:
  1. Data preprocessing
  2. Feature engineering
  3. Model training & evaluation
  4. Visualization

Usage:
    python main.py                    # full pipeline
    python main.py --skip-viz         # skip chart generation
    python main.py --only-train       # preprocessing + features + train
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Put src on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_preprocessing  import run_preprocessing
from feature_engineering import run_feature_engineering
from model_training      import run_training
from visualization       import run_visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Traffic Congestion Prediction Pipeline")
    parser.add_argument("--skip-viz",   action="store_true", help="Skip visualization step")
    parser.add_argument("--only-train", action="store_true", help="Run only pre/feat/train")
    args = parser.parse_args()

    t_start = time.time()
    log.info("▶  Starting Traffic Congestion Prediction Pipeline")
    log.info("=" * 60)

    # Step 1 – Preprocessing
    run_preprocessing()

    # Step 2 – Feature engineering
    run_feature_engineering()

    # Step 3 – Model training
    results = run_training()

    # Step 4 – Visualizations (optional)
    if not args.skip_viz and not args.only_train:
        run_visualization(results)

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info(f"✅  Pipeline complete in {elapsed:.1f}s")
    log.info("Next step → launch the UI:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
