"""Run the whole offline pipeline: split -> train -> evaluate -> SHAP.

Usage:  python -m src.pipeline [--trials 20]
"""
import argparse

from src.data.split import make_splits
from src.explain import shap_utils
from src.models import evaluate, train


def main(trials: int) -> None:
    print("=" * 60, "\n1/4  Data split\n" + "=" * 60)
    make_splits()
    print("\n" + "=" * 60, "\n2/4  Training + MLflow tracking\n" + "=" * 60)
    train.main(trials)
    print("\n" + "=" * 60, "\n3/4  Evaluation + threshold\n" + "=" * 60)
    evaluate.main()
    print("\n" + "=" * 60, "\n4/4  SHAP explainability\n" + "=" * 60)
    shap_utils.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    main(parser.parse_args().trials)