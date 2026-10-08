import argparse

from hintrag.config import load_config
from hintrag.pipeline import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train HINT-RAG")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.opts)
    run_training(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
