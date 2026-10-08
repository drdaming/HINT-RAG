import argparse

from hintrag.config import load_config
from hintrag.pipeline import run_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate HINT-RAG")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.opts)
    run_evaluation(cfg, args.checkpoint, split=args.split, output=args.output)


if __name__ == "__main__":
    main()
