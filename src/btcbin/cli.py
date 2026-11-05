"""Command line for preparing data and running the study."""

from __future__ import annotations

import argparse
from pathlib import Path

from btcbin import binary, polymarket, vol


def main() -> None:
    parser = argparse.ArgumentParser(prog="btcbin", description="15-minute bitcoin binaries.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-data", help="Download Binance, Deribit and Polymarket data.")
    b.add_argument("--out", type=Path, default=Path("data/real"))
    b.add_argument("--parts", nargs="+", default=["binance", "deribit", "polymarket"], choices=["binance", "deribit", "polymarket"])
    s = sub.add_parser("sample-data", help="Write synthetic inputs for testing the study.")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--seed", type=int, default=42)
    r = sub.add_parser("run", help="Run a part of the study and write report tables.")
    r.add_argument("part", choices=["vol", "binary", "polymarket", "all"])
    r.add_argument("--sample", choices=["discovery", "holdout", "posthoc"], default="discovery",
                   help="Polymarket sample; the holdout is run once, after the protocol is committed; "
                        "posthoc runs the checks added after the holdout.")
    r.add_argument("--data", type=Path, default=Path("data/real"))
    r.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.command == "build-data":
        from btcbin.data.build import build

        build(args.out, tuple(args.parts))
        return
    if args.command == "sample-data":
        from btcbin.data.synthetic import write_sample

        if args.seed < 0:
            parser.error("--seed must be non-negative")
        write_sample(args.out, args.seed)
        print(f"synthetic data written to {args.out}")
        return
    work = args.data / "work"
    if args.part in ("vol", "all"):
        vol.run(args.data, args.out / "vol", work)
        print(f"vol: tables written to {args.out / 'vol'}")
    if args.part in ("binary", "all"):
        binary.run(work, args.out / "binary")
        print(f"binary: tables written to {args.out / 'binary'}")
    if args.part in ("polymarket", "all"):
        polymarket.run(args.data, work, args.out / "polymarket" / args.sample, args.sample)
        print(f"polymarket ({args.sample}): tables written to {args.out / 'polymarket' / args.sample}")
    if args.part == "all":
        from btcbin import figures

        figures.render(args.out)


if __name__ == "__main__":
    main()
