import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.embedding_eval import run_eval


def _run_single_video_eval(args):
    """Generate base + optional LoRA audio for one video, then score AV alignment and metrics."""
    from src.modules.vidmuse import VidMuseGenerator
    from src.utils.av_alignment import score as av_score

    device_for_gen = "cuda" if args.device == "auto" else args.device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    if not args.skip_base:
        variants.append(("base", None))
    if args.lora_path:
        variants.append(("lora", str(args.lora_path)))
    if not variants:
        raise ValueError("No variants to evaluate: enable base or provide --lora-path.")

    results = []
    for label, lora_path in variants:
        gen = VidMuseGenerator(device=device_for_gen, use_float16=not args.no_fp16, lora_path=lora_path)
        audio = gen.generate(
            video_path=args.video,
            duration=args.duration,
            merge_video=False,
            rerank=args.rerank,
        )
        wav_path = args.output_dir / f"{label}.wav"
        gen.save(audio, wav_path)
        score_val = av_score(str(args.video), str(wav_path), device=args.device)
        entry = {"variant": label, "audio": str(wav_path), "av_alignment": score_val}

        # Run embedding metrics for this single pair unless disabled
        if not args.no_single_metrics:
            tmp_manifest = args.output_dir / f"{label}_manifest.csv"
            tmp_manifest.write_text("video_path,audio_path\n", encoding="utf-8")
            with tmp_manifest.open("a", encoding="utf-8") as f:
                f.write(f"{args.video},{wav_path}\n")
            metrics = run_eval(
                manifest=tmp_manifest,
                metrics=args.metrics_set,
                reference_audio_dir=args.reference_audio_dir,
                device=args.device,
                compute_map_ndcg=args.compute_map_ndcg,
                fad_backend=args.fad_backend,
                num_video_frames=args.num_video_frames,
                audio_backend_name=args.audio_backend,
                audio_cache_dir=args.audio_cache_dir,
                save_embeddings=None,
                load_embeddings=None,
            )
            entry["metrics"] = metrics
        results.append(entry)
        gen.clear_memory()

    print("\n=== Single-Video AV Alignment ===")
    for r in results:
        print(f"{r['variant']:>5}: {r['av_alignment']:.4f} -> {r['audio']}")
    if len(results) == 2:
        delta = results[1]["av_alignment"] - results[0]["av_alignment"]
        verdict = "LoRA improves alignment" if delta > 0 else "LoRA does not improve alignment"
        print(f"\n{verdict} (delta {delta:+.4f})")

    # Pretty-print metrics if available
    for r in results:
        if "metrics" in r:
            print(f"\n--- Metrics: {r['variant']} ---")
            print(json.dumps(r["metrics"], indent=2))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, help="CSV with video_path,audio_path (required unless --video)")
    parser.add_argument("--video", type=Path, help="Single video to generate + evaluate (base and optional LoRA)")
    parser.add_argument("--lora-path", type=Path, help="LoRA checkpoint to compare against base")
    parser.add_argument("--output-dir", type=Path, default=Path("output/offline_eval_gen"), help="Where to save generated audio when using --video")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration override (seconds) for generation")
    parser.add_argument("--skip-base", action="store_true", help="Skip base VidMuse generation; only run LoRA")
    parser.add_argument("--rerank", type=int, default=1, help="Number of candidates for reranking (>=1)")
    parser.add_argument("--no-fp16", action="store_true", help="Force float32 generation")
    parser.add_argument("--no-single-metrics", action="store_true", help="In --video mode, skip embedding metrics and only print AV alignment")
    parser.add_argument(
        "--metrics",
        default="sim,retrieval,dist,rhythm",
        help="Comma-separated: sim,retrieval,dist,rhythm",
    )
    parser.add_argument("--reference-audio-dir", type=Path, help="Dir of reference real music for dist metric")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu for embeddings and AV score")
    parser.add_argument("--compute-map-ndcg", action="store_true", help="Compute mAP/nDCG (requires scikit-learn)")
    parser.add_argument(
        "--fad-backend",
        default="auto",
        help="auto|frechet_audio_distance|none; auto tries frechet_audio_distance if installed",
    )
    parser.add_argument("--save-embeddings", type=Path, help="Optional npz to save video/audio embeddings")
    parser.add_argument("--load-embeddings", type=Path, help="Optional npz to load precomputed embeddings")
    parser.add_argument("--num-video-frames", type=int, default=8, help="Frames to sample for video embeddings")
    parser.add_argument(
        "--audio-backend",
        default="clap",
        choices=["clap", "muq_mulan", "vggish", "vggsound"],
        help="Audio embedding backend for sim/retrieval/dist metrics.",
    )
    parser.add_argument("--audio-cache-dir", type=Path, help="Optional cache dir for MuQ/VGGish weights")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.video:
        args.metrics_set = set([m.strip() for m in args.metrics.split(",") if m.strip()])
        _run_single_video_eval(args)
        return

    if not args.manifest:
        parser.error("Either --manifest or --video must be provided.")

    metrics = set([m.strip() for m in args.metrics.split(",") if m.strip()])

    results = run_eval(
        manifest=args.manifest,
        metrics=metrics,
        reference_audio_dir=args.reference_audio_dir,
        device=args.device,
        compute_map_ndcg=args.compute_map_ndcg,
        fad_backend=args.fad_backend,
        num_video_frames=args.num_video_frames,
        audio_backend_name=args.audio_backend,
        audio_cache_dir=args.audio_cache_dir,
        save_embeddings=args.save_embeddings,
        load_embeddings=args.load_embeddings,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
