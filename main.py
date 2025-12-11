"""
Multimodal Audio Creator - Main Entry Point
Command-line interface for video-to-audio generation
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import validate_config, print_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def mode_vidmuse(args):
    """Run VidMuse video-to-music generation"""
    from src.modules.vidmuse import VidMuseGenerator

    print("\n" + "=" * 80)
    print("MODE: VidMuse - Video-to-Music Generation")
    print("=" * 80 + "\n")

    # Initialize generator
    generator = VidMuseGenerator(
        device="cuda" if args.device == "auto" else args.device,
        use_float16=not args.no_fp16,
        lora_path=args.vidmuse_lora,
    )

    # Generate music
    print(f"Processing video: {args.video_path}")
    audio = generator.generate(
        video_path=args.video_path,
        duration=args.duration,
        merge_video=args.merge_video,
        merged_video_path=args.merged_video_path,
        rerank=args.rerank,
    )

    # Save output
    generator.save(audio, args.output_path)

    print(f"\n[SUCCESS] Music generated successfully: {args.output_path}\n")

    # Cleanup
    if args.clear_memory:
        generator.clear_memory()


def mode_tts(args):
    """Run TTS text-to-speech generation"""
    from src.modules.tts import TTSEngine

    print("\n" + "=" * 80)
    print("MODE: TTS - Text-to-Speech Generation")
    print("=" * 80 + "\n")

    # Initialize TTS engine
    tts = TTSEngine(
        engine_type=args.tts_engine,
        device="cuda" if args.device == "auto" else args.device,
        use_float16=not args.no_fp16,
        model_name=args.tts_model,
    )

    print(f"TTS Engine: {tts.engine_type}")
    print(f"Synthesizing text: '{args.text[:50]}...'")
    if args.speaker:
        print(f"Speaker: {args.speaker}")

    # Synthesize speech
    tts.synthesize(
        text=args.text,
        output_path=args.output_path,
        speaker=args.speaker,
        language=args.language,
        speed=args.speed,
    )

    print(f"\n[SUCCESS] Speech generated successfully: {args.output_path}\n")

    # Cleanup
    if args.clear_memory:
        tts.clear_memory()


def mode_wavjourney(args):
    """Run WavJourney complete creative workflow"""
    from src.modules.wavjourney import AudioCreativePlanner

    print("\n" + "=" * 80)
    print("MODE: WavJourney - Creative Audio Planning")
    print("=" * 80 + "\n")

    # Initialize planner
    planner = AudioCreativePlanner(llm_model=args.llm_model, api_key=args.api_key)

    # Create audio
    print(f"Instruction: {args.instruction}")
    if args.video_path:
        print(f"Video: {args.video_path}")

    output_path, script = planner.create(
        instruction=args.instruction,
        video_path=args.video_path,
        output_path=args.output_path,
        merge_video=args.merge_video,
        merged_video_path=args.merged_video_path,
    )

    print(f"\n[SUCCESS] Audio created successfully: {output_path}")
    print(f"\nGenerated Script:")
    print("-" * 80)
    import json

    print(json.dumps(script, indent=2))
    print("-" * 80 + "\n")

    # Cleanup
    if args.clear_memory:
        planner.clear_memory()


def mode_mix(args):
    """Mix multiple audio files"""
    from src.utils.audio_mixer import mix_audio

    print("\n" + "=" * 80)
    print("MODE: Audio Mixing")
    print("=" * 80 + "\n")

    print(f"Mixing files:")
    print(f"  Music: {args.music_path} (volume: {args.music_volume})")
    print(f"  Speech: {args.speech_path} (volume: {args.speech_volume})")

    mixed = mix_audio(
        music_path=args.music_path,
        speech_path=args.speech_path,
        output_path=args.output_path,
        music_volume=args.music_volume,
        speech_volume=args.speech_volume,
        ducking_db=args.ducking_db,
        target_lufs=args.target_lufs,
        attack=args.duck_attack,
        release=args.duck_release,
    )

    print(f"\n[SUCCESS] Audio mixed successfully: {args.output_path}\n")


def mode_autopipeline(args):
    """Run full pipeline: caption -> music -> tts -> mix."""
    from src.modules.captioner import Captioner
    from src.modules.vidmuse import VidMuseGenerator
    from src.modules.vidmuse.video_processor import merge_video_audio
    from src.modules.tts import TTSEngine
    from src.utils.audio_mixer import mix_audio
    import soundfile as sf
    from moviepy.editor import VideoFileClip

    print("\n" + "=" * 80)
    print("MODE: AutoPipeline - Video -> Caption -> Music -> Narration -> Mix")
    print("=" * 80 + "\n")

    video_path = Path(args.video_path)
    final_out = Path(args.output_path)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    stem = final_out.stem
    caption_path = final_out.parent / f"{stem}_caption.txt"
    music_path = final_out.parent / f"{stem}_music.wav"
    narr_path = final_out.parent / f"{stem}_narr.wav"

    # Determine video duration to keep narration/music aligned
    with VideoFileClip(str(video_path)) as clip:
        video_duration = clip.duration
    run_duration = args.duration if args.duration else video_duration

    # 1) Caption
    captioner = Captioner(
        backend=args.caption_backend,
        max_len=args.caption_max_len,
        num_frames=args.caption_frames,
        device=args.device,
    )
    caption = captioner.generate(str(video_path))
    caption_path.write_text(caption, encoding="utf-8")
    print(f"Caption: {caption}")
    print(f"Caption saved to: {caption_path}")

    # 2) Music via VidMuse
    generator = VidMuseGenerator(
        device="cuda" if args.device == "auto" else args.device,
        use_float16=not args.no_fp16,
        lora_path=args.vidmuse_lora,
    )
    music_audio = generator.generate(
        video_path=str(video_path),
        duration=run_duration,
        merge_video=False,
        rerank=args.rerank,
    )
    generator.save(music_audio, music_path)
    print(f"Music saved to: {music_path}")

    # 3) Narration via TTS
    tts = TTSEngine(
        engine_type=args.tts_engine,
        device="cuda" if args.device == "auto" else args.device,
        use_float16=not args.no_fp16,
        model_name=args.tts_model,
    )
    tts.synthesize(
        text=caption,
        output_path=narr_path,
        speaker=args.speaker,
        language=args.language,
        speed=args.speed,
    )
    print(f"Narration saved to: {narr_path}")

    # Trim narration to video duration (avoid overshoot)
    narr_wave, narr_sr = sf.read(narr_path)
    max_len = int(video_duration * narr_sr)
    if len(narr_wave) > max_len:
        narr_wave = narr_wave[:max_len]
        sf.write(narr_path, narr_wave, narr_sr)
        print(f"Narration trimmed to {video_duration:.2f}s to match video length.")

    # 4) Mix with ducking
    mix_audio(
        music_path=str(music_path),
        speech_path=str(narr_path),
        output_path=str(final_out),
        music_volume=args.music_volume,
        speech_volume=args.speech_volume,
        ducking_db=args.ducking_db,
        target_lufs=args.target_lufs,
        attack=args.duck_attack,
        release=args.duck_release,
    )
    print(f"\n[SUCCESS] AutoPipeline audio complete: {final_out}")

    # 5) Optional mux back into video
    if args.merge_video:
        if not args.merged_video_path:
            raise ValueError("--merged-video-path is required when --merge-video is set")
        merged_path = Path(args.merged_video_path)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merge_video_audio(video_path, final_out, merged_path)
        print(f"[SUCCESS] Video+audio muxed to: {merged_path}\n")
    else:
        print()

    if args.clear_memory:
        generator.clear_memory()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multimodal Audio Creator - VidMuse + TTS + WavJourney",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["vidmuse", "tts", "wavjourney", "mix", "autopipeline"],
        help="Operation mode",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output audio file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable float16 precision"
    )
    parser.add_argument(
        "--clear-memory", action="store_true", help="Clear GPU memory after completion"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--show-config", action="store_true", help="Show configuration and exit"
    )

    # VidMuse arguments
    parser.add_argument("--video_path", type=str, help="Input video path (for vidmuse/wavjourney)")
    parser.add_argument("--duration", type=float, help="Music duration in seconds")
    parser.add_argument("--prompt", type=str, help="(Deprecated) Prompt override for legacy VidMuse stub")
    parser.add_argument(
        "--merge-video",
        action="store_true",
        help="Mux generated audio back into the input MP4 (vidmuse mode)",
    )
    parser.add_argument(
        "--merged-video-path",
        type=str,
        help="Destination MP4 when --merge-video is set",
    )
    parser.add_argument(
        "--vidmuse-lora",
        type=str,
        help="Optional path to a PEFT/LoRA checkpoint for VidMuse LM",
    )
    parser.add_argument(
        "--rerank",
        type=int,
        default=1,
        help="Number of VidMuse candidates to generate for AV reranking (default: 1 = disabled)",
    )

    # Captioner / autopipeline arguments
    parser.add_argument(
        "--caption-backend",
        type=str,
        default="openai",
        choices=["openai", "stub"],
        help="Backend for automatic captioning (autopipeline)",
    )
    parser.add_argument(
        "--caption-max-len",
        type=int,
        default=40,
        help="Max words for generated caption (autopipeline)",
    )
    parser.add_argument(
        "--caption-frames",
        type=int,
        default=4,
        help="Number of frames to sample for captioning (autopipeline)",
    )

    # TTS arguments
    parser.add_argument("--text", type=str, help="Text to synthesize (for tts)")
    parser.add_argument(
        "--tts-engine",
        type=str,
        default="auto",
        choices=["auto", "coqui", "pyttsx3"],
        help="TTS engine to use (for tts)",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        help="Optional TTS model name for Coqui (e.g., tts_models/multilingual/multi-dataset/xtts_v2)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        help="Speaker/voice to use (for tts)",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (e.g., 'en', 'zh-cn') (for tts)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (for tts)"
    )

    # WavJourney arguments
    parser.add_argument(
        "--instruction", type=str, help="Creative instruction (for wavjourney)"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4", help="LLM model (for wavjourney)"
    )
    parser.add_argument("--api-key", type=str, help="API key for LLM provider")

    # Mix arguments
    parser.add_argument("--music_path", type=str, help="Music file path (for mix)")
    parser.add_argument("--speech_path", type=str, help="Speech file path (for mix)")
    parser.add_argument(
        "--music_volume", type=float, default=0.6, help="Music volume (for mix)"
    )
    parser.add_argument(
        "--speech_volume", type=float, default=1.0, help="Speech volume (for mix)"
    )
    parser.add_argument(
        "--ducking-db",
        type=float,
        default=8.0,
        help="Sidechain ducking depth applied to music when speech is present (dB, default: 8)",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-18.0,
        help="Target integrated loudness for loudness matching / normalization (LUFS, default: -18)",
    )
    parser.add_argument(
        "--duck-attack",
        type=float,
        default=0.02,
        help="Ducking attack time in seconds (music fades down when speech starts; default: 0.02s)",
    )
    parser.add_argument(
        "--duck-release",
        type=float,
        default=0.2,
        help="Ducking release time in seconds (music fades back after speech; default: 0.2s)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Show configuration if requested
    if args.show_config:
        print_config()
        return

    # Validate configuration
    if not validate_config():
        print("\n[WARNING] Configuration validation failed!")
        print("Please check the errors above and fix your configuration.\n")
        sys.exit(1)

    # Route to appropriate mode
    try:
        if args.mode == "vidmuse":
            if not args.video_path:
                print("Error: --video_path is required for vidmuse mode")
                sys.exit(1)
            mode_vidmuse(args)

        elif args.mode == "tts":
            if not args.text:
                print("Error: --text is required for tts mode")
                sys.exit(1)
            mode_tts(args)

        elif args.mode == "wavjourney":
            if not args.instruction:
                print("Error: --instruction is required for wavjourney mode")
                sys.exit(1)
            mode_wavjourney(args)

        elif args.mode == "mix":
            if not args.music_path or not args.speech_path:
                print("Error: --music_path and --speech_path are required for mix mode")
                sys.exit(1)
            mode_mix(args)

        elif args.mode == "autopipeline":
            if not args.video_path:
                print("Error: --video_path is required for autopipeline mode")
                sys.exit(1)
            mode_autopipeline(args)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Error: {e}")
        import traceback

        if args.log_level == "DEBUG":
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
