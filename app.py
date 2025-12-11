"""
Web UI for Multimodal Audio Creator (FastAPI + Gradio).
Run:
    conda activate audio-creator
    python app.py
Then open http://127.0.0.1:7860/
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Callable, Any

import gradio as gr
from fastapi import FastAPI

from src.config import OUTPUT_DIR
from src.modules.captioner import Captioner
from src.modules.tts import TTSEngine
from src.modules.vidmuse import VidMuseGenerator
from src.modules.vidmuse.video_processor import merge_video_audio
from src.utils.audio_mixer import mix_audio


WEB_OUT_DIR = OUTPUT_DIR / "webui"
WEB_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _unique_stem(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def run_autopipeline(
    video_file: Path,
    caption_backend: str = "openai",
    caption_max_len: int = 32,
    caption_frames: int = 4,
    vidmuse_lora: Optional[str] = None,
    rerank: int = 1,
    ducking_db: float = 12.0,
    speed: float = 1.0,
    merge_video: bool = True,
    device: str = "auto",
    progress: Optional[Callable[[float, str], Any]] = None,
) -> tuple[str, str, str]:
    """
    Full chain: caption -> music -> tts -> mix -> optional mux.
    Returns (caption_text, audio_path, merged_video_path or "").
    """
    stem = _unique_stem("auto")
    caption_path = WEB_OUT_DIR / f"{stem}_caption.txt"
    music_path = WEB_OUT_DIR / f"{stem}_music.wav"
    narr_path = WEB_OUT_DIR / f"{stem}_narr.wav"
    out_wav = WEB_OUT_DIR / f"{stem}.wav"
    out_mp4 = WEB_OUT_DIR / f"{stem}.mp4"

    # 1) Caption
    if progress:
        progress(0.05, "Captioning...")
    captioner = Captioner(
        backend=caption_backend,
        max_len=caption_max_len,
        num_frames=caption_frames,
        device=device,
    )
    caption = captioner.generate(str(video_file))
    caption_path.write_text(caption, encoding="utf-8")

    # 2) Music
    if progress:
        progress(0.2, "Generating music...")
    vm = VidMuseGenerator(
        device="cuda" if device == "auto" else device,
        use_float16=True if device != "cpu" else False,
        lora_path=vidmuse_lora,
    )
    music_audio = vm.generate(
        video_path=str(video_file),
        duration=None,
        merge_video=False,
        rerank=rerank,
        rerank_topk=1,
        rerank_tempdir=None,
    )
    vm.save(music_audio, music_path)

    # 3) TTS
    if progress:
        progress(0.6, "Synthesizing narration...")
    tts = TTSEngine(
        engine_type="coqui",
        device="cuda" if device == "auto" else device,
        use_float16=True if device != "cpu" else False,
    )
    tts.synthesize(
        text=caption,
        output_path=narr_path,
        speaker=None,
        language=None,
        speed=speed,
    )

    # 4) Mix with ducking
    if progress:
        progress(0.8, "Mixing with ducking...")
    mix_audio(
        music_path=str(music_path),
        speech_path=str(narr_path),
        output_path=str(out_wav),
        music_volume=0.6,
        speech_volume=1.0,
        ducking_db=ducking_db,
        target_lufs=-18.0,
        attack=0.02,
        release=0.2,
    )

    merged = ""
    if merge_video:
        if progress:
            progress(0.9, "Muxing video...")
        merge_video_audio(Path(video_file), out_wav, out_mp4)
        merged = str(out_mp4)

    if progress:
        progress(1.0, "Done")
    return caption, str(out_wav), merged


def run_vidmuse(
    video_file: Path,
    vidmuse_lora: Optional[str],
    rerank: int,
    merge_video: bool,
    device: str,
    progress: Optional[Callable[[float, str], Any]] = None,
) -> tuple[str, str]:
    stem = _unique_stem("vidmuse")
    out_wav = WEB_OUT_DIR / f"{stem}.wav"
    out_mp4 = WEB_OUT_DIR / f"{stem}.mp4"

    if progress:
        progress(0.05, "Generating music...")
    vm = VidMuseGenerator(
        device="cuda" if device == "auto" else device,
        use_float16=True if device != "cpu" else False,
        lora_path=vidmuse_lora,
    )
    audio = vm.generate(
        video_path=str(video_file),
        duration=None,
        merge_video=False,
        rerank=rerank,
        rerank_topk=1,
        rerank_tempdir=None,
    )
    vm.save(audio, out_wav)
    merged = ""
    if merge_video:
        if progress:
            progress(0.6, "Muxing video...")
        merge_video_audio(Path(video_file), out_wav, out_mp4)
        merged = str(out_mp4)
    if progress:
        progress(1.0, "Done")
    return str(out_wav), merged


def run_tts(text: str, speed: float, device: str, progress: Optional[Callable[[float, str], Any]] = None) -> str:
    stem = _unique_stem("tts")
    out_wav = WEB_OUT_DIR / f"{stem}.wav"
    if progress:
        progress(0.05, "Synthesizing speech...")
    tts = TTSEngine(
        engine_type="coqui",
        device="cuda" if device == "auto" else device,
        use_float16=True if device != "cpu" else False,
    )
    tts.synthesize(text=text, output_path=out_wav, speed=speed)
    if progress:
        progress(1.0, "Done")
    return str(out_wav)


with gr.Blocks(title="Multimodal Audio Creator") as gradio_app:
    gr.Markdown("# 🎵 Multimodal Audio Creator\nVideo → Music/TTS/Mix in one place.")
    with gr.Tab("AutoPipeline"):
        with gr.Row():
            video_in = gr.Video(label="Video")
            with gr.Column():
                caption_backend = gr.Dropdown(
                    ["openai", "stub"], value="openai", label="Caption backend"
                )
                caption_len = gr.Slider(8, 64, value=28, step=1, label="Caption max words")
                caption_frames = gr.Slider(1, 8, value=4, step=1, label="Frames sampled")
                lora_path = gr.Textbox(label="VidMuse LoRA path (optional)", value="")
                rerank = gr.Slider(1, 3, value=1, step=1, label="VidMuse rerank N")
                duck_db = gr.Slider(6, 24, value=15, step=1, label="Ducking dB")
                speed = gr.Slider(0.7, 1.3, value=0.9, step=0.01, label="TTS speed")
                device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                merge_video = gr.Checkbox(value=True, label="Mux back to MP4")
        run_btn = gr.Button("Run AutoPipeline", variant="primary")
        caption_out = gr.Textbox(label="Caption", interactive=False)
        audio_out = gr.Audio(label="Mixed Audio (wav)", interactive=False)
        video_out = gr.Video(label="Merged Video (mp4)")
        auto_status = gr.Textbox(label="Status", interactive=False)
        auto_log = gr.Textbox(label="Log", interactive=False, lines=6)

        def _on_autopipeline(
            video, caption_backend, caption_len, caption_frames, lora_path, rerank, duck_db, speed, device, merge_video, progress=gr.Progress(track_tqdm=True)
        ):
            if video is None:
                return "Please upload a video.", None, None, "Error: missing video", "No video provided."
            log_lines = []
            try:
                log_lines.append("Captioning...")
                cap, wav, mp4 = run_autopipeline(
                    Path(video),
                    caption_backend=caption_backend,
                    caption_max_len=int(caption_len),
                    caption_frames=int(caption_frames),
                    vidmuse_lora=lora_path or None,
                    rerank=int(rerank),
                    ducking_db=float(duck_db),
                    speed=float(speed),
                    merge_video=bool(merge_video),
                    device=device,
                    progress=progress,
                )
                log_lines.append(f"Caption saved. Audio: {wav}")
                if mp4:
                    log_lines.append(f"Muxed video: {mp4}")
                else:
                    log_lines.append("Mux disabled.")
                return cap, wav, mp4 or None, "OK", "\n".join(log_lines)
            except Exception as e:  # pragma: no cover - UI path
                log_lines.append(f"Error: {e}")
                return f"Error: {e}", None, None, f"Error: {e}", "\n".join(log_lines)

        run_btn.click(
            _on_autopipeline,
            inputs=[video_in, caption_backend, caption_len, caption_frames, lora_path, rerank, duck_db, speed, device, merge_video],
            outputs=[caption_out, audio_out, video_out, auto_status, auto_log],
        )

    with gr.Tab("VidMuse only"):
        vid_video = gr.Video(label="Video")
        vid_lora = gr.Textbox(label="LoRA path (optional)", value="")
        vid_rerank = gr.Slider(1, 3, value=1, step=1, label="Rerank N")
        vid_merge = gr.Checkbox(value=True, label="Mux back to MP4")
        vid_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        vid_btn = gr.Button("Generate Music")
        vid_audio_out = gr.Audio(label="Music WAV", interactive=False)
        vid_video_out = gr.Video(label="Merged MP4")
        vid_status = gr.Textbox(label="Status", interactive=False)
        vid_log = gr.Textbox(label="Log", interactive=False, lines=4)

        def _on_vid(video, lora, rerank, merge_video, device, progress=gr.Progress(track_tqdm=True)):
            if video is None:
                return None, None, "Error: missing video", "No video provided."
            log_lines = []
            try:
                log_lines.append("Generating music...")
                wav, mp4 = run_vidmuse(Path(video), lora or None, int(rerank), bool(merge_video), device, progress=progress)
                log_lines.append(f"Music saved: {wav}")
                if mp4:
                    log_lines.append(f"Muxed video: {mp4}")
                else:
                    log_lines.append("Mux disabled.")
                return wav, mp4 or None, "OK", "\n".join(log_lines)
            except Exception as e:
                log_lines.append(f"Error: {e}")
                return None, None, f"Error: {e}", "\n".join(log_lines)

        vid_btn.click(_on_vid, inputs=[vid_video, vid_lora, vid_rerank, vid_merge, vid_device], outputs=[vid_audio_out, vid_video_out, vid_status, vid_log])

    with gr.Tab("TTS only"):
        tts_text = gr.Textbox(label="Text", value="Hello from Multimodal Audio Creator.")
        tts_speed = gr.Slider(0.7, 1.3, value=1.0, step=0.01, label="Speed")
        tts_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        tts_btn = gr.Button("Generate Speech")
        tts_audio_out = gr.Audio(label="Speech WAV", interactive=False)
        tts_status = gr.Textbox(label="Status", interactive=False)
        tts_log = gr.Textbox(label="Log", interactive=False, lines=3)

        def _on_tts(text, speed, device, progress=gr.Progress(track_tqdm=True)):
            try:
                log_lines = ["Synthesizing speech..."]
                wav = run_tts(text, float(speed), device, progress=progress)
                log_lines.append(f"Saved: {wav}")
                return wav, "OK", "\n".join(log_lines)
            except Exception as e:
                return None, f"Error: {e}", f"Error: {e}"

        tts_btn.click(_on_tts, inputs=[tts_text, tts_speed, tts_device], outputs=[tts_audio_out, tts_status, tts_log])


app = FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
