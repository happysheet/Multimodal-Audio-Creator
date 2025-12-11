"""
WavJourney Creative Planner
LLM-driven audio composition integrating VidMuse and TTS
"""

import json
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    # Optional import; only used when muxing video+audio
    from src.modules.vidmuse.video_processor import merge_video_audio
except Exception:  # pragma: no cover - defensive import
    merge_video_audio = None

try:
    from moviepy.editor import VideoFileClip
except Exception:  # pragma: no cover - moviepy is optional until needed
    VideoFileClip = None


class AudioCreativePlanner:
    """
    WavJourney-style creative audio planner
    Uses LLM to plan and coordinate audio generation
    """

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        """
        Initialize audio creative planner

        Args:
            llm_model: Model name
            api_key: API key for provider
        """
        from src.config import WAVJOURNEY_CONFIG

        self.config = WAVJOURNEY_CONFIG
        self.llm_provider = llm_provider or self.config.get("llm_provider", "openai")
        self.llm_model = llm_model or self.config["llm_model"]

        # API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self.config.get("openai_api_key", "")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it via --api-key or .env (OPENAI_API_KEY=...)."
            )

        logger.info(f"Initialized planner with openai/{self.llm_model}")

        # Initialize modules (lazy loading)
        self.vidmuse = None
        self.tts = None
        self.mixer = None

    def _init_modules(self):
        """Initialize audio generation modules"""
        if self.vidmuse is None:
            from src.modules.vidmuse import VidMuseGenerator
            from src.modules.tts import TTSEngine
            from src.utils.audio_mixer import AudioMixer

            self.vidmuse = VidMuseGenerator()
            self.tts = TTSEngine(engine_type="auto")
            self.mixer = AudioMixer()

            logger.info("Audio generation modules initialized")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM to generate response

        Args:
            prompt: Input prompt

        Returns:
            LLM response
        """
        return self._call_openai(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            from openai import OpenAI

            base_url = (self.config.get("openai_base_url") or "").strip() or None
            client = OpenAI(api_key=self.api_key, base_url=base_url)

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert audio producer. Generate structured audio production plans.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _generate_script(
        self, instruction: str, video_context: Optional[str] = None
    ) -> Dict:
        """
        Generate audio script from user instruction

        Args:
            instruction: User's creative instruction
            video_context: Optional video context

        Returns:
            Structured audio script
        """
        # Build prompt
        prompt = f"""Generate a detailed audio production script based on this instruction:

Instruction: {instruction}

{f'Video Context: {video_context}' if video_context else ''}

Provide a JSON script with the following structure:
{{
    "description": "Brief description of the audio",
    "duration": <total duration in seconds>,
    "elements": [
        {{
            "type": "music" or "speech" or "sfx",
            "content": "description or text",
            "start_time": <start time in seconds>,
            "duration": <duration in seconds>,
            "volume": <0.0-1.0>,
            "properties": {{
                // For music: "style", "tempo", "mood"
                // For speech: "voice", "emotion", "speed"
            }}
        }}
    ]
}}

Generate the script:"""

        logger.info("Generating audio script with LLM...")

        try:
            response = self._call_llm(prompt)

            # Extract JSON from response
            # Look for JSON block in markdown code fence
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            script = json.loads(json_str)

            logger.info(f"Generated script with {len(script['elements'])} elements")
            return script

        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Fallback to simple script
            return self._create_fallback_script(instruction)
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return self._create_fallback_script(instruction)

    def _create_fallback_script(self, instruction: str) -> Dict:
        """Create a simple fallback script if LLM fails"""
        return {
            "description": instruction,
            "duration": 30,
            "elements": [
                {
                    "type": "music",
                    "content": "background music matching the video",
                    "start_time": 0,
                    "duration": 30,
                    "volume": 0.6,
                    "properties": {"style": "cinematic", "mood": "neutral"},
                }
            ],
        }

    def _execute_script(
        self, script: Dict, video_path: Optional[Union[str, Path]] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Execute audio script by calling generation modules

        Args:
            script: Audio script dictionary
            video_path: Optional video path for music generation

        Returns:
            List of (audio_file_path, volume, element_type) tuples
        """
        from src.config import TEMP_DIR

        # Initialize modules
        self._init_modules()

        audio_clips = []

        for i, element in enumerate(script["elements"]):
            elem_type = element["type"]
            temp_path = TEMP_DIR / f"clip_{i}_{elem_type}.wav"

            logger.info(f"Generating element {i+1}/{len(script['elements'])}: {elem_type}")

            try:
                if elem_type == "music":
                    # Generate music with VidMuse
                    if video_path:
                        audio = self.vidmuse.generate(
                            video_path=video_path,
                            duration=element.get("duration", 30),
                        )
                        self.vidmuse.save(audio, temp_path)
                    else:
                        logger.warning("No video provided for music generation, skipping")
                        continue

                elif elem_type == "speech":
                    # Generate speech with TTS
                    props = element.get("properties", {})
                    self.tts.synthesize(
                        text=element["content"],
                        output_path=temp_path,
                        speaker=props.get("speaker"),
                        language=props.get("language"),
                        speed=props.get("speed", 1.0),
                    )

                else:
                    logger.warning(f"Unknown element type: {elem_type}, skipping")
                    continue

                # Add to clips list
                audio_clips.append((str(temp_path), element.get("volume", 1.0), elem_type))

            except Exception as e:
                logger.error(f"Failed to generate element {i}: {e}")
                continue

        return audio_clips

    def create(
        self,
        instruction: str,
        video_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        merge_video: bool = False,
        merged_video_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[str, Dict]:
        """
        Create audio from instruction

        Args:
            instruction: User's creative instruction
            video_path: Optional video path
            output_path: Output file path
            merge_video: If True, mux final audio back into the input video
            merged_video_path: Optional output path for merged mp4

        Returns:
            Tuple of (output_path, script)
        """
        from src.config import OUTPUT_DIR

        if output_path is None:
            output_path = OUTPUT_DIR / "wavjourney_output.wav"

        # Step 1: Generate script
        video_context = None
        if video_path:
            video_context = f"Video file: {Path(video_path).name}"

        # If video available, clamp durations to video length
        video_duration = None
        if video_path and VideoFileClip:
            try:
                with VideoFileClip(str(video_path)) as clip:
                    video_duration = clip.duration
            except Exception as e:  # pragma: no cover - runtime check
                logger.warning(f"Could not read video duration: {e}")

        script = self._generate_script(instruction, video_context)

        # Align script duration to video length if known
        if video_duration:
            script["duration"] = min(script.get("duration", video_duration), video_duration)
            for element in script.get("elements", []):
                if "duration" in element:
                    element["duration"] = min(element["duration"], video_duration)
                else:
                    element["duration"] = video_duration
                # Keep start_time within bounds
                if element.get("start_time", 0) >= video_duration:
                    element["start_time"] = max(0, video_duration - element["duration"])

        logger.info(f"Script: {json.dumps(script, indent=2)}")

        # Step 2: Execute script
        audio_clips = self._execute_script(script, video_path)

        if not audio_clips:
            raise RuntimeError("No audio clips generated")

        # Step 3: Mix audio
        logger.info("Mixing audio clips...")

        # Mixer tuning (aligned with CLI defaults)
        target_lufs = -18.0
        ducking_db = 8.0
        duck_attack = 0.02
        duck_release = 0.2

        music_tracks: List[Tuple[np.ndarray, float]] = []
        speech_tracks: List[Tuple[np.ndarray, float]] = []
        other_tracks: List[Tuple[np.ndarray, float]] = []

        for clip_path, volume, elem_type in audio_clips:
            audio, sr = self.mixer.load_audio(clip_path)
            audio = self.mixer.resample(audio, sr, self.mixer.target_sample_rate)
            audio = self.mixer.match_loudness(audio, self.mixer.target_sample_rate, target_lufs=target_lufs)

            if elem_type == "speech":
                speech_tracks.append((audio, volume))
            elif elem_type == "music":
                music_tracks.append((audio, volume))
            else:
                other_tracks.append((audio, volume))

        def _mix_group(group: List[Tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
            if not group:
                return None
            return self.mixer.mix_tracks(group, method="overlay")

        music_mix = _mix_group(music_tracks)
        speech_mix = _mix_group(speech_tracks)
        other_mix = _mix_group(other_tracks)

        if music_mix is not None and speech_mix is not None:
            music_mix = self.mixer.duck_music(
                music_mix,
                speech_mix,
                self.mixer.target_sample_rate,
                ducking_db=ducking_db,
                attack=duck_attack,
                release=duck_release,
            )

        final_tracks = []
        if music_mix is not None:
            final_tracks.append((music_mix, 1.0))
        if speech_mix is not None:
            final_tracks.append((speech_mix, 1.0))
        if other_mix is not None:
            final_tracks.append((other_mix, 1.0))

        mixed = self.mixer.mix_tracks(final_tracks, method="overlay")
        mixed = self.mixer.normalize(mixed, target_loudness=target_lufs)
        mixed = self.mixer.apply_fade(mixed, fade_in=0.5, fade_out=1.0)

        self.mixer.save(mixed, output_path)

        logger.info(f"Final audio saved to {output_path}")

        # Optional: mux audio back into video
        if merge_video:
            if not video_path:
                logger.warning("merge_video=True but no video_path provided; skipping mux.")
            elif not merge_video_audio:
                logger.error("merge_video requested but merge_video_audio helper unavailable.")
            else:
                merged_out = merged_video_path or Path(output_path).with_suffix(".mp4")
                try:
                    merge_video_audio(Path(video_path), Path(output_path), Path(merged_out))
                    logger.info(f"Merged video saved to {merged_out}")
                except Exception as e:
                    logger.error(f"Failed to merge video and audio: {e}")

        return str(output_path), script

    def clear_memory(self):
        """Clear GPU memory"""
        if self.vidmuse:
            self.vidmuse.clear_memory()
        if self.tts:
            self.tts.clear_memory()


# ============================================================================
# Standalone usage example
# ============================================================================

if __name__ == "__main__":
    # Example usage
    planner = AudioCreativePlanner(llm_model="gpt-4", api_key="your-key-here")

    # Create audio from instruction
    output_path, script = planner.create(
        instruction="Create a 30-second dramatic video intro with orchestral music and a deep male narrator",
        video_path="examples/intro.mp4",
        output_path="output/intro_audio.wav",
    )

    print(f"Generated audio: {output_path}")
    print(f"Script: {json.dumps(script, indent=2)}")
