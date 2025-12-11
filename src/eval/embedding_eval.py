import json
import logging
import shutil
import tempfile
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm

from src.utils import av_alignment

logger = logging.getLogger(__name__)

# ------------------------
# Audio embedding backends
# ------------------------

def _load_audio_backend(name: str, device: str, use_fp16: bool, cache_dir: Optional[Path] = None):
    name = name.lower()
    if name == "clap":
        return {"name": "clap", "model": None, "cache_dir": cache_dir}

    if name == "muq_mulan":
        from muq import MuQMuLan

        logger.info("Loading MuQ-MuLan (may download ~700MB, CC-BY-NC)...")
        model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=str(cache_dir) if cache_dir else None)
        model = model.to(device).eval()
        return {"name": "muq_mulan", "model": model, "cache_dir": cache_dir}

    if name in ("vggish", "vggsound"):
        import torchvggish
        from torchvggish import vggish_input

        logger.info("Loading torchvggish backend (AudioSet/VGGish weights)...")
        model = torchvggish.vggish(postprocess=True)
        model = model.to(device).eval()
        return {"name": name, "model": model, "vggish_input": vggish_input, "cache_dir": cache_dir}

    raise ValueError(f"Unknown audio backend: {name}")


def _extract_audio_embedding_backend(
    audio_path: Path,
    backend: dict,
    av_models,  # CLAP/CLIP bundle for CLAP path
    device: str,
    use_fp16: bool,
) -> torch.Tensor:
    name = backend["name"]
    if name == "clap":
        return av_alignment._extract_audio_embedding(audio_path, av_models)

    if name == "muq_mulan":
        import librosa

        mulan = backend["model"]
        wav, sr = librosa.load(audio_path, sr=24000)
        wavs = torch.tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = mulan(wavs=wavs)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb

    if name in ("vggish", "vggsound"):
        import soundfile as sf

        vggish_input = backend["vggish_input"]
        model = backend["model"]
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        examples = vggish_input.waveform_to_examples(wav, sr)
        x = torch.tensor(examples).unsqueeze(1).float().to(device)
        if use_fp16 and device.startswith("cuda"):
            x = x.half()
        with torch.no_grad():
            feats = model(x)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.mean(dim=0, keepdim=True)

    raise ValueError(f"Unhandled backend: {name}")


def embed_video_audio(
    video_path: Path,
    audio_path: Path,
    av_models,
    audio_backend: dict,
    device: str,
    use_fp16: bool,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    video_emb = av_alignment._extract_video_embedding(video_path, av_models, num_frames=num_frames).cpu().numpy()
    audio_emb = _extract_audio_embedding_backend(audio_path, audio_backend, av_models, device, use_fp16).cpu().numpy()
    return video_emb, audio_emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieval(video_embs: np.ndarray, audio_embs: np.ndarray, ks=(1, 5, 10), compute_map_ndcg=False):
    sims = video_embs @ audio_embs.T
    ranks = []
    recalls = {k: 0 for k in ks}
    ap_list = []
    ndcg_list = []
    for i in range(len(video_embs)):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0] + 1)
        ranks.append(rank)
        for k in ks:
            recalls[k] += int(rank <= k)
        if compute_map_ndcg:
            rel = np.zeros(len(audio_embs))
            rel[i] = 1
            try:
                from sklearn.metrics import average_precision_score, ndcg_score

                ap_list.append(average_precision_score(rel, sims[i]))
                ndcg_list.append(ndcg_score(rel.reshape(1, -1), sims[i].reshape(1, -1)))
            except Exception:
                pass
    recalls = {f"Recall@{k}": v / len(video_embs) for k, v in recalls.items()}
    med_rank = np.median(ranks)
    out = {"recalls": recalls, "med_rank": med_rank}
    if ap_list:
        out["mAP"] = float(np.mean(ap_list))
    if ndcg_list:
        out["nDCG"] = float(np.mean(ndcg_list))
    return out


def fid_like(emb_ref: np.ndarray, emb_gen: np.ndarray) -> float:
    mu1, mu2 = emb_ref.mean(0), emb_gen.mean(0)
    sigma1, sigma2 = np.cov(emb_ref, rowvar=False), np.cov(emb_gen, rowvar=False)
    diff = mu1 - mu2
    eps = 1e-6
    covmean = sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ (sigma2 + eps * np.eye(sigma2.shape[0])))
    covmean = np.real(covmean)
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def rhythm_motion_corr(video_path: Path, audio_path: Path) -> float:
    import cv2
    import librosa
    import soundfile as sf

    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.nan
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 3, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(mag.mean())
        prev_gray = gray
    cap.release()
    if len(mags) < 2:
        return np.nan
    motion = np.array(mags)
    motion = (motion - motion.mean()) / (motion.std() + 1e-8)

    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=wav, sr=sr)
    onset_env = (onset_env - onset_env.mean()) / (onset_env.std() + 1e-8)

    target_len = len(motion)
    onset_resamp = librosa.resample(onset_env, orig_sr=len(onset_env), target_sr=target_len)
    onset_resamp = onset_resamp[:target_len]
    if len(onset_resamp) < 2:
        return np.nan
    corr = np.corrcoef(motion, onset_resamp)[0, 1]
    return float(corr)


def load_pairs(manifest: Path) -> List[Tuple[Path, Path]]:
    import pandas as pd

    df = pd.read_csv(manifest)
    pairs = []
    for _, row in df.iterrows():
        v = Path(row["video_path"])
        a = Path(row["audio_path"])
        if v.exists() and a.exists():
            pairs.append((v, a))
        else:
            logger.warning("Missing pair: %s / %s", v, a)
    return pairs


def run_eval(
    manifest: Path,
    metrics: Set[str],
    reference_audio_dir: Optional[Path],
    device: str = "auto",
    compute_map_ndcg: bool = False,
    fad_backend: str = "auto",
    num_video_frames: int = 8,
    audio_backend_name: str = "clap",
    audio_cache_dir: Optional[Path] = None,
    save_embeddings: Optional[Path] = None,
    load_embeddings: Optional[Path] = None,
) -> Dict[str, float]:
    pairs = load_pairs(manifest)
    if not pairs:
        raise RuntimeError("No valid pairs in manifest")

    device, use_fp16 = av_alignment._select_device(device)
    av_models = av_alignment._load_models(device, use_fp16)
    audio_backend = _load_audio_backend(audio_backend_name, device, use_fp16, cache_dir=audio_cache_dir)

    if load_embeddings and load_embeddings.exists():
        data = np.load(load_embeddings, allow_pickle=True)
        video_embs = data["video_embs"]
        audio_embs = data["audio_embs"]
    else:
        video_embs, audio_embs = [], []
        for v, a in pairs:
            v_emb, a_emb = embed_video_audio(
                v,
                a,
                av_models,
                audio_backend,
                device,
                use_fp16,
                num_frames=num_video_frames,
            )
            video_embs.append(v_emb.squeeze())
            audio_embs.append(a_emb.squeeze())
        video_embs = np.stack(video_embs)
        audio_embs = np.stack(audio_embs)
        if save_embeddings:
            np.savez(save_embeddings, video_embs=video_embs, audio_embs=audio_embs)

    results: Dict[str, float] = {}

    if "sim" in metrics:
        sims = [cosine(video_embs[i], audio_embs[i]) for i in range(len(pairs))]
        results["sim/mean_cosine"] = float(np.mean(sims))

    if "retrieval" in metrics:
        r = retrieval(video_embs, audio_embs, ks=(1, 5, 10), compute_map_ndcg=compute_map_ndcg)
        results.update({f"retrieval/{k}": v for k, v in r["recalls"].items()})
        results["retrieval/MedianRank"] = float(r["med_rank"])
        if "mAP" in r:
            results["retrieval/mAP"] = float(r["mAP"])
        if "nDCG" in r:
            results["retrieval/nDCG"] = float(r["nDCG"])

    if "dist" in metrics and reference_audio_dir:
        ref_audio = sorted([p for p in reference_audio_dir.glob("*.wav")])
        if ref_audio:
            ref_embs = []
            for a in ref_audio:
                ref_embs.append(
                    _extract_audio_embedding_backend(a, audio_backend, av_models, device, use_fp16)
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            ref_embs = np.stack(ref_embs)
            if ref_embs.shape[0] > 1 and audio_embs.shape[0] > 1:
                results[f"dist/frechet_{audio_backend_name}"] = fid_like(ref_embs, audio_embs)
            else:
                results[f"dist/frechet_{audio_backend_name}"] = np.nan

            fad_score = np.nan
            if fad_backend in ("auto", "frechet_audio_distance"):
                try:
                    import frechet_audio_distance as fad

                    fad_model = fad.FrechetAudioDistance(
                        model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False
                    )
                    eval_files = [str(a) for a in [p[1] for p in pairs]]
                    ref_files = [str(p) for p in ref_audio]

                    sig = inspect.signature(fad_model.score)
                    if "ref_files" in sig.parameters:  # newer API
                        fad_score = fad_model.score(ref_files=ref_files, eval_files=eval_files)
                    else:  # API expects directories; stage eval files into a temp dir
                        with tempfile.TemporaryDirectory(prefix="fad_eval_") as tmp_eval_dir, tempfile.TemporaryDirectory(prefix="fad_ref_") as tmp_ref_dir:
                            tmp_eval = Path(tmp_eval_dir)
                            tmp_ref = Path(tmp_ref_dir)
                            for src in eval_files:
                                shutil.copy(src, tmp_eval / Path(src).name)
                            for src in ref_files:
                                shutil.copy(src, tmp_ref / Path(src).name)
                            fad_score = fad_model.score(
                                background_dir=str(tmp_ref),
                                eval_dir=str(tmp_eval),
                            )
                except Exception:
                    fad_score = np.nan
            results["dist/FAD"] = float(fad_score) if not np.isnan(fad_score) else np.nan
        else:
            results[f"dist/frechet_{audio_backend_name}"] = np.nan

    if "rhythm" in metrics:
        corrs = []
        for v, a in pairs:
            c = rhythm_motion_corr(v, a)
            if not np.isnan(c):
                corrs.append(c)
        results["rhythm/motion_onset_corr"] = float(np.mean(corrs)) if corrs else np.nan

    return results


__all__ = [
    "run_eval",
    "_load_audio_backend",
    "_extract_audio_embedding_backend",
    "embed_video_audio",
    "cosine",
    "retrieval",
    "fid_like",
    "rhythm_motion_corr",
]
