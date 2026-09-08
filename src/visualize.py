from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from . import config
from .audio_io import load_audio_file
from .preprocess import to_mono, resample_to, trim_silence

sns.set_theme(style="whitegrid")

BASE_DIR = config.OUTPUTS_DIR / "signal_stages"


def dur(y, sr):
    return len(y) / sr if sr else 0.0


def save_fig(fig, ax, title, xlabel, ylabel, path, legend=True):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_stages(out_dir, prefix, y_raw, sr_raw, label):
    channels = 1 if y_raw.ndim == 1 else y_raw.shape[1]
    y_mono = to_mono(y_raw)
    y_res, sr_res = resample_to(y_mono, sr_raw)
    y_trim = trim_silence(y_res)
    removed = dur(y_res, sr_res) - dur(y_trim, sr_res)

    make_channel_plot(out_dir, prefix, y_raw, sr_raw, channels, label)
    make_resample_zoom(out_dir, prefix, y_mono, sr_raw, y_res, sr_res, label)
    make_trim_plot(out_dir, prefix, y_res, sr_res, removed, label)
    make_spectrogram(out_dir, prefix, y_trim, sr_res, label)

    print(f"  {label}: {channels}ch/{sr_raw}Hz/{len(y_raw)} -> "
          f"1ch/{sr_res}Hz/{len(y_trim)} ({removed:.2f}s trimmed)")


def make_channel_plot(out_dir, prefix, y_raw, sr_raw, channels, label):
    fig, ax = plt.subplots(figsize=(11, 4))
    if channels >= 2:
        left, right = y_raw[:, 0], y_raw[:, 1]
        t = np.linspace(0, dur(left, sr_raw), num=len(left))
        ax.plot(t, left + 0.5, linewidth=0.5, color="lightblue", label="left channel (+0.5)")
        ax.plot(t, right - 0.5, linewidth=0.5, color="orange", label="right channel (-0.5)")
        ax.plot(t, y_raw.mean(axis=1), linewidth=0.5, color="black", label="mono = average")
        title = "Step A - stereo to mono: two channels averaged into one"
    else:
        y = to_mono(y_raw)
        t = np.linspace(0, dur(y, sr_raw), num=len(y))
        ax.plot(t, y, linewidth=0.5, color="lightblue", label="single channel (already mono)")
        title = "Step A - source already mono: nothing to average"
    save_fig(fig, ax, f"{title}\n{label}", "time (s)", "amplitude (offset for clarity)",
              out_dir / f"{prefix}1_stereo_to_mono.png")


def make_resample_zoom(out_dir, prefix, y_mono, sr_raw, y_res, sr_res, label):
    win = 0.012
    seg_raw = y_mono[len(y_mono) // 2:len(y_mono) // 2 + max(int(win * sr_raw), 1)]
    seg_res = y_res[len(y_res) // 2:len(y_res) // 2 + max(int(win * sr_res), 1)]
    t_raw = np.arange(len(seg_raw)) / sr_raw * 1000.0
    t_res = np.arange(len(seg_res)) / sr_res * 1000.0

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_raw, seg_raw, "-o", markersize=3, linewidth=0.7, color="lightblue",
            label=f"before: {sr_raw} Hz ({len(seg_raw)} samples)")
    ax.plot(t_res, seg_res, "-s", markersize=5, linewidth=0.9, color="lightgreen",
            label=f"after: {sr_res} Hz ({len(seg_res)} samples)")
    save_fig(fig, ax, f"Step B - resampling: fewer sample points per millisecond\n{label}",
              "time (ms) - zoomed to a 12 ms window", "amplitude",
              out_dir / f"{prefix}2_resample_zoom.png")


def make_trim_plot(out_dir, prefix, y_res, sr_res, removed, label):
    _, index = librosa.effects.trim(y_res, top_db=config.TRIM_TOP_DB)
    start_s, end_s = index[0] / sr_res, index[1] / sr_res

    fig, ax = plt.subplots(figsize=(11, 4))
    t = np.linspace(0, dur(y_res, sr_res), num=len(y_res))
    ax.plot(t, y_res, linewidth=0.6, color="grey", label="full signal (resampled)")
    ax.axvspan(start_s, end_s, color="lightgreen", alpha=0.15, label="kept (speech)")
    ax.axvspan(0, start_s, color="red", alpha=0.12, label="removed (silence)")
    ax.axvspan(end_s, dur(y_res, sr_res), color="red", alpha=0.12)
    ax.set_ylim(-1.05, 1.05)
    save_fig(fig, ax, f"Step C - silence trimming: removed {removed:.2f} s from the ends\n{label}",
              "time (s)", "amplitude", out_dir / f"{prefix}3_silence_trim.png")


def make_spectrogram(out_dir, prefix, y, sr, label):
    if y.size == 0:
        return
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(11, 4))
    img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.grid(False)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    save_fig(fig, ax, f"Step D - cleaned signal spectrogram (frequencies over time)\n{label}",
              "", "", out_dir / f"{prefix}4_spectrogram.png", legend=False)


def make_stage_plots(audio_path, out_dir=None):
    y_raw, sr_raw = load_audio_file(audio_path)
    out_dir = Path(out_dir or (BASE_DIR / Path(audio_path).stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    render_stages(out_dir, "", y_raw, int(sr_raw), str(audio_path))
    print(f"Saved 4 figures -> {out_dir}")
    return out_dir


def make_dataset_plots(n=5, out_dir=None):
    from .data import iter_clips

    out_dir = Path(out_dir or (BASE_DIR / "dataset"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for row_id, y, sr in iter_clips(limit=n):
        render_stages(out_dir, f"clip{row_id}_", y, int(sr), f"dataset clip id={row_id}")
    print(f"Saved figures for {n} dataset clips -> {out_dir}")
    return out_dir


def main():
    make_dataset_plots(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
