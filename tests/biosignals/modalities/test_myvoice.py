import sys
sys.path.insert(0, 'src')

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21

speech = Speech("resources/ADReSSo21_WAV_tests/cn/adrso002.wav", source=ADReSSo21)

# ── Basic info ────────────────────────────────────────────────────────────────
print(speech)
print("Sampling frequency :", speech.sampling_frequency, "Hz")
print("Duration           :", speech.duration)

# ── Silence ───────────────────────────────────────────────────────────────────
silence = speech.silence_percentage()
print("Silence            : {:.1f}%".format(silence['audio'] * 100))

# ── Quality ───────────────────────────────────────────────────────────────────
quality_timeline = speech.acceptable_quality()
print("Good quality periods:", quality_timeline)

# ── Convert to numpy array ────────────────────────────────────────────────────
arr = speech.to_array()
print("As numpy array shape:", arr.shape)

# ── Convert to DataFrame ──────────────────────────────────────────────────────
df = speech.to_dataframe()
print(df.head())

# # ── Plots ─────────────────────────────────────────────────────────────────────
# speech.plot(show=True)
# speech.plot_spectrum(show=True)
