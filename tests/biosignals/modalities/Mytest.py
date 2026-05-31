from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21


speech = Speech("/Users/homi/Documents/Master Thesis/LTBio/resources/ADReSSo21_WAV_tests/cn/adrso002.wav", source=ADReSSo21)

# speech.plot()
print(speech)

# Get only patient speech periods
patient_timeline = ADReSSo21.patient_speaking(speech)

# patient_timeline.domain_timeline.plot()
# patient_timeline.plot()

# Slice the biosignal to patient speech only
patient_speech = speech[patient_timeline]

# Now extract features only from patient's voice
print(f"Silence Percentage: {patient_speech.silence_percentage()}")
print(f"Acceptable Quality: {patient_speech.acceptable_quality()}")

# 1. Silence on the FULL recording (includes pauses between turns)
print(f"Silence on the FULL recording: {speech.silence_percentage()}")

# 2. How much of the total recording is patient speaking vs silent/interviewer
total_seconds = speech.duration.total_seconds()
patient_seconds = patient_timeline.duration.total_seconds()
print(f"Patient speaking: {patient_seconds:.1f}s out of {total_seconds:.1f}s ({100*patient_seconds/total_seconds:.1f}%)")

# 3. Silence within patient speech with a stricter threshold
print(patient_speech.silence_percentage(threshold=0.05))

for e in speech.events:
    print(e.duration)