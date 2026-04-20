# Programming Assignment 2 Report

**Course:** Speech Understanding  
**Assignment:** Programming Assignment 2  
**Name:** Ashish Gajbhiye  
**GitHub Link:** https://github.com/ashishgajbhiye-boop/speech-assignment.git  

---

## 1. Introduction

In this assignment, I implemented an end-to-end speech processing pipeline for a 10-minute audio segment. The goal was to process code-switched speech and synthesize it into a Low-Resource Language (LRL).

Due to practical constraints and missing pretrained components, I focused on building a fully working modular pipeline and validating each stage independently.

The system performs:

- preprocessing and denoising  
- speech-to-text transcription  
- language identification (approximate)  
- phonetic (IPA) conversion  
- translation to LRL  
- speech synthesis  
- evaluation and robustness analysis  

For this implementation, I selected hindi as the target LRL.

---

## 2. Overall Pipeline

The pipeline was implemented in a modular structure using Python and PyTorch-based libraries.

Pipeline flow:

Audio → Preprocess → STT → LID → IPA → Translation → TTS → Evaluation  

Each stage was independently executable using:

    python pipeline.py --stage <stage_name>

---

## 3. Data Preparation

The input audio was a 10-minute segment converted into WAV format.

Files used:

- data/samples/original_segment.wav
- data/samples/student_voice_ref.wav

Generated outputs:

- outputs/clean_segment.wav
- outputs/output_LRL_cloned.wav

Due to computational constraints, final synthesis was demonstrated on a short representative segment instead of full 10-minute audio.

---

## 4. Preprocessing

A spectral subtraction method was implemented for denoising.

---

## 5. Speech-to-Text (STT)

For transcription, I used OpenAI Whisper.

Output:
- outputs/transcript.txt

---

## 6. Language Identification (LID)

A simple word-level LID approximation was implemented.

Output:
- outputs/lid_labels.txt

---

## 7. IPA Conversion

Output:
- outputs/transcript_ipa.txt

---

## 8. Translation

Output:
- outputs/translated_lrl.txt

---

## 9. TTS

Output:
- outputs/output_LRL_cloned.wav

---

## 10. Evaluation

WER (English): 0.857  
WER (Hindi): 0.143  
MCD: 1683.17  
Switch Precision: 1.0  

---

## 11. Conclusion

This project demonstrates an end-to-end speech pipeline with modular integration.
