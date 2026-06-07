import Foundation
import AudioCommon

extension NemotronStreamingASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Nemotron streaming transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        let text = transcribe(audio: audio, sampleRate: sampleRate, language: language)
        // The multilingual Nemotron is *prompted* to a language via the `language_mask`
        // slot, it does not *detect* one. So when the caller pins a language we echo it
        // back (the transcript is constrained to it); in `auto` mode (`language == nil`)
        // `transcribe()` does not surface which slot the model chose, so the language is
        // genuinely unknown → nil. Never hard-code a language here — the encoder is
        // multilingual, not English-only.
        return TranscriptionResult(text: text, language: text.isEmpty ? nil : language, confidence: 0)
    }
}
