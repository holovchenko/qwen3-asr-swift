import Foundation

/// Configuration for Nemotron Speech Streaming 0.6B (English).
///
/// Native punctuation + capitalization are emitted as regular BPE tokens —
/// there is no separate EOU/EOB head. End of stream is signaled by the caller
/// via `finalize()` on the streaming session.
public struct NemotronStreamingConfig: Codable, Sendable {
    public let numMelBins: Int
    public let sampleRate: Int
    public let nFFT: Int
    public let hopLength: Int
    public let winLength: Int
    public let preEmphasis: Float
    public let encoderHidden: Int
    public let encoderLayers: Int
    public let subsamplingFactor: Int
    public let attentionContext: Int
    public let convCacheSize: Int
    public let decoderHidden: Int
    public let decoderLayers: Int
    public let vocabSize: Int
    public let blankTokenId: Int
    public let streaming: StreamingConfig

    /// Maps the Swift property `attentionContext` onto the exporter's NeMo-style
    /// key `attentionLeftContext` (cache-aware Conformer left-context size,
    /// `att_context_size[0]`). The CoreML encoder is exported with a fixed
    /// `cache_last_channel` left-context dimension; this value must match it or
    /// `predict()` fails with a shape mismatch at transcription time. Declaring
    /// `CodingKeys` does NOT suppress the synthesized memberwise initializer, so
    /// `.default` keeps working.
    private enum CodingKeys: String, CodingKey {
        case numMelBins
        case sampleRate
        case nFFT
        case hopLength
        case winLength
        case preEmphasis
        case encoderHidden
        case encoderLayers
        case subsamplingFactor
        case attentionContext = "attentionLeftContext"
        case convCacheSize
        case decoderHidden
        case decoderLayers
        case vocabSize
        case blankTokenId
        case streaming
    }

    public struct StreamingConfig: Codable, Sendable {
        public let chunkMs: Int
        public let chunkSize: Int
        public let rightContext: Int
        public let melFrames: Int
        public let preCacheSize: Int
        public let outputFrames: Int
    }

    /// Default config matching the 160ms bundle at
    /// aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8.
    public static let `default` = NemotronStreamingConfig(
        numMelBins: 128,
        sampleRate: 16000,
        nFFT: 512,
        hopLength: 160,
        winLength: 400,
        preEmphasis: 0.97,
        encoderHidden: 1024,
        encoderLayers: 24,
        subsamplingFactor: 8,
        attentionContext: 70,
        convCacheSize: 8,
        decoderHidden: 640,
        decoderLayers: 2,
        vocabSize: 1024,
        blankTokenId: 1024,
        streaming: StreamingConfig(
            chunkMs: 160,
            chunkSize: 2,
            rightContext: 1,
            melFrames: 17,
            preCacheSize: 16,
            outputFrames: 2
        )
    )
}
