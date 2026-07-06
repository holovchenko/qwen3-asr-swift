import Foundation

/// Configuration for Parakeet TDT ASR model.
public struct ParakeetConfig: Codable, Sendable {
    /// Number of mel-spectrogram frequency bins.
    public let numMelBins: Int
    /// Expected input audio sample rate in Hz.
    public let sampleRate: Int
    /// FFT window size.
    public let nFFT: Int
    /// Hop length between successive STFT frames.
    public let hopLength: Int
    /// Window length for STFT.
    public let winLength: Int
    /// Pre-emphasis coefficient applied to the raw waveform.
    public let preEmphasis: Float
    /// Hidden dimension of the FastConformer encoder.
    public let encoderHidden: Int
    /// Number of FastConformer encoder layers.
    public let encoderLayers: Int
    /// Subsampling factor (encoder output frames = input frames / factor).
    public let subsamplingFactor: Int
    /// Hidden dimension of the LSTM prediction network.
    public let decoderHidden: Int
    /// Number of LSTM layers in the prediction network.
    public let decoderLayers: Int
    /// Vocabulary size (excluding blank).
    public let vocabSize: Int
    /// Token ID used for blank in the TDT decoder.
    public let blankTokenId: Int
    /// Number of duration bins for TDT.
    public let numDurationBins: Int
    /// Duration values corresponding to each bin index.
    public let durationBins: [Int]
    /// Maximum number of tokens the TDT decoder may emit from a single encoder
    /// frame (guards against infinite loops when the duration head keeps
    /// predicting `duration == 0`, i.e. "stay on this frame"). Mirrors
    /// FluidAudio's `maxSymbolsPerStep` default.
    public let maxSymbolsPerStep: Int

    public init(
        numMelBins: Int = 128,
        sampleRate: Int = 16000,
        nFFT: Int = 512,
        hopLength: Int = 160,
        winLength: Int = 400,
        preEmphasis: Float = 0.97,
        encoderHidden: Int = 1024,
        encoderLayers: Int = 24,
        subsamplingFactor: Int = 8,
        decoderHidden: Int = 640,
        decoderLayers: Int = 2,
        vocabSize: Int = 8192,
        blankTokenId: Int = 8192,
        numDurationBins: Int = 5,
        durationBins: [Int] = [0, 1, 2, 3, 4],
        maxSymbolsPerStep: Int = 10
    ) {
        self.numMelBins = numMelBins
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.preEmphasis = preEmphasis
        self.encoderHidden = encoderHidden
        self.encoderLayers = encoderLayers
        self.subsamplingFactor = subsamplingFactor
        self.decoderHidden = decoderHidden
        self.decoderLayers = decoderLayers
        self.vocabSize = vocabSize
        self.blankTokenId = blankTokenId
        self.numDurationBins = numDurationBins
        self.durationBins = durationBins
        self.maxSymbolsPerStep = maxSymbolsPerStep
    }

    /// Default configuration matching Parakeet-TDT 0.6B v3.
    public static let `default` = ParakeetConfig()

    // Custom Codable conformance: `maxSymbolsPerStep` is decoded with a default
    // of 10 when absent so existing `config.json` files (which predate this
    // field) keep decoding without modification.
    private enum CodingKeys: String, CodingKey {
        case numMelBins, sampleRate, nFFT, hopLength, winLength, preEmphasis
        case encoderHidden, encoderLayers, subsamplingFactor
        case decoderHidden, decoderLayers, vocabSize, blankTokenId
        case numDurationBins, durationBins, maxSymbolsPerStep
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try c.decode(Int.self, forKey: .numMelBins)
        sampleRate = try c.decode(Int.self, forKey: .sampleRate)
        nFFT = try c.decode(Int.self, forKey: .nFFT)
        hopLength = try c.decode(Int.self, forKey: .hopLength)
        winLength = try c.decode(Int.self, forKey: .winLength)
        preEmphasis = try c.decode(Float.self, forKey: .preEmphasis)
        encoderHidden = try c.decode(Int.self, forKey: .encoderHidden)
        encoderLayers = try c.decode(Int.self, forKey: .encoderLayers)
        subsamplingFactor = try c.decode(Int.self, forKey: .subsamplingFactor)
        decoderHidden = try c.decode(Int.self, forKey: .decoderHidden)
        decoderLayers = try c.decode(Int.self, forKey: .decoderLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        blankTokenId = try c.decode(Int.self, forKey: .blankTokenId)
        numDurationBins = try c.decode(Int.self, forKey: .numDurationBins)
        durationBins = try c.decode([Int].self, forKey: .durationBins)
        maxSymbolsPerStep = try c.decodeIfPresent(Int.self, forKey: .maxSymbolsPerStep) ?? 10
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(numMelBins, forKey: .numMelBins)
        try c.encode(sampleRate, forKey: .sampleRate)
        try c.encode(nFFT, forKey: .nFFT)
        try c.encode(hopLength, forKey: .hopLength)
        try c.encode(winLength, forKey: .winLength)
        try c.encode(preEmphasis, forKey: .preEmphasis)
        try c.encode(encoderHidden, forKey: .encoderHidden)
        try c.encode(encoderLayers, forKey: .encoderLayers)
        try c.encode(subsamplingFactor, forKey: .subsamplingFactor)
        try c.encode(decoderHidden, forKey: .decoderHidden)
        try c.encode(decoderLayers, forKey: .decoderLayers)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(blankTokenId, forKey: .blankTokenId)
        try c.encode(numDurationBins, forKey: .numDurationBins)
        try c.encode(durationBins, forKey: .durationBins)
        try c.encode(maxSymbolsPerStep, forKey: .maxSymbolsPerStep)
    }
}
