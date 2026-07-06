import Accelerate
import CoreML
import Foundation

/// Reusable MLFeatureProvider that avoids dictionary allocation on every CoreML prediction.
/// Backing MLMultiArray references can be updated via `update(_:_:)` when the underlying
/// array changes (e.g. new decoder hidden state after a non-blank token).
private class ReusableFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private var values: [String: MLFeatureValue]

    init(_ dict: [String: MLMultiArray]) {
        self.featureNames = Set(dict.keys)
        self.values = dict.mapValues { MLFeatureValue(multiArray: $0) }
    }

    func featureValue(for name: String) -> MLFeatureValue? { values[name] }

    func update(_ name: String, _ array: MLMultiArray) {
        values[name] = MLFeatureValue(multiArray: array)
    }
}

/// Greedy decoder for Token-and-Duration Transducer (TDT) models.
///
/// TDT extends standard transducers with a duration prediction head.
/// When a non-blank token is emitted, the duration head predicts how many
/// encoder frames to advance (from `durationBins`), enabling variable-rate
/// alignment between audio and text.
///
/// Standard greedy decoding: starts with blank token to initialize the LSTM,
/// then runs the encoder-joint-decoder loop. The v3 multilingual model handles
/// all 25 European languages natively (EncDecRNNTBPEModel, no prompt tokens).
struct TDTGreedyDecoder {
    let config: ParakeetConfig
    let decoder: MLModel
    let joint: MLModel

    /// Decode encoded audio representations into token IDs with per-token log-probabilities.
    ///
    /// - Parameters:
    ///   - encoded: Encoder output as MLMultiArray, shape `[1, T, encoderHidden]`
    ///   - encodedLength: Number of valid encoder frames
    /// - Returns: Tuple of (token IDs, per-token log-probs, overall confidence 0.0–1.0)
    func decode(encoded: MLMultiArray, encodedLength: Int) throws -> (tokens: [Int], tokenLogProbs: [Float], confidence: Float) {
        var tokens = [Int]()
        var tokenLogProbs = [Float]()

        // Initialize LSTM state
        let hShape = [config.decoderLayers, 1, config.decoderHidden] as [NSNumber]
        let h = try MLMultiArray(shape: hShape, dataType: .float16)
        let c = try MLMultiArray(shape: hShape, dataType: .float16)
        zeroFill(h)
        zeroFill(c)

        // Pre-allocate token array — reused every decoder step
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)

        // Pre-allocate Float buffer for vDSP argmax on token logits
        let argmaxBuf = UnsafeMutablePointer<Float>.allocate(capacity: config.vocabSize + 1)
        defer { argmaxBuf.deallocate() }

        // Initialize decoder with blank token (standard TDT decoding)
        tokenPtr.pointee = Int32(config.blankTokenId)
        let decoderProvider = ReusableFeatureProvider([
            "token": tokenArray, "h": h, "c": c,
        ])
        let initOut = try decoder.prediction(from: decoderProvider)
        var hState = initOut.featureValue(for: "h_out")!.multiArrayValue!
        var cState = initOut.featureValue(for: "c_out")!.multiArrayValue!
        var decoderOutput = initOut.featureValue(for: "decoder_output")!.multiArrayValue!

        // Encoder slice buffer: [1, 1, encoderHidden]
        let encSlice = try MLMultiArray(shape: [1, 1, config.encoderHidden as NSNumber], dataType: .float16)

        // Pre-allocate reusable feature providers for the decode loop
        let jointProvider = ReusableFeatureProvider([
            "encoder_output": encSlice, "decoder_output": decoderOutput,
        ])
        decoderProvider.update("h", hState)
        decoderProvider.update("c", cState)

        // Special tokens (0..273) are filtered from output — they include
        // language tags, speaker tags, control tokens from SentencePiece training.
        // Text tokens start at index 274.
        let firstTextTokenId = 274

        // TDT decode loop. A non-blank emission whose duration head predicts
        // `duration == 0` means "another token likely follows from this SAME
        // encoder frame" — the joint network must be re-run on frame `t` again
        // (multi-symbol-per-frame emission) instead of unconditionally
        // advancing. Bounded by `maxSymbolsPerStep` so the loop always
        // terminates. Mirrors FluidAudio's TdtDecoderV3/RnntDecoder pattern.
        var t = 0
        var symbolsAtThisFrame = 0
        while t < encodedLength {
            // Extract encoder frame at position t (mutates encSlice data in-place)
            copyEncoderFrame(from: encoded, at: t, to: encSlice)

            // Joint network: (encoder_slice, decoder_output) → (token_logits, duration_logits)
            let jointOut = try joint.prediction(from: jointProvider)

            let tokenLogits = jointOut.featureValue(for: "token_logits")!.multiArrayValue!
            let durationLogits = jointOut.featureValue(for: "duration_logits")!.multiArrayValue!

            let tokenId = argmax(tokenLogits, count: config.vocabSize + 1, floatBuf: argmaxBuf)

            if tokenId == config.blankTokenId {
                t += 1
                symbolsAtThisFrame = 0
                continue
            }

            if tokenId >= firstTextTokenId {
                tokens.append(tokenId)
                // Compute log-softmax: log_prob = logit[id] - log(sum(exp(logits)))
                let logProb = logSoftmax(tokenLogits, tokenId: tokenId, count: config.vocabSize + 1, floatBuf: argmaxBuf)
                tokenLogProbs.append(logProb)
            }

            let durationIdx = argmax(durationLogits, count: config.numDurationBins, floatBuf: nil)
            let duration = config.durationBins[durationIdx]

            // Update decoder state with the emitted token — required before the
            // next joint() call regardless of whether we stay on this frame or
            // advance past it.
            tokenPtr.pointee = Int32(tokenId)
            decoderProvider.update("h", hState)
            decoderProvider.update("c", cState)
            let decOut = try decoder.prediction(from: decoderProvider)
            decoderOutput = decOut.featureValue(for: "decoder_output")!.multiArrayValue!
            hState = decOut.featureValue(for: "h_out")!.multiArrayValue!
            cState = decOut.featureValue(for: "c_out")!.multiArrayValue!

            // Update joint provider with new decoder output
            jointProvider.update("decoder_output", decoderOutput)

            let outcome = Self.frameOutcome(
                duration: duration,
                symbolsAtThisFrame: symbolsAtThisFrame,
                maxSymbolsPerStep: config.maxSymbolsPerStep
            )
            switch outcome {
            case .stayOnFrame(let next):
                symbolsAtThisFrame = next
            case .advance(let by, let next):
                t += by
                symbolsAtThisFrame = next
            }
        }

        // Overall confidence: exp(mean log-prob) maps to 0–1
        let confidence: Float
        if !tokenLogProbs.isEmpty {
            let meanLogProb = tokenLogProbs.reduce(0, +) / Float(tokenLogProbs.count)
            confidence = min(1.0, exp(meanLogProb))
        } else {
            confidence = 0.0
        }
        return (tokens, tokenLogProbs, confidence)
    }

    // MARK: - Frame Advancement (multi-symbol-per-frame TDT decoding)

    /// Outcome of processing one non-blank emission at the current encoder frame.
    enum FrameOutcome: Equatable {
        /// Duration head predicted 0 — re-run the joint on the SAME frame to
        /// emit another token (`symbolsAtThisFrame` is the updated count).
        case stayOnFrame(symbolsAtThisFrame: Int)
        /// Advance the encoder frame index by `by` (either the predicted
        /// duration, or a forced safety advance of 1 when the per-frame
        /// symbol cap is hit); `symbolsAtThisFrame` resets to the given value.
        case advance(by: Int, symbolsAtThisFrame: Int)
    }

    /// Decide how the outer decode loop should advance after a non-blank
    /// emission, given the model's predicted `duration` for that emission.
    ///
    /// Mirrors FluidAudio's `TdtDecoderV3`/`RnntDecoder` gating: `duration == 0`
    /// means "another token likely follows from this same frame", so the
    /// decoder stays on frame `t` (bounded by `maxSymbolsPerStep`, after which
    /// it is forced to advance by 1 so the outer loop always terminates).
    /// `duration > 0` always advances immediately, resetting the per-frame count.
    ///
    /// Blank tokens are handled directly in the decode loop (`t += 1`) and never
    /// reach this function.
    static func frameOutcome(
        duration: Int,
        symbolsAtThisFrame: Int,
        maxSymbolsPerStep: Int
    ) -> FrameOutcome {
        guard duration == 0 else {
            return .advance(by: duration, symbolsAtThisFrame: 0)
        }
        let next = symbolsAtThisFrame + 1
        if next >= maxSymbolsPerStep {
            return .advance(by: 1, symbolsAtThisFrame: 0)
        }
        return .stayOnFrame(symbolsAtThisFrame: next)
    }

    // MARK: - Array Operations

    /// Compute log-softmax for a specific token: log_prob = logit[id] - log(sum(exp(logits)))
    /// Uses vDSP for efficient log-sum-exp over the vocabulary.
    private func logSoftmax(_ array: MLMultiArray, tokenId: Int, count: Int, floatBuf: UnsafeMutablePointer<Float>) -> Float {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count { floatBuf[i] = Float(ptr[i]) }

        // log-sum-exp: find max, subtract, exp, sum, log, add max back
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf, 1, &maxVal, &maxIdx, vDSP_Length(count))

        // Subtract max for numerical stability
        var negMax = -maxVal
        vDSP_vsadd(floatBuf, 1, &negMax, floatBuf, 1, vDSP_Length(count))

        // exp in-place
        var n = Int32(count)
        vvexpf(floatBuf, floatBuf, &n)

        // sum
        var sumExp: Float = 0
        vDSP_sve(floatBuf, 1, &sumExp, vDSP_Length(count))

        let logSumExp = log(sumExp) + maxVal
        let logit = Float(ptr[tokenId])
        return logit - logSumExp
    }

    /// Copy encoder frame at time `t` into the slice buffer using memcpy.
    private func copyEncoderFrame(from encoded: MLMultiArray, at t: Int, to slice: MLMultiArray) {
        let hidden = config.encoderHidden
        let src = encoded.dataPointer.advanced(by: t * hidden * MemoryLayout<Float16>.stride)
        memcpy(slice.dataPointer, src, hidden * MemoryLayout<Float16>.stride)
    }

    /// Find the index of the maximum value in the first `count` elements.
    /// Uses vDSP for large arrays (token logits); scalar for small arrays (duration logits).
    private func argmax(_ array: MLMultiArray, count: Int, floatBuf: UnsafeMutablePointer<Float>?) -> Int {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)

        // Small arrays or no buffer: scalar path
        if count <= 16 || floatBuf == nil {
            var maxIdx = 0
            var maxVal = ptr[0]
            for i in 1..<count {
                if ptr[i] > maxVal {
                    maxVal = ptr[i]
                    maxIdx = i
                }
            }
            return maxIdx
        }

        // Large arrays: convert Float16→Float, then vDSP_maxvi
        for i in 0..<count { floatBuf![i] = Float(ptr[i]) }
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf!, 1, &maxVal, &maxIdx, vDSP_Length(count))
        return Int(maxIdx)
    }

    /// Zero-fill an MLMultiArray using memset.
    private func zeroFill(_ array: MLMultiArray) {
        memset(array.dataPointer, 0, array.count * MemoryLayout<Float16>.stride)
    }
}
