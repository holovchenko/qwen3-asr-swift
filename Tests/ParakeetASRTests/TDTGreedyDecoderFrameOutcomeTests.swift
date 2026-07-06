import XCTest
@testable import ParakeetASR

/// Unit tests for `TDTGreedyDecoder.frameOutcome`, the pure decision function
/// that drives multi-symbol-per-frame TDT decoding (root cause of dropped
/// spoken numbers — see `.superpowers/sdd/digit-drop-rootcause-report.md` in
/// the Hovor repo). Before the fix, the decode loop did
/// `t += max(duration, 1)` unconditionally, capping emission to exactly one
/// token per encoder frame. These tests pin the corrected frame-advancement
/// semantics: `duration == 0` re-enters the same frame (bounded by
/// `maxSymbolsPerStep`), `duration > 0` always advances immediately.
final class TDTGreedyDecoderFrameOutcomeTests: XCTestCase {

    // MARK: - duration == 0 → stay on frame (the actual bug fix)

    func testStaysOnFrameWhenDurationZeroAndBelowCap() {
        let outcome = TDTGreedyDecoder.frameOutcome(
            duration: 0, symbolsAtThisFrame: 0, maxSymbolsPerStep: 10
        )
        XCTAssertEqual(outcome, .stayOnFrame(symbolsAtThisFrame: 1),
            "duration==0 must re-enter the same frame to allow a second token — " +
            "this is what the pre-fix decoder could never do")
    }

    func testSymbolCounterIncrementsAcrossConsecutiveZeroDurationEmissions() {
        let first = TDTGreedyDecoder.frameOutcome(duration: 0, symbolsAtThisFrame: 0, maxSymbolsPerStep: 10)
        guard case .stayOnFrame(let afterFirst) = first else {
            return XCTFail("expected stayOnFrame, got \(first)")
        }
        XCTAssertEqual(afterFirst, 1)

        let second = TDTGreedyDecoder.frameOutcome(duration: 0, symbolsAtThisFrame: afterFirst, maxSymbolsPerStep: 10)
        guard case .stayOnFrame(let afterSecond) = second else {
            return XCTFail("expected stayOnFrame, got \(second)")
        }
        XCTAssertEqual(afterSecond, 2, "burst of duration==0 emissions must allow >1 token per frame")
    }

    // MARK: - duration > 0 → single-symbol behavior unchanged

    func testAdvancesImmediatelyWhenDurationPositive() {
        let outcome = TDTGreedyDecoder.frameOutcome(
            duration: 3, symbolsAtThisFrame: 0, maxSymbolsPerStep: 10
        )
        XCTAssertEqual(outcome, .advance(by: 3, symbolsAtThisFrame: 0),
            "duration>0 must behave exactly as before the fix: advance by duration, reset the per-frame counter")
    }

    func testAdvancesImmediatelyWhenDurationPositiveEvenMidBurst() {
        // Even if we've already emitted a few tokens at this frame, a positive
        // duration always ends the burst and advances — matches FluidAudio.
        let outcome = TDTGreedyDecoder.frameOutcome(
            duration: 2, symbolsAtThisFrame: 4, maxSymbolsPerStep: 10
        )
        XCTAssertEqual(outcome, .advance(by: 2, symbolsAtThisFrame: 0))
    }

    // MARK: - maxSymbolsPerStep cap → forced safety advance (termination guard)

    func testForcesSafetyAdvanceWhenHittingSymbolCap() {
        // 9 emissions already happened at this frame; the 10th would hit the cap.
        let outcome = TDTGreedyDecoder.frameOutcome(
            duration: 0, symbolsAtThisFrame: 9, maxSymbolsPerStep: 10
        )
        XCTAssertEqual(outcome, .advance(by: 1, symbolsAtThisFrame: 0),
            "hitting maxSymbolsPerStep must force t += 1 so the outer loop always terminates")
    }

    func testNeverExceedsMaxSymbolsPerStepAcrossASimulatedBurst() {
        // Simulate the model always predicting duration==0 (worst case for an
        // infinite loop) and drive frameOutcome in a loop exactly like the
        // decode() method does. Must terminate (advance) within
        // maxSymbolsPerStep iterations — this is the anti-infinite-loop guarantee.
        let maxSymbolsPerStep = 10
        var symbolsAtThisFrame = 0
        var iterations = 0
        var advanced = false

        while iterations < 1000 {  // hard safety bound for the test itself
            iterations += 1
            let outcome = TDTGreedyDecoder.frameOutcome(
                duration: 0, symbolsAtThisFrame: symbolsAtThisFrame, maxSymbolsPerStep: maxSymbolsPerStep
            )
            switch outcome {
            case .stayOnFrame(let next):
                symbolsAtThisFrame = next
            case .advance(let by, let next):
                XCTAssertEqual(by, 1, "forced safety advance must be exactly 1 frame")
                symbolsAtThisFrame = next
                advanced = true
            }
            if advanced { break }
        }

        XCTAssertTrue(advanced, "decode loop must eventually force an advance, never loop forever")
        XCTAssertLessThanOrEqual(iterations, maxSymbolsPerStep,
            "must not take more than maxSymbolsPerStep joint() calls to force an advance")
    }

    // MARK: - maxSymbolsPerStep boundary is configurable

    func testRespectsCustomMaxSymbolsPerStep() {
        // With a cap of 1, duration==0 must force-advance immediately — this
        // degenerates to the OLD one-token-per-frame behavior, proving the cap
        // is load-bearing and not hardcoded.
        let outcome = TDTGreedyDecoder.frameOutcome(
            duration: 0, symbolsAtThisFrame: 0, maxSymbolsPerStep: 1
        )
        XCTAssertEqual(outcome, .advance(by: 1, symbolsAtThisFrame: 0))
    }

    // MARK: - Config plumbing

    func testConfigDefaultMaxSymbolsPerStepIsTen() {
        XCTAssertEqual(ParakeetConfig.default.maxSymbolsPerStep, 10)
    }

    func testConfigDecodesWhenMaxSymbolsPerStepKeyIsAbsent() throws {
        // Real-world config.json files (aufklarer's export) predate this field
        // entirely. Decoding must not throw and must default to 10.
        let json = """
        {
            "numMelBins": 128,
            "sampleRate": 16000,
            "nFFT": 512,
            "hopLength": 160,
            "winLength": 400,
            "preEmphasis": 0.97,
            "encoderHidden": 1024,
            "encoderLayers": 24,
            "subsamplingFactor": 8,
            "decoderHidden": 640,
            "decoderLayers": 2,
            "vocabSize": 8192,
            "blankTokenId": 8192,
            "numDurationBins": 5,
            "durationBins": [0, 1, 2, 3, 4]
        }
        """
        let config = try JSONDecoder().decode(ParakeetConfig.self, from: Data(json.utf8))
        XCTAssertEqual(config.maxSymbolsPerStep, 10)
        XCTAssertEqual(config.vocabSize, 8192)
    }

    func testConfigRoundTripsCustomMaxSymbolsPerStep() throws {
        let original = ParakeetConfig(maxSymbolsPerStep: 7)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ParakeetConfig.self, from: data)
        XCTAssertEqual(decoded.maxSymbolsPerStep, 7)
    }
}
