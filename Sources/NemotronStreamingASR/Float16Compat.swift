import Foundation

#if !arch(arm64)
// `Float16` is unavailable on x86_64 macOS. To let NemotronStreamingASR compile
// in the Intel slice of a universal Developer ID build, these helpers read CoreML
// half-precision (IEEE-754 binary16) buffers as raw `UInt16` and convert to `Float`
// in software. The conversion is exact for normal values, zero, and inf/NaN, and
// correct (within representable range) for subnormals — so the Intel build is not
// merely compilable but numerically faithful.
@inline(__always)
func fp16BitsToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    var exp = Int32((h >> 10) & 0x1F)
    var mant = UInt32(h & 0x03FF)
    let bits: UInt32
    if exp == 0 {
        if mant == 0 {
            bits = sign                                   // ±0
        } else {
            // Subnormal: normalize the mantissa, adjusting the exponent.
            exp = 1
            while (mant & 0x0400) == 0 { mant <<= 1; exp -= 1 }
            mant &= 0x03FF
            bits = sign | (UInt32(exp + 112) << 23) | (mant << 13)
        }
    } else if exp == 0x1F {
        bits = sign | 0x7F80_0000 | (mant << 13)          // ±inf / NaN
    } else {
        bits = sign | (UInt32(exp + 112) << 23) | (mant << 13)  // normal (15→127 bias shift = +112)
    }
    return Float(bitPattern: bits)
}
#endif
