#ifndef NEUXS_TRANSPORT_CUH
#define NEUXS_TRANSPORT_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace neuxs {

// NOTE: the previous version of this file had the `#endif` above the
// namespace block and a stray `public` keyword at namespace scope — both
// were compile errors. Fixed.
    enum class EventType { COLLIDE, ESCAPE, DIE };

// Simple 64-bit LCG (Knuth / MMIX constants). Adequate for a transport
// skeleton; a production code would use cuRAND or a counter-based generator
// (Philox / Threefry). We keep this header-only so we don't drag -lcurand
// into the link line just to write a couple of demo kernels.
    struct SimpleRNG {
        uint64_t _state;

        __host__ __device__ explicit SimpleRNG(uint64_t seed = 1ULL) : _state(seed) {}

        // Uniform in (0, 1]. Top 53 bits → a double-precision mantissa.
        __device__ __forceinline__ double nextDouble() {
            _state = _state * 6364136223846793005ULL + 1442695040888963407ULL;
            uint64_t hi = (_state >> 11);
            if (hi == 0)
                hi = 1; // exclude exact zero so log() is well-defined downstream
            return static_cast<double>(hi) * (1.0 / static_cast<double>(1ULL << 53));
        }

        __device__ __forceinline__ float nextFloat() {
            return static_cast<float>(nextDouble());
        }
    };

// Minimal Particle definition. Templated on precision to stay consistent
// with the rest of the stack.
    template <typename FPrecision> struct Particle {
        FPrecision _position[3] = {static_cast<FPrecision>(0),
                                   static_cast<FPrecision>(0),
                                   static_cast<FPrecision>(0)};
        FPrecision _direction[3] = {static_cast<FPrecision>(0),
                                    static_cast<FPrecision>(0),
                                    static_cast<FPrecision>(1)};
        FPrecision _energy = static_cast<FPrecision>(0);
        FPrecision _weight = static_cast<FPrecision>(1);
        unsigned int _cell_id = 0;
        bool _alive = true;
        SimpleRNG _rng{1ULL};
    };

} // namespace neuxs

#endif // NEUXS_TRANSPORT_CUH
