#include "frame.cuh"
#include "FP16_frame.cuh"
#include "micro.cuh"
#include "depthwise_dconv3D_deltaW.cuh"

extern "C"
int flash_dk_fp32(
    const float* x,
    const float* dy,
    float* dw,
    int B, int H, int L, int K, int GZ,
    void* stream_ptr)
{
    if (!x || !dy || !dw ||
        B <= 0 || H <= 0 || H % 32 != 0 ||
        L <= 0 || K <= 0 ||
        GZ <= 0 || B % GZ != 0 ||
        (B / GZ) % 16 != 0)
        return -2;

    // x, dy: [B, 1, L, H]; dw: [1, K, H].
    // Caller must zero dw before launching.
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(stream_ptr);

    return bn_act_DWC3D_dW_GemmSK_CE_32_72<-1, false>(
        stream, GZ,
        x, 1, L,
        nullptr, nullptr, 0.0f, nullptr, nullptr,
        dy, 1, L,
        dw, 1, K,
        B, H, 0, K / 2,
        0.0f, 0.0f, 0.0f);
}
