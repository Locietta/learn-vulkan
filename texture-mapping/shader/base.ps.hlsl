#include "common.hlsli"

[[vk::binding(1)]] Texture2D gTexture;

[[vk::binding(1)]] SamplerState gSampler;

float4 main(PSInput input) : SV_Target {
    return gTexture.Sample(gSampler, input.uv);
}
