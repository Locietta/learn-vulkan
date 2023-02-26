
cbuffer MVP : register(b0) {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

struct PSInput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

struct VSInput {
    float2 position : POSITION;
    float3 color : COLOR;
};

PSInput main(VSInput input) {
    PSInput result;

    const float4 world_position = mul(model, float4(input.position, 0.0, 1.0));
    const float4 view_position = mul(view, world_position);
    const float4 clip_position = mul(proj, view_position);

    result.position = clip_position;
    result.color = input.color;
    return result;
}
