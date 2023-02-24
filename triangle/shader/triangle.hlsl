struct PSInput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

static float2 positions[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5),
};

static float3 colors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
PSInput VSMain(uint vertexID: SV_VertexID) {
    PSInput result;
    result.position = float4(positions[vertexID], 0.0, 1.0);
    result.color = colors[vertexID];
    return result;
}

[shader("pixel")]
float4 PSMain(PSInput input) : SV_Target {
    return float4(input.color, 1.0);
}
