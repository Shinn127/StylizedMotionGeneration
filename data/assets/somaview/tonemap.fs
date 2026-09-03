#version 410

in vec2 fragTexCoord;

uniform sampler2D inputTexture;
uniform float exposure;
// 0 ACES (default), 1 Reinhard, 2 AgX (minified fit)
uniform int toneCurve;

out vec4 finalColor;

vec3 ACESApprox(vec3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 Reinhard(vec3 color)
{
    return color / (1.0 + color);
}

// Minified AgX fit (Benjamin Wrensch's MinifiedAgX): soft shoulder, desaturating
// highlights, no clipped primaries. Ordered-applied matrix, log encode, fixed
// contrast curve, then decode back to display linear.
vec3 AgxDefaultContrastApprox(vec3 x)
{
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 AgX(vec3 color)
{
    const mat3 agxEnc = mat3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0783847640628302, 0.878468636469772, 0.0788490078745569,
        0.079142364254093, 0.0791799115125815, 0.879142448299278);
    const float minEv = -12.47393;
    const float maxEv = 4.026069;

    color = agxEnc * color;
    color = clamp(log2(max(color, vec3(1e-10))), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);
    return AgxDefaultContrastApprox(color);
}

vec3 AgXInverse(vec3 color)
{
    const mat3 agxDec = mat3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

    color = agxDec * color;
    return pow(max(color, vec3(0.0)), vec3(2.2));
}

vec3 LinearToSRGB(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 ToneMap(vec3 color)
{
    if (toneCurve == 1) { return Reinhard(color); }
    if (toneCurve == 2) { return AgXInverse(AgX(color)); }
    return ACESApprox(color);
}

void main()
{
    vec3 hdr = texture(inputTexture, fragTexCoord).rgb;
    finalColor = vec4(LinearToSRGB(ToneMap(exposure * hdr)), 1.0);
}
