#version 410

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
in vec4 fragTangent;

uniform vec4 colDiffuse;
uniform vec4 materialBaseColor;
uniform float pbrMetallic;
uniform float pbrRoughness;
uniform float pbrAo;
uniform sampler2D baseColorMap;
uniform sampler2D metallicRoughnessMap;
uniform sampler2D normalMap;
uniform int useBaseColorMap;
uniform int useMetallicRoughnessMap;
uniform int useNormalMap;
uniform int pbrGroundPattern;
uniform float camClipNear;
uniform float camClipFar;

layout (location = 0) out vec4 gbufferColor;
layout (location = 1) out vec4 gbufferNormal;
layout (location = 2) out float gbufferMaterialAO;

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

vec3 SRGBToLinear(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2));
}

float Grid(in vec2 uv, in float lineWidth)
{
    vec4 uvDDXY = vec4(dFdx(uv), dFdy(uv));
    vec2 uvDeriv = vec2(length(uvDDXY.xz), length(uvDDXY.yw));
    float targetWidth = lineWidth > 0.5 ? 1.0 - lineWidth : lineWidth;
    vec2 drawWidth = clamp(vec2(targetWidth), uvDeriv, vec2(0.5));
    vec2 lineAA = uvDeriv * 1.5;
    vec2 gridUV = abs(fract(uv) * 2.0 - 1.0);
    gridUV = lineWidth > 0.5 ? gridUV : 1.0 - gridUV;
    vec2 g2 = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    g2 *= clamp(targetWidth / drawWidth, 0.0, 1.0);
    g2 = mix(g2, vec2(targetWidth), clamp(uvDeriv * 2.0 - 1.0, 0.0, 1.0));
    g2 = lineWidth > 0.5 ? 1.0 - g2 : g2;
    return mix(g2.x, 1.0, g2.y);
}

float Checker(in vec2 uv)
{
    vec4 uvDDXY = vec4(dFdx(uv), dFdy(uv));
    vec2 w = vec2(length(uvDDXY.xz), length(uvDDXY.yw));
    vec2 i = 2.0 * (abs(fract((uv - 0.5 * w) * 0.5) - 0.5) -
        abs(fract((uv + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;
}

vec3 ResolveNormal(vec3 geometricNormal, vec2 texcoord)
{
    // Gram-Schmidt re-orthogonalization keeps the TBN valid when interpolated
    // tangents drift away from the normal; degenerate tangents fall back to
    // the geometric normal instead of producing a garbage frame.
    vec3 normal = normalize(geometricNormal);
    vec3 tangent = fragTangent.xyz - normal * dot(normal, fragTangent.xyz);
    float tangentLength = length(tangent);
    if (tangentLength < 1e-5) { return normal; }
    tangent /= tangentLength;
    vec3 bitangent = cross(normal, tangent) * fragTangent.w;
    mat3 tbn = mat3(tangent, bitangent, normal);
    vec2 tangentNormalXY = texture(normalMap, texcoord).rg * 2.0 - 1.0;
    float tangentNormalZ = sqrt(max(0.0, 1.0 - dot(tangentNormalXY, tangentNormalXY)));
    return normalize(tbn * vec3(tangentNormalXY, tangentNormalZ));
}

void main()
{
    vec3 albedo = SRGBToLinear((fragColor.rgb * colDiffuse.rgb) * materialBaseColor.rgb);
    if (useBaseColorMap != 0) {
        albedo *= SRGBToLinear(texture(baseColorMap, fragTexCoord).rgb);
    }
    if (pbrGroundPattern != 0) {
        float gridFine = Grid(20.0 * 10.0 * fragTexCoord, 0.025);
        float gridCoarse = Grid(2.0 * 10.0 * fragTexCoord, 0.02);
        float check = Checker(2.0 * 10.0 * fragTexCoord);
        float groundValue = mix(mix(mix(0.9, 0.95, check), 0.85, gridFine), 1.0, gridCoarse);
        albedo *= groundValue;
    }
    vec3 normal = useNormalMap != 0 ? ResolveNormal(fragNormal, fragTexCoord) : normalize(fragNormal);
    float metallic = clamp(pbrMetallic, 0.0, 1.0);
    float roughness = clamp(pbrRoughness, 0.04, 1.0);
    float ao = clamp(pbrAo, 0.0, 1.0);
    if (useMetallicRoughnessMap != 0) {
        vec3 packedMaterial = texture(metallicRoughnessMap, fragTexCoord).rgb;
        metallic = packedMaterial.r;
        roughness = clamp(packedMaterial.g, 0.04, 1.0);
        ao = packedMaterial.b;
    }
    gbufferColor = vec4(albedo, metallic);
    gbufferNormal = vec4(normal * 0.5 + 0.5, roughness);
    gbufferMaterialAO = ao;
    gl_FragDepth = LinearDepth(gl_FragCoord.z, camClipNear, camClipFar);
}
