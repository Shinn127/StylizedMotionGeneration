#version 410

in vec2 fragTexCoord;

uniform sampler2D gbufferColor;
uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform sampler2D ssao;

uniform vec3 camPos;
uniform mat4 camInvViewProj;
uniform vec3 lightDir;
uniform vec3 sunColor;
uniform float sunStrength;
uniform vec3 skyColor;
uniform float skyStrength;
uniform float groundStrength;
uniform float ambientStrength;
uniform float exposure;
uniform float camClipNear;
uniform float camClipFar;

out vec4 finalColor;

#define PI 3.14159265358979323846264338327950288

float NonlinearDepth(float depth, float near, float far)
{
    return (((2.0 * near) / depth) - far - near) / (near - far);
}

vec3 SRGBToLinear(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2));
}

vec3 LinearToSRGB(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 ACES(vec3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

float DistributionGGX(float nDotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1e-5);
}

float GeometrySchlickGGX(float nDotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 1e-5);
}

float GeometrySmith(float nDotV, float nDotL, float roughness)
{
    return GeometrySchlickGGX(nDotV, roughness) * GeometrySchlickGGX(nDotL, roughness);
}

vec3 FresnelSchlick(float vDotH, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
}

void main()
{
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth == 1.0) { discard; }

    vec3 positionClip = vec3(fragTexCoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0 - 1.0;
    vec4 positionHomo = camInvViewProj * vec4(positionClip, 1.0);
    vec3 position = positionHomo.xyz / positionHomo.w;
    vec4 colorMetallic = texture(gbufferColor, fragTexCoord);
    vec4 normalRoughness = texture(gbufferNormal, fragTexCoord);
    vec4 ssaoData = texture(ssao, fragTexCoord);

    vec3 albedo = colorMetallic.rgb;
    float metallic = colorMetallic.a;
    vec3 normal = normalize(normalRoughness.rgb * 2.0 - 1.0);
    float roughness = normalRoughness.a;
    vec3 view = normalize(camPos - position);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    vec3 sun = normalize(-lightDir);
    vec3 halfVector = normalize(view + sun);
    float nDotL = max(dot(normal, sun), 0.0);
    float nDotV = max(dot(normal, view), 0.0);
    float nDotH = max(dot(normal, halfVector), 0.0);
    float vDotH = max(dot(view, halfVector), 0.0);

    float distribution = DistributionGGX(nDotH, roughness);
    float geometry = GeometrySmith(nDotV, nDotL, roughness);
    vec3 fresnel = FresnelSchlick(vDotH, f0);
    vec3 specular = distribution * geometry * fresnel / max(4.0 * nDotV * nDotL, 1e-4);
    vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic) * albedo / PI;

    vec3 sunRadiance = SRGBToLinear(sunColor) * sunStrength;
    vec3 direct = (diffuse + specular) * sunRadiance * nDotL * ssaoData.g;

    vec3 skyRadiance = SRGBToLinear(skyColor);
    float skyFactor = max(normal.y, 0.0);
    float groundFactor = max(-normal.y, 0.0);
    vec3 ambient = (1.0 - metallic) * albedo * skyRadiance *
        (ambientStrength * ssaoData.r + skyStrength * skyFactor + groundStrength * groundFactor);

    vec3 color = ACES(exposure * (direct + ambient));
    finalColor = vec4(LinearToSRGB(color), 1.0);
    gl_FragDepth = NonlinearDepth(depth, camClipNear, camClipFar);
}
