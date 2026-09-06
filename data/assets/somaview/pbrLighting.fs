#version 410

in vec2 fragTexCoord;

uniform sampler2D gbufferColor;
uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform sampler2D ssao;
uniform sampler2D materialAO;
uniform sampler2D shadowMap0;
uniform sampler2D shadowMap1;
uniform sampler2D shadowMap2;
uniform samplerCube environmentMap;
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLut;

uniform vec3 camPos;
uniform mat4 camView;
uniform mat4 camInvViewProj;
uniform mat4 lightViewProj0;
uniform mat4 lightViewProj1;
uniform mat4 lightViewProj2;
uniform vec3 lightDir;
uniform vec3 sunColor;
uniform float sunStrength;
uniform vec3 skyColor;
uniform float skyStrength;
uniform float groundStrength;
uniform float ambientStrength;
uniform float camClipNear;
uniform float camClipFar;
uniform float iblStrength;
uniform float prefilterMaxLod;
uniform int useIBL;
uniform int whiteBackground;
// EVSM: warped-depth moments replace depth-compare + PCF entirely. No bias,
// no texel size — variance bounds handle self-shadowing without acne.
uniform float evsmPosK;
uniform float evsmNegK;
uniform float evsmLightBleed;
uniform float evsmMinVariance;
uniform vec3 cascadeSplits;
uniform float cascadeBlendFraction;
// 0 final image, 1 shadow, 2 direct diffuse, 3 direct specular, 4 indirect light
uniform int debugMode;


out vec4 finalColor;

#define PI 3.14159265358979323846264338327950288
// Background marker written by the sky branch under --white-background. Must
// stay representable in RGBA16F storage (half-float max 65504) and far above
// any physical radiance the scene can produce (sun strength <= ~1).
#define BACKGROUND_SENTINEL 6.0e4

float NonlinearDepth(float depth, float near, float far)
{
    return (((2.0 * near) / depth) - far - near) / (near - far);
}

vec3 SRGBToLinear(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2));
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

float ChebyshevUpperBound(vec2 moments, float mean, float minVariance)
{
    // One-sided Chebyshev (Cantrell's inequality) on the warped depth:
    // P(x > mean) <= variance / (variance + (mean - m1)^2).
    float d = mean - moments.x;
    float variance = max(moments.y - moments.x * moments.x, minVariance);
    float p = variance / (variance + d * d);
    // Behind the mean depth the bound is meaningless; that region is lit.
    return max(mean <= moments.x ? 1.0 : p, 0.0);
}

// Mirror of ChebyshevUpperBound for the negative-warp distribution. There the
// warp exp(k(1-z)) DECREASES with depth, so "receiver occluded" means its
// warped value is SMALLER than the stored mean. Mirroring the distribution
// through Y = 1 - X flips the guarded tail back to the upper side where the
// shared Cantelli bound applies; the mirrored moments are E[Y] = 1 - m1 and
// E[Y^2] = 1 - 2 m1 + m2 (variance is mirror-invariant). Background texels
// (m1 = m2 = 1, receiver mean <= 1) land in the always-lit branch.
float ChebyshevUpperBoundNeg(vec2 moments, float mean, float minVariance)
{
    float mirroredMean = 1.0 - mean;
    float mirroredM1 = 1.0 - moments.x;
    float mirroredM2 = 1.0 - 2.0 * moments.x + moments.y;
    return ChebyshevUpperBound(vec2(mirroredM1, mirroredM2), mirroredMean, minVariance);
}

float ShadowFactorFor(vec3 position, vec3 normal, mat4 lightViewProj, sampler2D shadowMap, float baseBias)
{
    vec4 lightPosition = lightViewProj * vec4(position, 1.0);
    lightPosition.xyz = (lightPosition.xyz / lightPosition.w + 1.0) * 0.5;
    bool inside = lightPosition.x > 0.0 && lightPosition.x < 1.0 &&
        lightPosition.y > 0.0 && lightPosition.y < 1.0 &&
        lightPosition.z > 0.0 && lightPosition.z < 1.0;
    if (!inside) { return 1.0; }
    // Warped moments were blurred offline (evsmBlur.fs), so this is a single
    // bilinear fetch — the blur radius IS the penumbra. The positive-warp pair
    // bounds transmittance from one side; the negative-warp pair from the
    // other, which is what suppresses light bleeding through thin occluders
    // (arms, feet). min() of both bounds, then reshape the residual
    // over-transmittance the bound allows.
    vec4 m = texture(shadowMap, lightPosition.xy);
    float receiverZ = lightPosition.z;
    float meanPos = exp(evsmPosK * (receiverZ - 1.0));
    float meanNeg = exp(evsmNegK * (1.0 - receiverZ));
    float shadowPos = ChebyshevUpperBound(m.rg, meanPos, evsmMinVariance);
    float shadowNeg = ChebyshevUpperBoundNeg(m.ba, meanNeg, evsmMinVariance);
    float shadow = min(shadowPos, shadowNeg);
    shadow = clamp((shadow - evsmLightBleed) / (1.0 - evsmLightBleed), 0.0, 1.0);
    return shadow;
}

float ShadowFactor(vec3 position, vec3 normal, float cameraDepth)
{
    float blendWidth0 = max((cascadeSplits.x - camClipNear) * cascadeBlendFraction, 1e-4);
    float blendWidth1 = max((cascadeSplits.y - cascadeSplits.x) * cascadeBlendFraction, 1e-4);

    if (cameraDepth < cascadeSplits.x - blendWidth0) {
        return ShadowFactorFor(position, normal, lightViewProj0, shadowMap0, 0.0);
    }
    if (cameraDepth <= cascadeSplits.x) {
        float shadow0 = ShadowFactorFor(position, normal, lightViewProj0, shadowMap0, 0.0);
        float shadow1 = ShadowFactorFor(position, normal, lightViewProj1, shadowMap1, 0.0);
        float blend = smoothstep(cascadeSplits.x - blendWidth0, cascadeSplits.x, cameraDepth);
        return mix(shadow0, shadow1, blend);
    }
    if (cameraDepth < cascadeSplits.y - blendWidth1) {
        return ShadowFactorFor(position, normal, lightViewProj1, shadowMap1, 0.0);
    }
    if (cameraDepth <= cascadeSplits.y) {
        float shadow1 = ShadowFactorFor(position, normal, lightViewProj1, shadowMap1, 0.0);
        float shadow2 = ShadowFactorFor(position, normal, lightViewProj2, shadowMap2, 0.0);
        float blend = smoothstep(cascadeSplits.y - blendWidth1, cascadeSplits.y, cameraDepth);
        return mix(shadow1, shadow2, blend);
    }
    return ShadowFactorFor(position, normal, lightViewProj2, shadowMap2, 0.0);
}

void main()
{
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth >= 0.99999) {
        // Procedural sky background: normally reconstruct the view ray through
        // the far plane and sample the environment cubemap (linear radiance
        // data — no sRGB decode); the fallback keeps a skyColor-tinted flat
        // dome so --disable-ibl stays coherent. The FSQ runs blended
        // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) — rlDisableColorBlend does not stick
        // through the batch flush — and RGBA16F attachments clamp negatives on
        // this GL backend, so the background marker is an unreachable large
        // radiance: real scene radiance stays far below BACKGROUND_SENTINEL.
        vec2 ndc = fragTexCoord * 2.0 - 1.0;
        vec4 farPointHomo = camInvViewProj * vec4(ndc, 1.0, 1.0);
        vec3 viewDir = normalize(farPointHomo.xyz / farPointHomo.w - camPos);
        vec3 sky = useIBL == 1
            ? textureLod(environmentMap, viewDir, 0.0).rgb
            : SRGBToLinear(skyColor) * 2.0;
        finalColor = vec4(sky, 1.0);
        if (whiteBackground == 1) { finalColor = vec4(vec3(BACKGROUND_SENTINEL), 1.0); }
        gl_FragDepth = 1.0;
        return;
    }

    vec3 positionClip = vec3(fragTexCoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0 - 1.0;
    vec4 positionHomo = camInvViewProj * vec4(positionClip, 1.0);
    vec3 position = positionHomo.xyz / positionHomo.w;
    vec4 colorMetallic = texture(gbufferColor, fragTexCoord);
    vec4 normalRoughness = texture(gbufferNormal, fragTexCoord);
    // AO only scales indirect light: SSAO covers small-scale occlusion,
    // the material AO attachment covers baked per-material occlusion.
    float ao = texture(ssao, fragTexCoord).r * texture(materialAO, fragTexCoord).r;

    vec3 albedo = colorMetallic.rgb;
    float metallic = colorMetallic.a;
    vec3 normal = normalize(normalRoughness.rgb * 2.0 - 1.0);
    float roughness = clamp(normalRoughness.a, 0.04, 1.0);
    vec3 view = normalize(camPos - position);
    float cameraDepth = -(camView * vec4(position, 1.0)).z;
    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    vec3 sun = normalize(-lightDir);
    vec3 halfVector = view + sun;
    float halfLength = length(halfVector);
    halfVector = halfLength > 1e-4 ? halfVector / halfLength : normal;
    float nDotL = max(dot(normal, sun), 0.0);
    float nDotV = max(dot(normal, view), 1e-4);
    float nDotH = max(dot(normal, halfVector), 0.0);
    float vDotH = max(dot(view, halfVector), 0.0);

    float distribution = DistributionGGX(nDotH, roughness);
    float geometry = GeometrySmith(nDotV, nDotL, roughness);
    vec3 fresnel = FresnelSchlick(vDotH, f0);
    vec3 specular = distribution * geometry * fresnel / max(4.0 * nDotV * nDotL, 1e-4);
    vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic) * albedo / PI;

    vec3 sunRadiance = SRGBToLinear(sunColor) * (sunStrength * PI);
    float shadow = ShadowFactor(position, normal, cameraDepth);
    vec3 direct = (diffuse + specular) * sunRadiance * nDotL * shadow;

    vec3 skyRadiance = SRGBToLinear(skyColor);
    float skyFactor = max(normal.y, 0.0);
    float groundFactor = max(-normal.y, 0.0);
    vec3 fallbackAmbient = (1.0 - metallic) * albedo * skyRadiance *
        (ambientStrength + skyStrength * skyFactor + groundStrength * groundFactor) * ao;

    vec3 ambient = fallbackAmbient;
    if (useIBL == 1) {
        vec3 reflection = reflect(-view, normal);
        vec3 irradiance = texture(irradianceMap, normal).rgb;
        vec3 prefiltered = textureLod(prefilterMap, reflection, roughness * prefilterMaxLod).rgb;
        vec2 brdf = texture(brdfLut, vec2(nDotV, roughness)).rg;
        vec3 iblFresnel = FresnelSchlick(nDotV, f0);
        vec3 diffuseIBL = (1.0 - metallic) * albedo * irradiance;
        vec3 specularIBL = prefiltered * (iblFresnel * brdf.x + brdf.y);
        ambient = (diffuseIBL + specularIBL) * ao * iblStrength;
    }

    finalColor = vec4(direct + ambient, 1.0);
    if (debugMode == 1) { finalColor = vec4(vec3(shadow), 1.0); }
    else if (debugMode == 2) { finalColor = vec4(diffuse * sunRadiance * nDotL * shadow, 1.0); }
    else if (debugMode == 3) { finalColor = vec4(specular * sunRadiance * nDotL * shadow, 1.0); }
    else if (debugMode == 4) { finalColor = vec4(ambient, 1.0); }
    gl_FragDepth = NonlinearDepth(depth, camClipNear, camClipFar);
}
