#version 410

in vec2 fragTexCoord;

uniform sampler2D gbufferColor;
uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform sampler2D ssao;
uniform sampler2D shadowMap;
uniform samplerCube environmentMap;
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLut;

uniform vec3 camPos;
uniform mat4 camInvViewProj;
uniform mat4 lightViewProj;
uniform vec3 lightDir;
uniform vec3 sunColor;
uniform float sunStrength;
uniform vec3 skyColor;
uniform float skyStrength;
uniform float groundStrength;
uniform float ambientStrength;
uniform float camClipNear;
uniform float camClipFar;
uniform float lightClipNear;
uniform float lightClipFar;
uniform float iblStrength;
uniform float prefilterMaxLod;
uniform int useIBL;
// 0 final image, 1 shadow, 2 direct diffuse, 3 direct specular, 4 indirect light
uniform int debugMode;

out vec4 finalColor;

#define PI 3.14159265358979323846264338327950288

float NonlinearDepth(float depth, float near, float far)
{
    return (((2.0 * near) / depth) - far - near) / (near - far);
}

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
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

float ShadowFactor(vec3 position, vec3 normal)
{
    vec4 lightPosition = lightViewProj * vec4(position + 0.01 * normal, 1.0);
    lightPosition.xyz = (lightPosition.xyz / lightPosition.w + 1.0) * 0.5;
    bool inside = lightPosition.x > 0.0 && lightPosition.x < 1.0 &&
        lightPosition.y > 0.0 && lightPosition.y < 1.0 &&
        lightPosition.z > 0.0 && lightPosition.z < 1.0;
    if (!inside) { return 1.0; }
    float receiverDepth = LinearDepth(lightPosition.z, lightClipNear, lightClipFar);
    float blockerDepth = texture(shadowMap, lightPosition.xy).r;
    return 1.0 - float(receiverDepth - 0.000005 > blockerDepth);
}

void main()
{
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth >= 0.99999) { discard; }

    vec3 positionClip = vec3(fragTexCoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0 - 1.0;
    vec4 positionHomo = camInvViewProj * vec4(positionClip, 1.0);
    vec3 position = positionHomo.xyz / positionHomo.w;
    vec4 colorMetallic = texture(gbufferColor, fragTexCoord);
    vec4 normalRoughness = texture(gbufferNormal, fragTexCoord);
    float ao = texture(ssao, fragTexCoord).r;

    vec3 albedo = colorMetallic.rgb;
    float metallic = colorMetallic.a;
    vec3 normal = normalize(normalRoughness.rgb * 2.0 - 1.0);
    float roughness = clamp(normalRoughness.a, 0.04, 1.0);
    vec3 view = normalize(camPos - position);
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
    float shadow = ShadowFactor(position, normal);
    vec3 direct = (diffuse + specular) * sunRadiance * nDotL * shadow;

    vec3 skyRadiance = SRGBToLinear(skyColor);
    float skyFactor = max(normal.y, 0.0);
    float groundFactor = max(-normal.y, 0.0);
    vec3 fallbackAmbient = (1.0 - metallic) * albedo * skyRadiance *
        (ambientStrength + skyStrength * skyFactor + groundStrength * groundFactor) * ao;

    vec3 ambient = fallbackAmbient;
    if (useIBL == 1) {
        vec3 reflection = reflect(-view, normal);
        vec3 irradiance = SRGBToLinear(texture(irradianceMap, normal).rgb);
        vec3 environment = SRGBToLinear(texture(environmentMap, reflection).rgb);
        vec3 prefiltered = SRGBToLinear(textureLod(prefilterMap, reflection, roughness * prefilterMaxLod).rgb);
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
