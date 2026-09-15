#version 410
in vec2 fragTexCoord;

uniform sampler2D gbufferColor;
uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform sampler2D ssao;
uniform sampler2D shadowMap;

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
uniform float exposure;
uniform float camClipNear;
uniform float camClipFar;
uniform float lightClipNear;
uniform float lightClipFar;

out vec4 finalColor;

#define PI 3.14159265358979323846264338327950288

vec3 ToGamma(in vec3 col)
{
    return vec3(pow(col.x, 2.2), pow(col.y, 2.2), pow(col.z, 2.2));
}

vec3 FromGamma(in vec3 col)
{
    return vec3(pow(col.x, 1.0/2.2), pow(col.y, 1.0/2.2), pow(col.z, 1.0/2.2));
}

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

float NonlinearDepth(float depth, float near, float far)
{
    return (((2.0 * near) / depth) - far - near) / (near - far);
}

float ShadowFactor(vec3 position, vec3 normal)
{
    vec4 lightPosition = lightViewProj * vec4(position + 0.01 * normal, 1.0);
    lightPosition.xyz = (lightPosition.xyz / lightPosition.w + 1.0) * 0.5;
    float inside = float(
        lightPosition.x > 0.0 && lightPosition.x < 1.0 &&
        lightPosition.y > 0.0 && lightPosition.y < 1.0 &&
        lightPosition.z > 0.0 && lightPosition.z < 1.0);
    float receiverDepth = LinearDepth(lightPosition.z, lightClipNear, lightClipFar);
    float blockerDepth = texture(shadowMap, lightPosition.xy).r;
    return 1.0 - inside * float(receiverDepth - 0.000005 > blockerDepth);
}

void main()
{
    // Unpack GBuffer
    
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth == 1.0f) { discard; }

    vec4 colorAndSpec = texture(gbufferColor, fragTexCoord);
    vec4 normalAndGlossiness = texture(gbufferNormal, fragTexCoord);
    vec3 positionClip = vec3(fragTexCoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0f - 1.0f;
    vec4 fragPositionHomo = camInvViewProj * vec4(positionClip, 1);
    vec3 fragPosition = fragPositionHomo.xyz / fragPositionHomo.w;
    vec4 ssaoData = texture(ssao, fragTexCoord);
    vec3 fragNormal = normalAndGlossiness.xyz * 2.0f - 1.0f;
    vec3 albedo = colorAndSpec.xyz;
    float specularity = colorAndSpec.w;
    float glossiness = normalAndGlossiness.w * 100.0f;
    float ambientShadow = ssaoData.r;
    float sunShadow = ShadowFactor(fragPosition, normalize(fragNormal));
    
    // Compute lighting
    
    vec3 eyeDir = normalize(fragPosition - camPos);

    vec3 lightSunColor = FromGamma(sunColor);
    vec3 lightSunHalf = normalize(-lightDir - eyeDir);

    vec3 lightSkyColor = FromGamma(skyColor);
    vec3 skyDir = vec3(0.0, -1.0, 0.0);
    vec3 lightSkyHalf = normalize(-skyDir - eyeDir);

    float sunFactorDiff = max(dot(fragNormal, -lightDir), 0.0);
    float sunFactorSpec = specularity *
        ((glossiness+2.0) / (8.0 * PI)) *
        pow(max(dot(fragNormal, lightSunHalf), 0.0), glossiness);

    float skyFactorDiff = max(dot(fragNormal, -skyDir), 0.0);
    float skyFactorSpec = specularity *
        ((glossiness+2.0) / (8.0 * PI)) *
        pow(max(dot(fragNormal, lightSkyHalf), 0.0), glossiness);

    float groundFactorDiff = max(dot(fragNormal, skyDir), 0.0);
    
    // Combine
    
    vec3 ambient = ambientShadow * ambientStrength * lightSkyColor * albedo;

    vec3 diffuse = sunShadow * sunStrength * lightSunColor * albedo * sunFactorDiff +
        groundStrength * lightSkyColor * albedo * groundFactorDiff +
        skyStrength * lightSkyColor * albedo * skyFactorDiff;

    float specular = sunShadow * sunStrength * sunFactorSpec + skyStrength * skyFactorSpec;

    vec3 final = diffuse + ambient + specular;
    
    finalColor = vec4(ToGamma(exposure * final), 1.0f);
    //finalColor = vec4(ToGamma(vec3(ambientShadow, ambientShadow, ambientShadow)), 1.0f);
    //finalColor = vec4(ToGamma(vec3(sunShadow, sunShadow, sunShadow)), 1.0f);
    //finalColor = vec4(ToGamma(vec3(specular, specular, specular)), 1.0f);
    gl_FragDepth = NonlinearDepth(depth, camClipNear, camClipFar);
}
