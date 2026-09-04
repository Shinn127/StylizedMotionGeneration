#version 410
#define PI 3.14159265358979323846264338327950288
#define GTAO_DIRECTION_COUNT 4
#define GTAO_STEP_COUNT 4

in vec2 fragTexCoord;

uniform sampler2D gbufferNormal;
uniform sampler2D gbufferDepth;
uniform mat4 camView;
uniform mat4 camProj;
uniform mat4 camInvProj;
uniform float camClipNear;
uniform float camClipFar;
uniform float ssaoIntensity;

float NonlinearDepth(float depth, float near, float far)
{
    return (((2.0 * near) / depth) - far - near) / (near - far);
}

vec3 CameraSpace(vec2 texcoord, float depth)
{
    vec4 positionClip = vec4(vec3(texcoord, NonlinearDepth(depth, camClipNear, camClipFar)) * 2.0 - 1.0, 1.0);
    vec4 position = camInvProj * positionClip;
    return position.xyz / position.w;
}

vec2 ProjectedRadius(float viewDepth, float worldRadius)
{
    float depth = max(-viewDepth, 0.001);
    return 0.5 * worldRadius / depth * vec2(camProj[0][0], camProj[1][1]);
}

float Hash(vec2 seed)
{
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}

out vec4 finalColor;

void main()
{
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth >= 0.99999) {
        finalColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    vec3 normal = normalize(texture(gbufferNormal, fragTexCoord).xyz * 2.0 - 1.0);
    vec3 viewNormal = normalize(mat3(camView) * normal);
    vec3 basePosition = CameraSpace(fragTexCoord, depth);
    float rotation = Hash(fragTexCoord) * 2.0 * PI;
    float radius = 0.5;
    float horizonBias = 0.05;
    vec2 radiusUV = ProjectedRadius(basePosition.z, radius);
    float occlusion = 0.0;

    // Horizon search over eight evenly spaced screen-space rays. The maximum
    // per ray keeps a nearby blocker from being counted once per step.
    for (int directionIndex = 0; directionIndex < GTAO_DIRECTION_COUNT; ++directionIndex)
    {
        float angle = rotation + float(directionIndex) * PI / 4.0;
        vec2 ray = vec2(cos(angle), sin(angle));
        for (int sideIndex = 0; sideIndex < 2; ++sideIndex)
        {
            float side = sideIndex == 0 ? -1.0 : 1.0;
            float horizon = 0.0;
            for (int stepIndex = 1; stepIndex <= GTAO_STEP_COUNT; ++stepIndex)
            {
                float stepFraction = float(stepIndex) / float(GTAO_STEP_COUNT);
                vec2 sampleTexCoord = fragTexCoord + ray * side * radiusUV * stepFraction;
                if (sampleTexCoord.x <= 0.0 || sampleTexCoord.x >= 1.0 ||
                    sampleTexCoord.y <= 0.0 || sampleTexCoord.y >= 1.0) { continue; }

                float sampleDepth = texture(gbufferDepth, sampleTexCoord).r;
                if (sampleDepth >= 0.99999) { continue; }
                vec3 samplePosition = CameraSpace(sampleTexCoord, sampleDepth);
                vec3 delta = samplePosition - basePosition;
                float distanceToSample = length(delta);
                if (distanceToSample <= 1e-4 || distanceToSample > radius * 1.75) { continue; }

                vec3 direction = delta / distanceToSample;
                float horizonCosine = max(dot(direction, viewNormal) - horizonBias, 0.0);
                float falloff = 1.0 - smoothstep(0.0, radius, distanceToSample);
                vec3 sampleNormal = normalize(texture(gbufferNormal, sampleTexCoord).xyz * 2.0 - 1.0);
                float normalWeight = clamp(0.5 + 0.5 * dot(sampleNormal, normal), 0.25, 1.0);
                horizon = max(horizon, horizonCosine * falloff * normalWeight);
            }
            occlusion += horizon;
        }
    }

    occlusion /= float(GTAO_DIRECTION_COUNT * 2);
    float ssao = clamp(1.0 - occlusion * ssaoIntensity * 4.0, 0.0, 1.0);
    finalColor = vec4(ssao, 1.0, 1.0, 1.0);
}
