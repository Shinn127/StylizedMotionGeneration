#version 410
#define PI 3.14159265358979323846264338327950288
#define SSAO_SAMPLE_NUM 9

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

vec3 Rand(vec2 seed)
{
    return 2.0 * fract(sin(dot(seed, vec2(12.9898, 78.233))) * vec3(43758.5453, 21383.21227, 20431.20563)) - 1.0;
}

vec2 Spiral(int sampleIndex, float turns, float seed)
{
	float alpha = (float(sampleIndex) + 0.5) / float(SSAO_SAMPLE_NUM);
	float angle = alpha * (turns * 2.0 * PI) + 2.0 * PI * seed;
	return alpha * vec2(cos(angle), sin(angle));
}

out vec4 finalColor;

void main()
{
    float depth = texture(gbufferDepth, fragTexCoord).r;
    if (depth >= 0.99999) { discard; }

    vec3 fragNormal = texture(gbufferNormal, fragTexCoord).xyz * 2.0 - 1.0;
    
    vec3 seed = Rand(fragTexCoord);
    
    float bias = 0.025f;
    float radius = 0.5f;
    float turns = 7.0f;

    vec3 norm = mat3(camView) * fragNormal;
    vec3 base = CameraSpace(fragTexCoord, texture(gbufferDepth, fragTexCoord).r);
    float occ = 0.0;
    for (int i = 0; i < SSAO_SAMPLE_NUM; i++)
    {
        vec3 next = base + radius * vec3(Spiral(i, turns, seed.x), 0.0);
        vec4 ntex = camProj * vec4(next, 1);
        vec2 sampleTexCoord = (ntex.xy / ntex.w) * 0.5f + 0.5f;
        if (sampleTexCoord.x <= 0.0 || sampleTexCoord.x >= 1.0 ||
            sampleTexCoord.y <= 0.0 || sampleTexCoord.y >= 1.0) { continue; }
        vec3 actu = CameraSpace(sampleTexCoord, texture(gbufferDepth, sampleTexCoord).r);
        vec3 diff = actu - base;
        float vv = dot(diff, diff);
        float vn = dot(diff, norm) - bias;
        float f = max(radius*radius - vv, 0.0);
        occ += f*f*f*max(vn / (0.001 + vv), 0.0);
    }
    occ = occ / pow(radius, 6.0);

    float ssao = max(0.0, 1.0 - occ * ssaoIntensity * (5.0 / float(SSAO_SAMPLE_NUM)));

    finalColor.r = ssao;
    finalColor.g = 1.0f;
    finalColor.b = 1.0f;
    finalColor.a = 1.0f;
}
