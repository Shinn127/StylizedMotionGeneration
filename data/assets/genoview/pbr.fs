#version 410

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;
uniform float pbrMetallic;
uniform float pbrRoughness;
uniform float camClipNear;
uniform float camClipFar;

layout (location = 0) out vec4 gbufferColor;
layout (location = 1) out vec4 gbufferNormal;

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

vec3 SRGBToLinear(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2));
}

void main()
{
    vec3 albedo = SRGBToLinear(fragColor.rgb * colDiffuse.rgb);
    vec3 normal = normalize(fragNormal);
    gbufferColor = vec4(albedo, clamp(pbrMetallic, 0.0, 1.0));
    gbufferNormal = vec4(normal * 0.5 + 0.5, clamp(pbrRoughness, 0.04, 1.0));
    gl_FragDepth = LinearDepth(gl_FragCoord.z, camClipNear, camClipFar);
}
