#version 410

in vec2 fragTexCoord;

uniform sampler2D inputTexture;
uniform float exposure;

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

vec3 LinearToSRGB(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}

void main()
{
    vec3 hdr = texture(inputTexture, fragTexCoord).rgb;
    finalColor = vec4(LinearToSRGB(ACESApprox(exposure * hdr)), 1.0);
}
