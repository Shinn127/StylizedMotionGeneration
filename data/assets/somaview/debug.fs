#version 410

// PBR debug display pass. Consumes the debug quantities produced either by the
// GBuffer attachments (modes 1-6) or by pbrLighting.fs's debugMode output
// (modes 7-11). Display transform is exposure + LinearToSRGB only: debug views
// must stay linear interpretable, so ACES tonemapping and FXAA are skipped.

in vec2 fragTexCoord;

uniform sampler2D texGbufferColor;
uniform sampler2D texGbufferNormal;
uniform sampler2D texGbufferDepth;
uniform sampler2D texSSAO;
uniform sampler2D texLighted;

uniform int debugMode;
uniform float exposure;

out vec4 finalColor;

vec3 LinearToSRGB(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 DisplayHDR(vec3 color)
{
    return LinearToSRGB(exposure * color);
}

void main()
{
    if (texture(texGbufferDepth, fragTexCoord).r >= 0.99999) {
        finalColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 colorMetallic = texture(texGbufferColor, fragTexCoord);
    vec4 normalRoughness = texture(texGbufferNormal, fragTexCoord);
    float depth = texture(texGbufferDepth, fragTexCoord).r;
    float ao = texture(texSSAO, fragTexCoord).r;
    vec3 lighted = texture(texLighted, fragTexCoord).rgb;

    // 1 base_color, 2 metallic, 3 roughness, 4 normal, 5 depth, 6 ao,
    // 7 shadow, 8 diffuse, 9 specular, 10 ibl, 11 hdr
    vec3 color = vec3(0.0);
    if (debugMode == 1) { color = LinearToSRGB(colorMetallic.rgb); }
    else if (debugMode == 2) { color = vec3(colorMetallic.a); }
    else if (debugMode == 3) { color = vec3(normalRoughness.a); }
    else if (debugMode == 4) { color = normalRoughness.rgb; }
    else if (debugMode == 5) { color = vec3(1.0 - depth); }
    else if (debugMode == 6) { color = vec3(ao); }
    else if (debugMode == 7) { color = vec3(lighted.r); }
    else if (debugMode == 8) { color = DisplayHDR(lighted); }
    else if (debugMode == 9) { color = DisplayHDR(lighted); }
    else if (debugMode == 10) { color = DisplayHDR(lighted); }
    else if (debugMode == 11) { color = DisplayHDR(lighted); }

    finalColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
