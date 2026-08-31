#version 410

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexTangent;
in vec4 vertexColor;

uniform mat4 mvp;

out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
out vec4 fragTangent;

void main()
{
    fragPosition = vertexPosition;
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = vertexNormal;
    fragTangent = vertexTangent;

    gl_Position = mvp * vec4(fragPosition, 1.0f);
}
