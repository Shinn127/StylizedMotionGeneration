#version 410

// PBR shadow maps use an orthographic light projection. The default depth
// generated from gl_Position is already the normalized light-space depth.
void main()
{
    gl_FragDepth = gl_FragCoord.z;
}
