#version 410

// EVSM shadow map pass: store warped depth moments instead of raw depth.
// The light projection is orthographic, so gl_FragCoord.z is already linear
// in [0,1]. Moments are warped with s = exp(k*(z-1)) so an empty texel at the
// far plane stores exactly (1, 1, ?, ?) — the fixed WHITE clear in
// begin_shadow_map is then already the correct "no occluder" value and no
// float-clear machinery is needed. The negative-warp channels are written for
// the 4-moment EVSM; M1 lights with the positive pair only.
//
// The hard precision cap: exp(k*(z-1)) for z=0 (near plane) is e^-k, while
// e^(2k(1)) = 1 is the largest second moment. For k=30 the smallest stored
// value is ~9e-14 — comfortably inside float32 subnormals-free range.

uniform float evsmPosK;
uniform float evsmNegK;

out vec4 fragColor;

void main()
{
    float z = gl_FragCoord.z;
    float sPos = exp(evsmPosK * (z - 1.0));
    float sNeg = exp(evsmNegK * (1.0 - z));
    fragColor = vec4(sPos, sPos * sPos, sNeg, sNeg * sNeg);
    gl_FragDepth = gl_FragCoord.z;
}
