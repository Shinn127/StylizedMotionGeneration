#version 410

// Separable Gaussian blur for EVSM moment maps. Moments are 3D-polynomial in
// depth, not linear, so blurring the *warped* moments is exactly what EVSM
// needs: the Chebyshev bound consumes mean/variance of the warped depth, and
// those are what this pass averages. Sampling clamps to the edge so adjacent
// cascade tiles never bleed into each other.

in vec2 fragTexCoord;

uniform sampler2D inputTexture;
uniform vec2 invTextureResolution;
uniform vec2 blurDirection;

out vec4 finalColor;

void main()
{
    // 9-tap sigma ~ 2.2 kernel (weights sum to 1.0). Kept as 9 straight taps:
    // measured on the target GPU, this chain is bandwidth-bound and coalesced
    // row reads beat both a 5-fetch bilinear fold and a single-pass 2D
    // outer product (cache thrash from scattered 2D offsets).
    const float weights[5] = float[5](0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

    vec2 step = blurDirection * invTextureResolution;
    vec4 sum = texture(inputTexture, fragTexCoord).rgba * weights[0];
    for (int i = 1; i < 5; ++i) {
        sum += texture(inputTexture, fragTexCoord + step * float(i)).rgba * weights[i];
        sum += texture(inputTexture, fragTexCoord - step * float(i)).rgba * weights[i];
    }
    finalColor = sum;
}
