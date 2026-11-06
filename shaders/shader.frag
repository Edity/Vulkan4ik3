#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColor;
} scene;

struct PointLight { vec4 position; vec4 colorIntensity; };
layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointBuffer;

struct SpotLight { vec4 position; vec4 direction; vec4 colorIntensity; vec4 coneParams; };
layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spotBuffer;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 tint;
    vec4 materialParams;
} modelData;

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;
layout(set = 1, binding = 2) uniform sampler2D specularTexture;
layout(set = 1, binding = 3) uniform sampler2D emissiveTexture;

// Push constants are reserved only for the optional animated sampling effect.
// flags bit 0 enables multi-sampling; model properties live in the dynamic UBO.
layout(push_constant) uniform SamplingControl {
    uint flags;
    float time;
    float modulation;
    float reserved;
} sampling;

vec3 blinnPhong(
    vec3 N, vec3 V, vec3 L,
    vec3 radiance,
    vec3 albedo,
    vec3 specularColor,
    float specularMask,
    float shininess
) {
    float ndotl = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * ndotl;
    vec3 H = normalize(L + V);
    float spec = ndotl > 0.0
        ? pow(max(dot(N, H), 0.0), max(shininess, 1.0))
        : 0.0;
    vec3 specular = specularColor * spec * specularMask;
    return (diffuse + specular) * radiance;
}

void main() {
    vec4 sampledAlbedo = texture(albedoTexture, uv);
    vec3 albedo = sampledAlbedo.rgb;

    // Optional animated, nontrivial sampling: sample the SAME image twice
    // with different, time-dependent UV coordinates, blend and modulate it.
    if ((sampling.flags & 1u) != 0u) {
        float t = sampling.time;
        vec2 uvA = uv + vec2(
            sin(uv.y * 18.0 + t * 1.7),
            cos(uv.x * 14.0 - t * 1.3)
        ) * 0.025;
        vec2 uvB = uv * 1.65 + vec2(t * 0.035, -t * 0.055);
        vec3 sampleA = texture(albedoTexture, uvA).rgb;
        vec3 sampleB = texture(albedoTexture, uvB).rgb;
        float blendFactor = 0.5 + 0.5 * sin(t * 1.2 + uv.y * 9.0);
        vec3 animatedModulation = vec3(
            0.85 + 0.15 * sin(t * 1.1),
            0.90 + 0.10 * sin(t * 0.8 + 1.0),
            0.90 + 0.10 * cos(t * 0.9)
        );
        albedo = mix(sampleA, sampleB, blendFactor)
               * animatedModulation * max(sampling.modulation, 0.0);
    }

    albedo *= modelData.tint.rgb;

    // Debug/light markers are rendered unlit, but still use the same material
    // descriptor layout and dynamic model UBO as regular meshes.
    if (modelData.materialParams.z > 0.5) {
        outColor = vec4(albedo, sampledAlbedo.a * modelData.tint.a);
        return;
    }

    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - worldPosition);
    float shininess = max(modelData.materialParams.x, 1.0);
    float specularStrength = max(modelData.materialParams.y, 0.0);
    float emissiveStrength = max(modelData.materialParams.w, 0.0);

    // The red channel of the specular map masks where highlights are visible.
    float specularMask = texture(specularTexture, uv).r * specularStrength;

    vec3 result = albedo * scene.ambientColor.rgb;
    vec3 directionalL = normalize(-scene.directionalDirection.xyz);
    result += blinnPhong(
        N, V, directionalL,
        scene.directionalColor.rgb * scene.directionalColor.a,
        albedo, vec3(1.0), specularMask, shininess
    );

    for (uint i = 0; i < pointBuffer.lights.length(); ++i) {
        PointLight light = pointBuffer.lights[i];
        vec3 delta = light.position.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.0001);
        vec3 L = delta * inversesqrt(d2);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a / d2;
        result += blinnPhong(N, V, L, radiance, albedo, vec3(1.0), specularMask, shininess);
    }

    for (uint i = 0; i < spotBuffer.lights.length(); ++i) {
        SpotLight light = spotBuffer.lights[i];
        vec3 delta = light.position.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.0001);
        vec3 L = delta * inversesqrt(d2);
        float theta = dot(normalize(light.direction.xyz), -L);
        float innerCos = light.coneParams.x;
        float outerCos = light.coneParams.y;
        float epsilon = max(innerCos - outerCos, 0.0001);
        float cone = clamp((theta - outerCos) / epsilon, 0.0, 1.0);
        cone = cone * cone * (3.0 - 2.0 * cone);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a * cone / d2;
        result += blinnPhong(N, V, L, radiance, albedo, vec3(1.0), specularMask, shininess);
    }

    // Emissive is added independently of Blinn-Phong lighting.
    result += texture(emissiveTexture, uv).rgb * emissiveStrength;

    // Simple tone mapping for high dynamic range accumulated lighting.
    result = result / (vec3(1.0) + result);
    outColor = vec4(result, sampledAlbedo.a * modelData.tint.a);
}
