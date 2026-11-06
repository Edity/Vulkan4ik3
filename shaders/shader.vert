#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 uv;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColor;
} scene;

// One shared dynamic UBO. The selected ModelUBO is chosen by the dynamic
// offset passed to vkCmdBindDescriptorSets for descriptor set 1.
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 tint;
    // x = shininess, y = specular strength, z = unlit flag, w = emissive strength
    vec4 materialParams;
} modelData;

void main() {
    vec4 world = modelData.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(transpose(inverse(mat3(modelData.model))) * inNormal);
    uv = inUV;
    gl_Position = scene.proj * scene.view * world;
}
