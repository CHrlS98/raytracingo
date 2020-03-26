#pragma once
#include <basicmaterial.h>
#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
namespace materials
{
/// Jaune
const BasicMaterial yellow(
    glm::vec3(1.0f, 1.0f, 0.0f), // ka
    glm::vec3(1.0f, 1.0f, 0.0f), // kd
    glm::vec3(1.0f, 1.0f, 1.0f), // ks
    glm::vec3(0.2f, 0.2f, 0.2f), // kr
    30.0f                        // alpha
);

/// Violet 100% miroir
const BasicMaterial purple(
    glm::vec3(1.0f, 0.0f, 1.0f), // ka
    glm::vec3(1.0f, 0.0f, 1.0f), // kd
    glm::vec3(1.0f, 0.0f, 1.0f), // ks
    glm::vec3(1.0f, 1.0f, 1.0f), // kr
    30.0f                        // alpha
);

/// Miroir incolore
const BasicMaterial blackMirror(
    glm::vec3(0.1f, 0.1f, 0.1f), // ka
    glm::vec3(0.0f, 0.0f, 0.0f), // kd
    glm::vec3(1.0f, 0.0f, 0.0f), // ks
    glm::vec3(1.0f, 1.0f, 1.0f), // kr
    30.0f                        // alpha
);

/// Cyan
const BasicMaterial cyan(
    glm::vec3(0.0f, 1.0f, 1.0f), // ka
    glm::vec3(0.0f, 1.0f, 1.0f), // kd
    glm::vec3(1.0f, 1.0f, 1.0f), // ks
    glm::vec3(0.5f, 0.5f, 0.5f), // kr
    30.0f                        // alpha
);

/// Rouge fonce
const BasicMaterial darkRed(
    glm::vec3(0.5f, 0.0f, 0.0f), // ka
    glm::vec3(0.5f, 0.0f, 0.0f), // kd
    glm::vec3(0.2f, 0.2f, 0.2f), // ks
    glm::vec3(0.2f, 0.2f, 0.2f), // kr
    30.0f                        // alpha
);

/// Joli vert reflechissant
const BasicMaterial prettyGreen(
    glm::vec3(0.16f, 0.83f, 0.18f), // ka
    glm::vec3(0.16f, 0.83f, 0.18f), // kd
    glm::vec3(0.16f, 0.83f, 0.18f), // ks
    glm::vec3(0.4f, 0.4f, 0.4f), // kr
    30.0f                        // alpha
);

/// Couleur doree metallique
const BasicMaterial metallicGold(
    glm::vec3(0.83f, 0.69f, 0.22f),
    glm::vec3(0.83f, 0.69f, 0.22f),
    glm::vec3(0.83f, 0.69f, 0.22f),
    glm::vec3(0.8f, 0.8f, 0.8f),
    10.0f
);
}
}
}