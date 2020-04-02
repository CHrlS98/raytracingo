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

/// Couleur bleu moyennement interessante
const BasicMaterial blue(
    glm::vec3(0.0f, 0.549f, 0.988f), // ka
    glm::vec3(0.0f, 0.549f, 0.988f), // kd
    glm::vec3(0.9f, 0.98f, 0.988f),  // ks
    glm::vec3(0.0f, 0.0f, 0.0f),     // kr
    5.0f                             // alpha
);

/// Couleur gris
const BasicMaterial grey(
    glm::vec3(0.29f, 0.29f, 0.29f), // ka
    glm::vec3(0.29f, 0.29f, 0.29f), // kd
    glm::vec3(1.0f, 1.0f, 1.0f),    // ks
    glm::vec3(0.0f, 0.0f, 0.0f),    // kr
    500.0f                          // alpha
);

/// Couleur blanc
const BasicMaterial white(
    glm::vec3(1.0f, 1.0f, 1.0f), // ka
    glm::vec3(1.0f, 1.0f, 1.0f), // kd
    glm::vec3(0.3f, 0.3f, 0.3f), // ks
    glm::vec3(0.0f, 0.0f, 0.0f), // kr
    50.0f                        // alpha
);

/// Couleur creme
const BasicMaterial cream(
    glm::vec3(1.0f, 0.941f, 0.729f), // ka
    glm::vec3(1.0f, 0.941f, 0.729f), // kd
    glm::vec3(1.0f, 0.941f, 0.729f), // ks
    glm::vec3(0.0f, 0.0f, 0.0f),     // kr
    7.0f                             // alpha
);

} // namespace materials
} // namespace host
} // namespace engine