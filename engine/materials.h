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
    glm::vec3(1.0f, 1.0f, 0.0f), // kd
    1.0f                         // roughness
);

/// Violet
const BasicMaterial purple(
    glm::vec3(1.0f, 0.0f, 1.0f), // kd
    0.005f                         // roughness
);

/// Miroir incolore
const BasicMaterial blackMirror(
    glm::vec3(0.1f, 0.1f, 0.1f), // kd
    0.0f                         // roughness
);

/// Cyan
const BasicMaterial cyan(
    glm::vec3(0.0f, 1.0f, 1.0f), // kd
    0.05f                         // roughness
);

/// Rouge fonce
const BasicMaterial darkRed(
    glm::vec3(0.5f, 0.0f, 0.0f), // kd
    1.0f                         // roughness
);

/// Joli vert
const BasicMaterial prettyGreen(
    glm::vec3(0.16f, 0.83f, 0.18f), // kd
    1.0f                            // roughness
);

/// Couleur doree metallique
const BasicMaterial metallicGold(
    glm::vec3(0.83f, 0.69f, 0.22f), // kd
    0.0f                            // roughness
);

/// Couleur bleu moyennement interessante
const BasicMaterial blue(
    glm::vec3(0.0f, 0.549f, 0.988f), // kd
    1.0f                             // roughness
);

/// Couleur gris
const BasicMaterial grey(
    glm::vec3(0.29f, 0.29f, 0.29f), // kd
    0.0f                            // roughness
);

/// Couleur blanc
const BasicMaterial white(
    glm::vec3(0.9f, 0.9f, 0.9f), // kd
    1.0f                         // roughness
);

/// Couleur creme
const BasicMaterial cream(
    glm::vec3(1.0f, 0.941f, 0.729f), // kd
    1.0f                             // roughness
);

/// Bleu de la Cornell box
const BasicMaterial cornellBlue(
    glm::vec3(0.0f, 0.0f, 1.0f), // kd
    1.0f                         // roughness
);

/// Rouge de la Cornell box
const BasicMaterial cornellRed(
    glm::vec3(1.0f, 0.0f, 0.0f), // kd
    1.0f                         // roughness
);

} // namespace materials
} // namespace host
} // namespace engine