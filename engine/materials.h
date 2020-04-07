#pragma once
#include <basicmaterial.h>
#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
namespace materials
{
/// Miroir incolore
const BasicMaterial blackMirror(
    glm::vec3(0.1f, 0.1f, 0.1f), // kd
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    0.0f                         // roughness
);

/// Couleur bleu moyennement interessante
const BasicMaterial blue(
    glm::vec3(0.0f, 0.549f, 0.988f), // kd
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                             // roughness
);

/// Couleur gris
const BasicMaterial grey(
    glm::vec3(0.29f, 0.29f, 0.29f), // kd
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                            // roughness
);

/// Couleur creme
const BasicMaterial cream(
    glm::vec3(1.0f, 0.941f, 0.729f), // kd
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                             // roughness
);

/// Couleur blanc
const BasicMaterial white(
    glm::vec3(0.8f, 0.8f, 0.8f), // kd
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                         // roughness
);

// Materiaux du plateau de primitives

/// Couleur doree metallique
const BasicMaterial plateMetallicGold(
    glm::vec3(0.83f, 0.69f, 0.22f), // kd
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    10000.0f
);

/// Violet
const BasicMaterial platePurple(
    glm::vec3(1.0f, 0.0f, 1.0f), // kd
    glm::vec3(0.5f, 0.5f, 0.5f), // kr
    glm::vec3(0.0f, 0.0f, 0.0f), // Le
    1000.0f // specularite
);

/// Cyan
const BasicMaterial plateCyan(
    glm::vec3(0.1f, 1.0f, 1.0f), // kd
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    5000.0f
);

/// Joli vert
const BasicMaterial platePrettyGreen(
    glm::vec3(0.16f, 0.83f, 0.18f), // kd
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1000.0f
);

/// Rouge fonce
const BasicMaterial plateDarkRed(
    glm::vec3(0.5f, 0.0f, 0.0f), // kd
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    100.0f
);

/// Jaune
const BasicMaterial plateYellow(
    glm::vec3(1.0f, 1.0f, 0.0f), // kd
    glm::vec3(0.7f, 0.7f, 0.7f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    0.0f
);

/// Lumiere eclairant le plateau
const BasicMaterial plateLight(
    glm::vec3(0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(15.f),
    0.0f
);

// Materiaux de la Cornell Box

/// Miroir noir reflechissant
const BasicMaterial cornellMirror(
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    0.0f
);

/// Couleur blanc
const BasicMaterial cornellWhite(
    glm::vec3(0.8f, 0.8f, 0.8f), // kd
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                         // roughness
);

/// Bleu de la Cornell box
const BasicMaterial cornellBlue(
    glm::vec3(0.0f, 0.0f, 1.0f), // kd
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                         // roughness
);

/// Rouge de la Cornell box
const BasicMaterial cornellRed(
    glm::vec3(1.0f, 0.0f, 0.0f), // kd
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f                         // roughness
);

const BasicMaterial cornellLight(
    glm::vec3(0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(15.f),
    1.0f
);

} // namespace materials
} // namespace host
} // namespace engine