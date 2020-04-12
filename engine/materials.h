#pragma once
#include <basicmaterial.h>
#include <glm/vec3.hpp>

namespace engine
{
namespace host
{
/// Definition de materiaux
namespace materials
{
/// Miroir incolore
const BasicMaterial blackMirror(
    glm::vec3(0.1f, 0.1f, 0.1f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    0.0f
);

/// Couleur bleu moyennement interessante
const BasicMaterial blue(
    glm::vec3(0.0f, 0.549f, 0.988f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
);

/// Couleur gris
const BasicMaterial grey(
    glm::vec3(0.29f, 0.29f, 0.29f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
);

/// Couleur creme
const BasicMaterial cream(
    glm::vec3(1.0f, 0.941f, 0.729f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
);

/// Couleur blanc
const BasicMaterial white(
    glm::vec3(0.8f, 0.8f, 0.8f),
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
);

// Materiaux du plateau de primitives

/// Couleur doree metallique
const BasicMaterial plateMetallicGold(
    glm::vec3(0.83f, 0.69f, 0.22f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    10000.0f
);

/// Violet
const BasicMaterial platePurple(
    glm::vec3(1.0f, 0.0f, 1.0f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1000.0f
);

/// Cyan
const BasicMaterial plateCyan(
    glm::vec3(0.1f, 1.0f, 1.0f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    5000.0f
);

/// Joli vert
const BasicMaterial platePrettyGreen(
    glm::vec3(0.16f, 0.83f, 0.18f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1000.0f
);

/// Rouge fonce
const BasicMaterial plateDarkRed(
    glm::vec3(0.5f, 0.0f, 0.0f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    100.0f
);

/// Jaune
const BasicMaterial plateYellow(
    glm::vec3(1.0f, 1.0f, 0.0f),
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
    glm::vec3(0.8f, 0.8f, 0.8f),
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
);

/// Bleu de la Cornell box
const BasicMaterial cornellBlue(
    glm::vec3(0.0f, 0.0f, 1.0f),
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
);

/// Rouge de la Cornell box
const BasicMaterial cornellRed(
    glm::vec3(1.0f, 0.0f, 0.0f),
    glm::vec3(0.3f, 0.3f, 0.3f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1.0f
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