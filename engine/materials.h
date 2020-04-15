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


// Materiaux de la scene Mirror Spheres

/// Mirroir
const BasicMaterial mirrorSpheresBlackMirror(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    100000.0f
);

/// Plancher
const BasicMaterial mirrorSpheresGroundMat(
    glm::vec3(0.9765f, 0.651f, 0.6549f),
    glm::vec3(0.7f),
    glm::vec3(0.0f),
    100000.0f
);

/// Orange metallique
const BasicMaterial mirrorSpheresMetallicOrange(
    glm::vec3(0.8549f, 0.4078f, 0.0588f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    10000.0f
);

/// Silver
const BasicMaterial mirrorSpheresSilver(
    glm::vec3(0.7529f, 0.7529f, 0.7529f),
    glm::vec3(0.5f),
    glm::vec3(0.0f),
    10000.0f
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

// Materiaux pour la scene Soft Mirrors

/// Mirroir 0
const BasicMaterial softMirrorsMirror0(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    500000.0f
);

/// Mirroir 1
const BasicMaterial softMirrorsMirror1(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    100000.0f
);

/// Mirroir 2
const BasicMaterial softMirrorsMirror2(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    50000.0f
);

/// Mirroir 3
const BasicMaterial softMirrorsMirror3(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    10000.0f
);

/// Mirroir 4
const BasicMaterial softMirrorsMirror4(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    5000.0f
);

/// Mirroir 5
const BasicMaterial softMirrorsMirror5(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    1000.0f
);

/// Mirroir 6
const BasicMaterial softMirrorsMirror6(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    500.0f
);

/// Mirroir 7
const BasicMaterial softMirrorsMirror7(
    glm::vec3(0.05f, 0.05f, 0.05f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    100.0f
);

// Diferents materiaux

/// Lumiere eclairant le damier
const BasicMaterial CheckeredLight(
    glm::vec3(0.0f),
    glm::vec3(0.0f),
    glm::vec3(12.0f),
    1.0f
);

/// Lumiere eclairant l'horde de spheres
const BasicMaterial BallsLight(
    glm::vec3(0.0f),
    glm::vec3(0.0f),
    glm::vec3(12.0f),
    1.0f
);

/// Lumiere simulant l'eclairage d'une fenetre
const BasicMaterial WindowLight(
    glm::vec3(0.0f),
    glm::vec3(0.0f),
    glm::vec3(12.0f),
    1.0f
);

const BasicMaterial windowWhite(
    glm::vec3(0.9f, 0.9f, 0.9f),
    glm::vec3(0.5f, 0.5f, 0.5f),
    glm::vec3(0.0f),
    100.0f
);

} // namespace materials
} // namespace host
} // namespace engine