# README
RayTracinGO est un moteur de rendu en temps réel basé sur le tracé de rayons. Il supporte les modes de _distributed ray tracing_ et de _path tracing_.

## Configuration
* OptiX SDK 7.0.0
* CUDA Toolkit 10.2

## Système de build
* CMake 3.16.3
* MSVC v141 (Visual Studio 2017)
* Configuration x64 requise

## Utilisation
Les arguments suivants sont obligatoires:
* --scene=[scene]: afficher la scène [scene] parmis _plateau_, _slide_, _cornell_, _mirror_spheres_, _soft_mirrors_, _window_, _balls_ et _checkered_.
* --mode=[mode]: spécifier le mode à utiliser, soit _distributed_ ou _path_.

Les arguments suivants sont optionnels:
* --sample=[nb samples] spécifier le nombre d'échantillons [nb samples]*[nb samples] (par défaut: 1)
* --useAmbient: spécifier si on utilise l'éclairage ambiant (par défaut: non)
* --help: afficher les commandes disponibles.