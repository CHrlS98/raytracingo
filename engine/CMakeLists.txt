#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Ajouter a la main un dossier a la liste des include directories pour
# chaque dossier ajoute dans l'arbre systeme
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(fichiers_source ${fichiers_source})

# Recupere tous les fichiers d'entete .h du projet
file(GLOB_RECURSE fichiers_h
  "*.h"
)

foreach(fichier ${fichiers_h})
  list(APPEND fichiers_source ${fichier})
endforeach()

# Recupere tous les fichiers source .cpp du projet
file(GLOB_RECURSE fichiers_cpp
"*.cpp"
)

foreach(fichier ${fichiers_cpp})
  list(APPEND fichiers_source ${fichier})
endforeach()

# Recupere tous les fichiers CUDA .cu du projet
file(GLOB_RECURSE fichiers_cuda
  "*.cu"
)

foreach(fichier ${fichiers_cuda})
   list(APPEND fichiers_source ${fichier})
endforeach()

message("Fichiers trouves: ")
foreach(fichier ${fichiers_source})
   message(${fichier})
endforeach()

OPTIX_add_sample_executable(engine target_name
  ${fichiers_source}
  OPTIONS -rdc true
)

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)
