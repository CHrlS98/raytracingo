//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix_function_table_definition.h>

#include <renderer.h>
#include <scene.h>

#include <iomanip>
#include <iostream>
#include <string>

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "         --help | -h                           Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>                Set image dimensions; defaults to 512x384\n";
    std::cerr << "         --mode=distributed OR path            Set render mode to distributed ray tracing or path tracing\n";
    std::cerr << "         --scene=cornell OR plateau OR prison  Select a scene to render\n";
    std::cerr << "         --sample=<sample>                     Set the number of sample*sample of rays lunched per pixel; default is 1\n";
    exit( 1 );
}

int main( int argc, char* argv[] )
{
    int width  = 600;
    int height = 600;
    int sample = 1;
    bool argModeFound = false;
    bool argSceneFound = false;
    engine::host::RenderMode renderMode = engine::host::RenderMode::PATH_TRACING;
    engine::host::SceneModel sceneModel = engine::host::SceneModel::CORNELL;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit(argv[0]);
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else if (arg.substr(0, 7) == "--mode=")
        {
            const std::string mode_arg = arg.substr(7);
            argModeFound = true;
            if (mode_arg == "distributed")
            {
                renderMode = engine::host::RenderMode::DISTRIBUTED_RAY_TRACING;
            }
            else if (mode_arg == "path")
            {
                renderMode = engine::host::RenderMode::PATH_TRACING;
            }
            else
            {
                std::cerr << "Unknown option '" << arg << "'\n";
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 8) == "--scene=")
        {
            const std::string mode_arg = arg.substr(8);
            argSceneFound = true;
            if (mode_arg == "plateau")
            {
                sceneModel = engine::host::SceneModel::PLATE;
            }
            else if (mode_arg == "cornell")
            {
                sceneModel = engine::host::SceneModel::CORNELL;
            }
            else if (mode_arg == "slide")
            {
                sceneModel = engine::host::SceneModel::SLIDE;
            }
            else
            {
                std::cerr << "Unknown option '" << arg << "'\n";
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 9) == "--sample=")
        {
            const std::string sample_arg = arg.substr(9);
            sample = atoi(sample_arg.c_str());
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if (!argModeFound)
    {
        std::cerr << "Argument manquant: --mode=" << std::endl;
        printUsageAndExit(argv[0]);
    }
    if (!argSceneFound)
    {
        std::cerr << "Argument manquant: --scene=" << std::endl;
        printUsageAndExit(argv[0]);
    }

    try
    {
        auto scene = std::make_shared<engine::host::Scene>(sceneModel, width, height);
        engine::host::Renderer renderer = engine::host::Renderer(scene, renderMode, sample);
        renderer.Display();
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
