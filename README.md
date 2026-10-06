- get_L_N.py tells you what you can send as inputs
- main_maze_md.py is the main file to run with inputs according to those templates in the above file
- run ./compile_c.sh to compile the C functions

## Compile using CMake

Needs cmake version 3.20 or higher to be installed.
Starting from the root directory of the project, run the following commands:

```bash

cd library

rm -rf build  # Remove the build directory if it exists

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=../maze_poisson  # Install the library next to the python files
cmake --build build  # Build the library
cmake --install build  # Install the library

```

compile in mode:
- `-DCMAKE_BUILD_TYPE=Release` for optimized code
- `-DCMAKE_BUILD_TYPE=Debug` for debug mode (without optimizations)
- `-DCMAKE_BUILD_TYPE=RelWithDebInfo` for optimized code with debug info

if from macOS:

```bash

cd library

rm -rf build

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=../maze_poisson \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang

cmake --build build -j8
cmake --install build
```

## Fixed particles (frozen probe)

An optional `fixed` column (0/1) in the `input_file` freezes a particle: the integrators (OVRVO, Verlet) never
move it and keep its velocity at zero, while its forces are still computed. The temperature counts only the mobile
particles, the restart file keeps the flag, and `rescale` must be `false`. Rebuild the C library after updating.
