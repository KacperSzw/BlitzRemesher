# Third-party provenance

| Component | Pin | License | Role |
|---|---|---|---|
| cgltf | 85cd62382dfea638278962690cf515023f33ed00 | MIT (in header) | glTF/GLB import |
| tinyobjloader | 45636bdcef1a4fec140346b90c0b50bf0bc3e23b | MIT (in header) | OBJ import |
| meshoptimizer | 9e1f07b159d3cb777f1c67ed31fc11fd117986f4 (1.3) | MIT | optional separate benchmark |
| Fast Quadric | 65df07dc54766e3ee480482f1c881a62767831cc | MIT | optional separate benchmark |
| CGAL | 6.1.1 in pinned Nix environment | package-specific GPL/LGPL | optional separate benchmark |

cmake/Baselines.cmake downloads the two Git-pinned research dependencies.
They are not linked into the shipped library. tools/baseline_cgal.cpp is
GPL-3.0-or-later because it uses CGAL's GPL Surface_mesh_simplification
package. Distribution of that executable must satisfy its license; the
separate executable is not a license exemption.

Dependency boundaries are selected independently:

| Build switch | Dependencies and scope |
|---|---|
| Portable runtime, all optional backends off | No third-party runtime dependency; C++ standard library only |
| `BLITZ_TOOLS` | Local CLI/import/export tools; cgltf, tinyobjloader, nlohmann/json (MIT) and OpenSSL (Apache-2.0) for artifact hashes |
| `BLITZ_ACQUISITION` | Corpus HTTP/archive tooling; libcurl (curl license), libarchive (BSD), local I/O and OpenSSL; off by default |
| `BLITZ_CUDA` | Neural inference/auditing; CUDA runtime, cuBLAS and OpenSSL; no LibTorch |
| `BLITZ_VULKAN` | CUDA-interoperable hardware audits; requires `BLITZ_CUDA`, Vulkan, NPP and build-time glslangValidator |
| `BLITZ_NEURAL_TOOLS` | CUDA preparation/profiling tools using local I/O; enabled by default with CUDA and general tools, independently disableable |
| `BLITZ_NEURAL_TRAIN` | Training executables and their private training library; CUDA, cuBLAS/cuBLASLt and LibTorch; requires `BLITZ_CUDA` |
| `BLITZ_RESEARCH` | Separate pinned external baseline executables; off by default |

For a portable library-only build, disable `BLITZ_TOOLS` and `BUILD_TESTING`;
CUDA, Vulkan, training, acquisition and external research are already opt-in.
Local mesh tools do not require HTTP or archive support. LibTorch and the training
library never enter the installed runtime targets. The Nix lock records the
package collection used for system dependencies.

Training pins the official LibTorch **2.10.0 cu128 shared-with-deps** distribution
(PyTorch BSD-style license). Its SHA-256 is
`429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9`, recorded in
[`research/neural/libtorch.sha256`](../research/neural/libtorch.sha256).
`BLITZ_LIBTORCH_ROOT` selects that extracted distribution. The pinned development
toolkit is CUDA 12.9; LibTorch carries its CUDA 12.8 libraries. CUDA components
retain their own licenses. Model inference reads the native `.blzn` format and
does not load Torch serialization or execute serialized model code.

Downloaded assets have independent licenses and provenance in
research/corpus.json. They are not covered by the source-code license.
cgltf_write.h is retained at the same cgltf pin; export currently writes
explicit accessors through nlohmann/json.
