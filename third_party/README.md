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

The default library has no third-party runtime dependency. Import/export
uses nlohmann/json (MIT); corpus/research tooling also uses libcurl
(curl license), OpenSSL (Apache-2.0), and libarchive (BSD).
The Nix lock records the package collection used for those dependencies.

Downloaded assets have independent licenses and provenance in
research/corpus.json. They are not covered by the source-code license.
cgltf_write.h is retained at the same cgltf pin; export currently writes
explicit accessors through nlohmann/json.
