# Acquisition and selection decisions

24 public CC0 candidate packages were downloaded/inspected. Nine supply the
40 selected models; their exact file inventories are frozen in manifest.json.
The discovery list in sources.json includes rejected candidates so failures
and source limitations remain visible. Candidates were researched through
primary author/provider pages, public provider APIs and Prophet-assisted lookup,
then checked against downloaded bytes. Search claims alone did not qualify an asset.

## Selected sources and duplicate review

Poly Haven Fern 02 contributes four authored clumps. Shrub 02/03 contribute
four each; Grass Medium 02 contributes five; Pine Sapling Small contributes
three complete saplings. Heliophila contributes three flower clumps and Weed
Plant 02 contributes three of its five authored variants. These are distinct
provider mesh objects, not alternate LOD exports. Their source families stay
together for any future dataset split.

Loaf's Foliage_1 archive contains overlapping `Trees.gltf` and
`Generic_Foliage.gltf` exports. Only the latter supplies selections. Three trees
are selected as complete author-named node subtrees, including trunk, twig and
leaf children. Individual tree parts are never counted as separate trees.
The six selected leafy-branch clusters are standalone author-named assets.
`LeafStick3_5` and `LeafStick3_6` have identical triangle-position fingerprints;
only one is selected. Grass 2 was left out because its three-card layout is too
close to a resized/rotated Grass 1. Grass 1, 3 and 4 instead provide three-card,
single-card and seven-card arrangements (6, 2 and 14 triangles). Grass 5 was
excluded after visual review showed a reed/cattail despite its source name.
The bare LeafStick3_1 was also replaced by a visibly leafy two-card branch.

Liefz Lily's fern pack contains 32 OBJ variants with an RGBA atlas but no MTL.
Variants 01 and 17 have different authored frond/growth arrangements (48 and
150 triangles); paired mirror/rotation variants are excluded. The missing MTL
is disclosed, and the color/alpha mapping is a sidecar rather than a fabricated
original material. The author page explicitly identifies CC0; the archive also
contains the author's permissive LICENSE.txt.

All selected models have different canonical triangle-position hashes. This
check removes translation, winding and index ordering but is not rotation/
scale invariant; topology counts, original identities and source previews
were reviewed in addition. No claim of universal semantic deduplication is made.

## Unselected candidate packages

| Candidate | Decision |
|---|---|
| [rubberduck Free Plant Pack](https://opengameart.org/content/free-plant-pack) | Blender source + textures; no OBJ/glTF interchange model in the archive. No conversion executed. |
| [rubberduck Free Vegetation](https://opengameart.org/content/free-vegetation-asset-pack) | Blender source; left for a separately specified conversion workflow. |
| [Grass Pack 01](https://opengameart.org/content/grass-pack-01) | OBJ/FBX + TGA material maps; usable candidate, but quota filled from sources with directly inspectable PNG/JPEG bindings. Recolored texture variants would not count separately. |
| [Grass Pack 02](https://opengameart.org/content/grass-pack-02) | No OBJ/glTF model found in the inspected archive. |
| [Grass Pack 03](https://opengameart.org/content/grass-pack-03) | OBJ/FBX + TGA maps; retained as an unselected candidate. |
| [Bushes](https://opengameart.org/content/bushes-0) | The discovered archive contains a Python generator, not the requested static model bundle. It was not executed. |
| [Tropical shrubs](https://opengameart.org/content/tropical-shrubs) | Five OBJ shapes found; older TGA/material hookups need manual review. Shrub quota is supplied by verified Poly Haven clumps. |
| [Tiny weeds](https://opengameart.org/content/tiny-weeds), [2](https://opengameart.org/content/tiny-weeds-2), [3](https://opengameart.org/content/tiny-weeds-3) | Available OBJ candidates with older texture packaging; not needed to fill this revision. |
| [Free 3D Plants Models](https://opengameart.org/content/free-3d-plants-models) | Mesh-only archive; required texture assets are absent. Cannot qualify as a complete foliage bundle. |
| [Lowpoly reed](https://opengameart.org/content/lowpoly-reed) | 380-triangle model; card/opacity evidence insufficient for inclusion. No dedicated reed is claimed in this revision. |
| [Fir Tree 01](https://polyhaven.com/a/fir_tree_01) | Two variants exceed the two-million-triangle cap; remaining variant lacks the required UVs. |
| [Fir Sapling](https://polyhaven.com/a/fir_sapling) | Existing source identity in the frozen 120-mesh corpus; rejected by the overlap check. |
| [Pine Sapling Medium](https://polyhaven.com/a/pine_sapling_medium) | One variant exceeds the cap; the other two are unusually heavy. The smaller pine saplings and new broadleaf assemblies provide the six selected trees. |

Other online fern leads failed before acquisition: Yughues Free Fern Plant is
CC-BY, not CC0; several fern scans are dense reconstructed surfaces rather than
card foliage. Neither was used to pad the six-fern quota.
