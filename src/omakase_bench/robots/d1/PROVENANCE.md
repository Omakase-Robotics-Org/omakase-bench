# D1 preset provenance

| File | Source | Commit | Licence |
|---|---|---|---|
| `d1.urdf` | manipulation-kit `src/manipulation_kit/description/generated/d1/d1.urdf` | d5b5166465168431897e9d7549d15a611834e3a1 | Apache-2.0 |
| `d1.meshes.urdf` | manipulation-kit, composed from the same preset with mesh visuals (the meshes: `assets.yaml`) | d5b5166465168431897e9d7549d15a611834e3a1 | Apache-2.0 |
| `spec.json` | manipulation-kit `src/manipulation_kit/description/generated/d1/spec.json` | d5b5166465168431897e9d7549d15a611834e3a1 | Apache-2.0 |
| `robot.yaml` | this repository | | Apache-2.0 |

manipulation-kit is the publisher's motion library; its repository is not public, so the paths and
the commit identify the source for maintainers.

`d1.urdf` is the mesh-free composition: every link's geometry is a box,
cylinder or sphere, so the primitive visual variant needs no other file.
To refresh it, copy the files from a newer manipulation-kit commit and
update the table.
