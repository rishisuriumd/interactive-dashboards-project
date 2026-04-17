import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import nilearn
    from nilearn import datasets
    from ipyniivue import NiiVue
    from pathlib import Path
    import nibabel as nib
    from scipy.spatial import KDTree
    import numpy as np
    from create_atlas_giis import build_highlight_gii
    import matplotlib.pyplot as plt


    return KDTree, NiiVue, Path, build_highlight_gii, datasets, mo, nib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive Dashboard
    """)
    return


@app.cell
def _(datasets):
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    return (fsaverage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot Glasser Atlas
    """)
    return


@app.cell
def _(mo):
    hemi = mo.ui.radio(
        options=["Left", "Right"],
        value = "Left",
        label="Hemisphere"
    )
    hemi
    return (hemi,)


@app.cell
def _(mo):
    get_label, set_label = mo.state("Hover over brain...")
    get_region, set_region = mo.state(None)
    return get_label, get_region, set_label, set_region


@app.cell
def _(KDTree, fsaverage, nib):
    lh_coords, _ = nib.load(fsaverage["infl_left"]).agg_data()
    rh_coords, _ = nib.load(fsaverage["infl_right"]).agg_data()

    lh_tree = KDTree(lh_coords)
    rh_tree = KDTree(rh_coords)


    lh_labels, _, lh_names = nib.freesurfer.read_annot("data/atlases/lh.HCPMMP1.annot")
    rh_labels, _, rh_names = nib.freesurfer.read_annot("data/atlases/rh.HCPMMP1.annot")

    hemisphere = {
        "Left":  {
            "tree":   lh_tree,
            "labels": lh_labels,
            "names":  ["Background"] + [n.decode() if isinstance(n, bytes) else n for n in lh_names[1:]],
        },
        "Right": {
            "tree":   rh_tree,
            "labels": rh_labels,
            "names":  ["Background"] + [n.decode() if isinstance(n, bytes) else n for n in rh_names[1:]],
        },
    }
    return (hemisphere,)


@app.cell
def _(mo):
    mode = mo.ui.switch(value=False, label="Show isolated region only")
    mode
    return (mode,)


@app.cell
def _():
    return


@app.cell
def _(
    NiiVue,
    Path,
    build_highlight_gii,
    fsaverage,
    get_region,
    hemi,
    hemisphere,
    mo,
    mode,
    set_label,
    set_region,
):
    region = get_region()
    surf_key = "infl_left" if hemi.value == "Left" else "infl_right"
    h = hemisphere[hemi.value]
    atlas_side = "lh" if hemi.value == "Left" else "rh"

    if mode.value and region is not None and region["hemi"] == hemi.value:
        parcel_id = region["parcel_id"]
        tmp_path = Path(f"data/atlases/_highlight_{atlas_side}_{parcel_id}.label.gii")
        build_highlight_gii(h["labels"], parcel_id, region["name"], tmp_path)
        mesh_entry = {
            "path": Path(fsaverage[surf_key]),
            "rgba255": [160, 160, 160, 255],
            "layers": [{"path": tmp_path}],
        }
        header = f"### Isolated: **{region['name']}** ({region['hemi']})"
    else:
        mesh_entry = {
            "path": Path(fsaverage[surf_key]),
            "layers": [{"path": Path(f"data/atlases/{atlas_side}.HCPMMP1.label.gii")}],
        }
        header = "### Full atlas"

    nv = NiiVue()
    nv.load_meshes([mesh_entry])


    @nv.on_location_change
    def show_location(location):
        coords_str = location["string"]
        try:
            x, y, z = [float(v) for v in coords_str.split("×")]
            hh = hemisphere[hemi.value]
            dist, idx = hh["tree"].query([x, y, z])
            parcel_id = int(hh["labels"][idx])
            region_name = hh["names"][parcel_id]

            set_label(f"{hemi.value} | {region_name} | vertex: {idx} | {coords_str}")
            set_region({"hemi": hemi.value, "parcel_id": parcel_id, "name": region_name})
        except Exception as e:
            set_label(f"Error: {e}")

    mo.vstack([mo.md(header), mo.ui.anywidget(nv)])

    return


@app.cell
def _(get_label, mo):

    mo.md(get_label())
    return


if __name__ == "__main__":
    app.run()
