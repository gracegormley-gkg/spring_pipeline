"""
Download and build the geocoder reference data into spring_pipeline/geodata/.

    python scripts/fetch_geodata.py          # from May25/; about 2 GB download, 2 minutes

Produces the two files spring_pipeline/.env points at:

    geodata/DomesticNames_National.txt   USGS GNIS domestic names (pipe-delimited)
    geodata/padus41_federal.gpkg         PAD-US 4.1 combined layer, federal units only

PAD-US comes from the public source.coop GeoParquet mirror, because USGS's own
ScienceBase download needs a login. Only units whose manager passes the
pipeline's own `_is_federal_manager` are kept -- the loader discards everything
else anyway -- so the result is identical to pointing it at the full database,
at a tenth of the size. The 1.9 GB source file is deleted afterwards.

Needs pyarrow for the conversion only (`pip install pyarrow`).
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

MAY25 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAY25))

from mcal import settings  # noqa: E402

GEODATA = settings.PROJECT_ROOT / "geodata"
GNIS_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/"
    "DomesticNames/DomesticNames_National_Text.zip"
)
PADUS_URL = "https://data.source.coop/cboettig/padus/padus-4-1/combined.parquet"
GNIS_OUT = GEODATA / "DomesticNames_National.txt"
PADUS_OUT = GEODATA / "padus41_federal.gpkg"
PADUS_LAYER = "PADUS4_1Combined_Federal"   # must contain "combined" (loader preference)


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.0f} MB")


def fetch_gnis() -> None:
    if GNIS_OUT.exists():
        print(f"{GNIS_OUT.name} already present")
        return
    zpath = GEODATA / "DomesticNames_National_Text.zip"
    _download(GNIS_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.endswith("DomesticNames_National.txt"))
        GNIS_OUT.write_bytes(z.read(member))
    zpath.unlink()


def fetch_padus() -> None:
    if PADUS_OUT.exists():
        print(f"{PADUS_OUT.name} already present")
        return
    import geopandas as gpd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from segment_b.postproc.location_pipeline import _is_federal_manager

    src = GEODATA / "padus41_combined.parquet"
    if not src.exists():
        _download(PADUS_URL, src)
    names = pc.unique(pq.read_table(src, columns=["Mang_Name"])["Mang_Name"]).to_pylist()
    federal = sorted(n for n in names if n and _is_federal_manager(n))
    print(f"federal managers kept: {federal}")
    gdf = gpd.read_parquet(
        src,
        columns=["Unit_Nm", "Loc_Nm", "Mang_Name", "Mang_Type", "Des_Tp", "State_Nm", "SHAPE"],
        filters=[("Mang_Name", "in", federal)],
    )
    tmp = PADUS_OUT.with_suffix(".part.gpkg")
    gdf.to_file(tmp, layer=PADUS_LAYER, driver="GPKG")
    tmp.replace(PADUS_OUT)
    src.unlink()
    print(f"  {PADUS_OUT.name}: {len(gdf)} units, {PADUS_OUT.stat().st_size / 1e6:.0f} MB")


def main() -> int:
    GEODATA.mkdir(exist_ok=True)
    fetch_gnis()
    fetch_padus()
    print(f"geocoder stack: {settings.geocoder_precheck()['stack']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
