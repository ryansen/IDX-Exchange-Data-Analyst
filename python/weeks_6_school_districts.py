"""Download public CDE polygons once; join MLS locations locally with Shapely.

The 2024–25 boundary vintage is applied as a fixed reference across the sales
period. It is district geography, not a school attendance or enrollment promise.
Elementary/high overlaps are expected. Same-type overlaps remain ambiguous.
No property coordinates are sent to the public service.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely import STRtree, contains, make_valid, points, prepare, union_all
from shapely.geometry import shape

SOURCE_PAGE = "https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172"
LAYER = "https://services3.arcgis.com/fdvHcZVgB2QSRNkL/arcgis/rest/services/SchoolDistrictAreas2425/FeatureServer/0"
CACHE = Path(__file__).resolve().parent.parent / "csv/reference/school_district_areas_2024_25.geojson"
DISTRICT_TYPES = {"Elementary": "ElementarySchoolDistrict", "High": "HighSchoolDistrict", "Unified": "UnifiedSchoolDistrict"}


def download_boundaries(cache=CACHE, refresh=False):
    """Fetch the official layer as WGS84 GeoJSON; check paging/counts and record provenance."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        metadata_path = cache.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if hashlib.sha256(cache.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("School boundary cache checksum changed; explicitly refresh the download")
        return cache
    session = requests.Session()
    def query(params):
        response = session.get(LAYER + "/query", params=params, timeout=(30, 240))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"School boundary API error: {payload['error']}")
        return payload
    expected = query({"f": "json", "where": "1=1", "returnCountOnly": "true"})["count"]
    features, offset = [], 0
    while len(features) < expected:
        page = query({"f": "geojson", "where": "1=1", "outSR": "4326",
                      "outFields": "OBJECTID,Year,CDCode,CountyName,DistrictName,DistrictType,GradeLow,GradeHigh",
                      "returnGeometry": "true", "orderByFields": "OBJECTID",
                      "resultOffset": offset, "resultRecordCount": 500})
        rows = page.get("features", [])
        if not rows:
            raise RuntimeError("School district download ended before the expected feature count")
        features.extend(rows)
        offset += len(rows)
        print(f"School polygons downloaded: {len(features):,}/{expected:,}", flush=True)
    ids = [f["properties"]["OBJECTID"] for f in features]
    if len(features) != expected or len(set(ids)) != expected:
        raise ValueError("Duplicate or incomplete school polygons in the download")
    collection = {"type": "FeatureCollection", "features": features}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".geojson.part")
    temporary.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    metadata = {"source_page": SOURCE_PAGE, "arcgis_layer": LAYER, "boundary_year": "2024-25",
                "crs": "EPSG:4326", "feature_count": expected, "sha256": digest,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "method": "Download public polygons; all point-in-polygon matching is local"}
    cache.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cache


def load_polygons(path=CACHE):
    collection = json.loads(Path(path).read_text(encoding="utf-8"))
    geometries, records, repaired = [], [], 0
    for feature in collection["features"]:
        record = feature["properties"]
        geometry = shape(feature["geometry"])
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            repaired += 1
        if geometry.geom_type == "GeometryCollection":
            geometry = union_all([part for part in geometry.geoms if part.geom_type in ["Polygon", "MultiPolygon"]])
        if geometry.is_empty or geometry.geom_type not in ["Polygon", "MultiPolygon"]:
            raise ValueError(f"District lacks polygon geometry: {record['CDCode']}")
        xmin, ymin, xmax, ymax = geometry.bounds
        if not (-180 <= xmin <= xmax <= 180 and -90 <= ymin <= ymax <= 90):
            raise ValueError("District coordinates are not WGS84 longitude/latitude")
        if record["Year"] != "2024-25" or record["DistrictType"] not in DISTRICT_TYPES:
            raise ValueError(f"Unexpected district vintage/type: {record}")
        geometries.append(geometry)
        records.append({"DistrictCode": str(record["CDCode"]).zfill(7), "DistrictName": record["DistrictName"],
                        "DistrictType": record["DistrictType"], "DistrictDataYear": record["Year"]})
    districts = pd.DataFrame(records)
    if not districts.DistrictCode.str.fullmatch(r"\d{7}").all() or not districts.DistrictCode.is_unique:
        raise ValueError("District codes must be unique seven-digit text identifiers")
    return geometries, districts, repaired


def match_school_districts(data, geometries, districts):
    """Return a one-row-per-record enrichment plus a separate membership bridge."""
    if not data.SourceRecordNumber.is_unique:
        raise ValueError("SourceRecordNumber must uniquely identify each MLS record")
    data = data.reset_index(drop=True)
    eligible = data.Latitude.notna() & data.Longitude.notna()
    eligible &= data.Latitude.between(-90, 90) & data.Longitude.between(-180, 180)
    if "map_analysis_eligible_flag" in data:
        eligible &= data.map_analysis_eligible_flag.fillna(False)
    row_positions = np.flatnonzero(eligible.to_numpy(dtype=bool))
    coordinates = points(data.loc[eligible, "Longitude"].to_numpy(dtype=float),
                         data.loc[eligible, "Latitude"].to_numpy(dtype=float))
    polygon_array = np.asarray(geometries, dtype=object)
    prepare(polygon_array)
    # Query prepared polygons against indexed points for a fast statewide join.
    # covers() includes exact boundary points; contains() distinguishes interiors.
    pairs = STRtree(coordinates).query(polygon_array, predicate="covers")
    columns = ["SourceRecordNumber", "DistrictCode", "DistrictName", "DistrictType", "DistrictDataYear", "OnBoundary"]
    if pairs.shape[1]:
        polygon_ids, point_ids = pairs
        membership = districts.iloc[polygon_ids].reset_index(drop=True).copy()
        membership.insert(0, "SourceRecordNumber", data.SourceRecordNumber.iloc[row_positions[point_ids]].to_numpy())
        membership["OnBoundary"] = ~contains(polygon_array[polygon_ids], coordinates[point_ids])
        # Multiple polygon pieces for the same district do not duplicate membership.
        membership = membership.sort_values("OnBoundary").drop_duplicates(["SourceRecordNumber", "DistrictCode"], keep="last")
    else:
        membership = pd.DataFrame(columns=columns)

    enrichment = data[["SourceRecordNumber"]].copy().set_index("SourceRecordNumber")
    for prefix in DISTRICT_TYPES.values():
        enrichment[prefix] = pd.Series(pd.NA, index=enrichment.index, dtype="string")
        enrichment[prefix + "Code"] = pd.Series(pd.NA, index=enrichment.index, dtype="string")
    ambiguous = pd.Series(False, index=enrichment.index)
    for kind, prefix in DISTRICT_TYPES.items():
        group = membership[membership.DistrictType.eq(kind)]
        counts = group.groupby("SourceRecordNumber").DistrictCode.nunique()
        ambiguous |= counts.reindex(enrichment.index, fill_value=0).gt(1)
        unique = group[group.SourceRecordNumber.map(counts).eq(1)].set_index("SourceRecordNumber")
        enrichment.loc[unique.index, prefix] = unique.DistrictName.astype("string")
        enrichment.loc[unique.index, prefix + "Code"] = unique.DistrictCode.astype("string")
    combined_structures = enrichment.UnifiedSchoolDistrict.notna() & (
        enrichment.ElementarySchoolDistrict.notna() | enrichment.HighSchoolDistrict.notna())
    ambiguous |= combined_structures
    enrichment["SchoolDistrictAmbiguousFlag"] = ambiguous
    enrichment["SchoolDistrictBoundaryFlag"] = membership.groupby("SourceRecordNumber").OnBoundary.any().reindex(enrichment.index, fill_value=False).astype(bool)
    enrichment["SchoolDistrictMatchCount"] = membership.groupby("SourceRecordNumber").DistrictCode.nunique().reindex(enrichment.index, fill_value=0).astype("Int64")
    labels = membership.sort_values(["DistrictType", "DistrictCode"]).groupby("SourceRecordNumber").DistrictName.agg(lambda x: " | ".join(x))
    enrichment["SchoolDistrictNames"] = labels.reindex(enrichment.index).astype("string")
    enrichment["SchoolDistrictDataYear"] = "2024-25"
    status = pd.Series("unmatched", index=enrichment.index, dtype="string")
    status.loc[~eligible.to_numpy(dtype=bool)] = "excluded_geographic_quality"
    status.loc[enrichment.SchoolDistrictMatchCount.gt(0)] = "matched"
    status.loc[enrichment.SchoolDistrictBoundaryFlag] = "matched_boundary"
    status.loc[ambiguous] = "ambiguous"
    enrichment["SchoolDistrictMatchStatus"] = status
    enrichment["SchoolDistrictUnmatchedFlag"] = status.eq("unmatched")
    return enrichment.reset_index(), membership[columns].reset_index(drop=True)


if __name__ == "__main__":
    print(download_boundaries())
