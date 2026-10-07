#!/usr/bin/env python3
"""
NHC Tropical Weather API Builder
=================================
Fetches data from:
  - NHC GIS RSS feeds (per-storm track, cone, wind field shapefiles)
  - NHC GTWO KMZ files (areas of interest / development areas)
  - xWeather/AerisWeather Tropical Cyclones API (storm position, track, forecast)

Outputs:
  docs/tropical_api/atlantic_nhc.geojson
  docs/tropical_api/epac_nhc.geojson
  docs/tropical_api/cpac_nhc.geojson
"""

import os
import sys
import json
import zipfile
import tempfile
import logging
import re
import math
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
import shapefile  # pyshp
from zipfile import ZipFile

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("nhc_builder")

# ── Constants ─────────────────────────────────────────────────────────────────
XWEATHER_BASE = "https://data.api.xweather.com"
NHC_RSS = {
    "atlantic": "https://www.nhc.noaa.gov/gis-at.xml",
    "epac":     "https://www.nhc.noaa.gov/gis-ep.xml",
    "cpac":     "https://www.nhc.noaa.gov/gis-cp.xml",
}
# GTWO KMZ live URLs (latest outlook)
GTWO_KMZ = {
    "atlantic": "https://www.nhc.noaa.gov/xgtwo/gtwo_atl.kmz",
    "epac":     "https://www.nhc.noaa.gov/xgtwo/gtwo_pac.kmz",
    "cpac":     "https://www.nhc.noaa.gov/xgtwo/gtwo_cpac.kmz",
}
# xWeather basin codes → our basin keys
XWEATHER_BASIN_MAP = {
    "AL": "atlantic",
    "EP": "epac",
    "CP": "cpac",
}

# Shapefile layer names inside the 5-day forecast ZIP
# NHC uses: al042026_5day_pgn  (cone polygon)
#           al042026_5day_pts  (forecast track points)
#           al042026_5day_lin  (forecast track line)
SHP_CONE_SUFFIX = "_5day_pgn"
SHP_TRACK_PTS_SUFFIX = "_5day_pts"
SHP_TRACK_LIN_SUFFIX = "_5day_lin"
# Wind field ZIP layer names (34kt, 50kt, 64kt radii)
SHP_WIND_SUFFIX_PATTERN = re.compile(r"_initialradii|_forecastradii|_fcst_\d+", re.I)

# NHC XML namespace
NHC_NS = "https://www.nhc.noaa.gov"

HEADERS = {
    "User-Agent": "NHC-GeoJSON-Builder/1.0 (github-actions; public data aggregation)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch(url: str, timeout: int = 30) -> Optional[bytes]:
    """Fetch a URL; return bytes or None on error."""
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        log.warning("  FETCH FAILED %s — %s", url, exc)
        return None


def kmz_to_kml_bytes(kmz_bytes: bytes) -> Optional[bytes]:
    """Extract the first .kml file from a KMZ (zipped KML)."""
    try:
        with ZipFile(BytesIO(kmz_bytes)) as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                return None
            return z.read(kml_names[0])
    except Exception as exc:
        log.warning("  KMZ extraction failed: %s", exc)
        return None


def kml_placemarks_to_features(kml_bytes: bytes) -> list[dict]:
    """
    Parse KML placemarks into GeoJSON Feature dicts.
    Handles Point, LineString, Polygon, MultiGeometry.
    Flattens MultiGeometry into individual features sharing the same properties.
    """
    features = []
    try:
        root = ET.fromstring(kml_bytes)
        ns = {"kml": "http://www.opengis.net/kml/2.2"}

        def parse_coords(text: str) -> list:
            """Parse KML coordinate string → [[lon, lat], ...]"""
            coords = []
            for token in text.strip().split():
                parts = token.split(",")
                if len(parts) >= 2:
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        coords.append([lon, lat])
                    except ValueError:
                        pass
            return coords

        def geom_from_element(el) -> Optional[dict]:
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag == "Point":
                c = el.find("kml:coordinates", ns)
                if c is None:
                    c = el.find("coordinates")
                if c is not None and c.text:
                    pts = parse_coords(c.text)
                    if pts:
                        return {"type": "Point", "coordinates": pts[0]}
            elif tag == "LineString":
                c = el.find("kml:coordinates", ns)
                if c is None:
                    c = el.find("coordinates")
                if c is not None and c.text:
                    pts = parse_coords(c.text)
                    if pts:
                        return {"type": "LineString", "coordinates": pts}
            elif tag == "Polygon":
                # Try with namespace prefix first, then bare tag, then full namespace URI
                outer = (
                    el.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
                    or el.find(".//outerBoundaryIs/LinearRing/coordinates")
                )
                # Fallback: walk all descendants looking for a coordinates element
                # inside an outerBoundaryIs (handles any namespace variation)
                if outer is None:
                    kml_ns = "http://www.opengis.net/kml/2.2"
                    for cands in [
                        f"{{{kml_ns}}}outerBoundaryIs",
                        "outerBoundaryIs",
                    ]:
                        obi = el.find(f".//{cands}")
                        if obi is not None:
                            for ctag in [
                                f"{{{kml_ns}}}coordinates",
                                "coordinates",
                            ]:
                                outer = obi.find(f".//{ctag}")
                                if outer is not None:
                                    break
                        if outer is not None:
                            break
                if outer is not None and outer.text:
                    ring = parse_coords(outer.text)
                    if ring:
                        # Close ring
                        if ring[0] != ring[-1]:
                            ring.append(ring[0])
                        return {"type": "Polygon", "coordinates": [ring]}
            return None

        def props_from_placemark(pm) -> dict:
            props = {}
            for child in pm:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag in ("name", "description", "styleUrl"):
                    props[tag.lower()] = (child.text or "").strip()
                elif tag == "ExtendedData":
                    for sd in child.findall(".//SimpleData"):
                        k = sd.get("name", "")
                        if k:
                            props[k] = (sd.text or "").strip()
            return props

        for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
            props = props_from_placemark(pm)
            geoms = []
            for child in pm:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "MultiGeometry":
                    for sub in child:
                        g = geom_from_element(sub)
                        if g:
                            geoms.append(g)
                elif tag in ("Point", "LineString", "Polygon"):
                    g = geom_from_element(child)
                    if g:
                        geoms.append(g)
            for g in geoms:
                features.append({"type": "Feature", "geometry": g, "properties": dict(props)})
    except Exception as exc:
        log.warning("  KML parse error: %s", exc)
    return features


def shapefile_zip_to_features(zip_bytes: bytes, layer_suffix: str) -> list[dict]:
    """
    Extract a specific shapefile layer from a ZIP and return GeoJSON features.
    layer_suffix: e.g. '_5day_pgn' — matched against shapefile basenames.
    """
    features = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(BytesIO(zip_bytes)) as z:
                z.extractall(tmpdir)
            # Find .shp files matching our layer suffix
            shp_files = list(Path(tmpdir).rglob("*.shp"))
            matched = [p for p in shp_files if layer_suffix.lower() in p.stem.lower()]
            if not matched:
                # Try prefix-agnostic match (some ZIPs strip the storm prefix)
                matched = shp_files  # fall back to all
            for shp_path in matched:
                try:
                    with shapefile.Reader(str(shp_path)) as sf:
                        fields = [f[0] for f in sf.fields[1:]]  # skip DeletionFlag
                        for sr in sf.shapeRecords():
                            geojson_geom = sr.shape.__geo_interface__
                            props = dict(zip(fields, sr.record))
                            # Convert bytes to str for JSON serialisation
                            props = {
                                k: (v.decode("utf-8", "replace").strip() if isinstance(v, bytes) else v)
                                for k, v in props.items()
                            }
                            features.append({
                                "type": "Feature",
                                "geometry": geojson_geom,
                                "properties": props,
                            })
                except Exception as exc:
                    log.warning("    Shapefile read error %s: %s", shp_path.name, exc)
    except Exception as exc:
        log.warning("  ZIP extract error: %s", exc)
    return features


def advisories_from_rss_item_title(title: str) -> Optional[str]:
    """
    Extract advisory number from titles like:
    'Advisory #023a Forecast [shp] - Hurricane Nolo (CP2/EP152026)'
    Returns '023A' or None.
    """
    m = re.search(r"Advisory\s*#?(\d+[A-Za-z]?)", title, re.I)
    if m:
        return m.group(1).upper()
    return None


def atcf_from_guid(guid: str) -> Optional[str]:
    """Extract ATCF id from RSS GUID like 'summary-ep152026-202609261147'."""
    parts = guid.split("-")
    for p in parts:
        if re.match(r"^(al|ep|cp)\d{6}$", p, re.I):
            return p.lower()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NHC RSS parser — active storms
# ─────────────────────────────────────────────────────────────────────────────

def parse_nhc_rss(basin: str, rss_bytes: bytes) -> list[dict]:
    """
    Parse NHC GIS RSS feed and return a list of storm dicts, each containing:
      - meta: name, type, atcf, wallet, center_lat, center_lon, wind, pressure,
              headline, movement, datetime_str
      - forecast_zip_url
      - wind_field_zip_url
      - best_track_zip_url
      - advisory_number
    """
    storms = {}  # keyed by atcf_id

    try:
        root = ET.fromstring(rss_bytes)
    except ET.ParseError as exc:
        log.warning("RSS XML parse error for %s: %s", basin, exc)
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    for item in channel.findall("item"):
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link_el = item.find("link")
        link = (link_el.text or "").strip() if link_el is not None else ""
        guid_el = item.find("guid")
        guid = (guid_el.text or "").strip() if guid_el is not None else ""

        # ── Summary item (has <nhc:Cyclone>) ─────────────────────────────────
        cyclone_el = item.find(f"{{{NHC_NS}}}Cyclone")
        if cyclone_el is not None:
            def nhc_text(tag: str) -> str:
                el = cyclone_el.find(f"{{{NHC_NS}}}{tag}")
                return (el.text or "").strip() if el is not None else ""

            center_raw = nhc_text("center")  # "lat, lon"
            center_parts = [p.strip() for p in center_raw.split(",")]
            try:
                center_lat = float(center_parts[0])
                center_lon = float(center_parts[1])
            except (ValueError, IndexError):
                center_lat = center_lon = None

            atcf_id = nhc_text("atcf").lower()
            if not atcf_id:
                atcf_id = atcf_from_guid(guid) or ""

            wallet = nhc_text("wallet")
            # Derive short storm_id like "04L", "15E", "02C"
            storm_id = derive_short_id(atcf_id, wallet)

            storms[atcf_id] = {
                "meta": {
                    "storm_id":        storm_id,
                    "atcf":            atcf_id,
                    "wallet":          wallet,
                    "name":            nhc_text("name"),
                    "type":            nhc_text("type"),
                    "center_lat":      center_lat,
                    "center_lon":      center_lon,
                    "wind_mph":        nhc_text("wind"),
                    "pressure_mb":     nhc_text("pressure"),
                    "movement":        nhc_text("movement"),
                    "headline":        nhc_text("headline"),
                    "datetime_str":    nhc_text("datetime"),
                    "advisory_number": None,
                },
                "forecast_zip_url":    None,
                "wind_field_zip_url":  None,
                "best_track_zip_url":  None,
            }
            continue

        # ── GIS product items — match to storm by ATCF id in URL/guid ────────
        # Determine ATCF from URL or guid
        atcf_id = None
        for candidate in [link, guid]:
            m = re.search(r"/(al|ep|cp)(\d{6})[_/]", candidate, re.I)
            if m:
                atcf_id = m.group(1).lower() + m.group(2)
                break

        if atcf_id and atcf_id not in storms:
            # Lazy-create a minimal stub if we didn't see the summary item yet
            storms[atcf_id] = {
                "meta": {
                    "storm_id": derive_short_id(atcf_id, ""),
                    "atcf": atcf_id,
                    "wallet": "",
                    "name": "",
                    "type": "",
                    "center_lat": None,
                    "center_lon": None,
                    "wind_mph": "",
                    "pressure_mb": "",
                    "movement": "",
                    "headline": "",
                    "datetime_str": "",
                    "advisory_number": None,
                },
                "forecast_zip_url": None,
                "wind_field_zip_url": None,
                "best_track_zip_url": None,
            }

        if not atcf_id:
            continue

        storm = storms[atcf_id]

        # Categorise by title keywords
        tl = title.lower()
        if "5day" in link.lower() or ("forecast" in tl and "[shp]" in tl):
            storm["forecast_zip_url"] = link
            adv = advisories_from_rss_item_title(title)
            if adv:
                storm["meta"]["advisory_number"] = adv
        elif "fcst" in link.lower() and link.endswith(".zip"):
            storm["wind_field_zip_url"] = link
        elif "best_track" in link.lower() and link.endswith(".zip"):
            storm["best_track_zip_url"] = link

    return list(storms.values())


def derive_short_id(atcf: str, wallet: str) -> str:
    """
    Derive human-readable storm ID from ATCF string.
    atcf = 'al042026'  → '04L'
    atcf = 'ep152026'  → '15E'
    atcf = 'cp022026'  → '02C'
    wallet = 'CP2'     → derive from wallet if atcf prefix is ep (cross-basin)
    """
    m = re.match(r"^(al|ep|cp)(\d{2})\d{4}$", atcf or "", re.I)
    if m:
        prefix = m.group(1).upper()
        num = m.group(2)
        suffix = {"AL": "L", "EP": "E", "CP": "C"}.get(prefix, "")
        # If wallet says it's really a CP storm (e.g. CP2) but ATCF is EP, use wallet
        if wallet and re.match(r"^CP\d+$", wallet, re.I):
            suffix = "C"
        return f"{num}{suffix}"
    # Fall back to wallet if available
    if wallet:
        wm = re.match(r"^(AL|EP|CP)(\d+)$", wallet, re.I)
        if wm:
            suffix = {"AL": "L", "EP": "E", "CP": "C"}.get(wm.group(1).upper(), "")
            return f"{int(wm.group(2)):02d}{suffix}"
    return atcf.upper() if atcf else "???"


# ─────────────────────────────────────────────────────────────────────────────
# xWeather API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_xweather_storms(client_id: str, client_secret: str) -> dict[str, list]:
    """
    Fetch all active tropical cyclones from xWeather and group by our basin keys.
    Returns {'atlantic': [...], 'epac': [...], 'cpac': [...]}
    """
    url = f"{XWEATHER_BASE}/tropicalcyclones/:all"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "filter": "active",
        "fields": ",".join([
            "id",
            "profile.name",
            "profile.year",
            "profile.isActive",
            "profile.stormType",
            "profile.maxStormCat",
            "profile.windSpeed",
            "profile.pressure",
            "position.loc",
            "position.details",
            "position.timestamp",
            "track",
            "forecast",
        ]),
        "limit": 50,
    }
    try:
        resp = SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("xWeather fetch failed: %s", exc)
        return {"atlantic": [], "epac": [], "cpac": []}

    by_basin: dict[str, list] = {"atlantic": [], "epac": [], "cpac": []}
    if not data.get("success"):
        log.warning("xWeather API error: %s", data.get("error", {}).get("description", "unknown"))
        return by_basin

    for item in (data.get("response") or []):
        prof = item.get("profile", {})
        pos = item.get("position", {})
        det = pos.get("details", {})

        # Derive basin from storm id or details
        storm_id_raw = item.get("id", "")  # e.g. "2026-AL-04"
        basin_code = ""
        m = re.search(r"\d{4}-([A-Z]{2})-\d+", storm_id_raw)
        if m:
            basin_code = m.group(1)
        if not basin_code:
            basin_code = (det.get("basin") or "").upper()

        basin_key = XWEATHER_BASIN_MAP.get(basin_code)
        if basin_key is None:
            continue

        # Build normalised storm record
        loc = pos.get("loc", {})
        record = {
            "source": "xweather",
            "xweather_id": storm_id_raw,
            "name": prof.get("name", ""),
            "storm_type": prof.get("stormType", det.get("stormType", "")),
            "max_cat": prof.get("maxStormCat", ""),
            "wind_speed_mph": (prof.get("windSpeed") or {}).get("maxMPH"),
            "pressure_mb": (prof.get("pressure") or {}).get("minMB"),
            "lat": loc.get("lat"),
            "lon": loc.get("long"),
            "timestamp": pos.get("timestamp"),
            "movement": det.get("movement", {}),
            "wind_radii_nm": det.get("windRadii", []),
            "track": _norm_track(item.get("track") or []),
            "forecast": _norm_forecast(item.get("forecast") or []),
        }
        by_basin[basin_key].append(record)

    return by_basin


def _norm_track(track: list) -> list:
    """Normalise xWeather track points to a consistent shape."""
    out = []
    for pt in track:
        loc = pt.get("loc", {})
        det = pt.get("details", {}) or {}
        out.append({
            "lat": loc.get("lat"),
            "lon": loc.get("long"),
            "timestamp": pt.get("timestamp"),
            "wind_mph": det.get("windSpeedMPH"),
            "pressure_mb": det.get("pressureMB"),
            "storm_type": det.get("stormType", ""),
        })
    return out


def _norm_forecast(forecast: list) -> list:
    """Normalise xWeather forecast points."""
    out = []
    for pt in (forecast or []):
        loc = pt.get("loc", {})
        det = pt.get("details", {}) or {}
        out.append({
            "lat": loc.get("lat"),
            "lon": loc.get("long"),
            "timestamp": pt.get("timestamp"),
            "storm_cat": det.get("stormCat", ""),
            "wind_mph": det.get("windSpeedMPH"),
            "wind_radii_nm": det.get("windRadii", []),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GTWO KMZ parser
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gtwo_features(basin: str) -> list[dict]:
    """
    Download the GTWO KMZ for a basin and return GeoJSON-like Feature dicts
    tagged with their 2-day / 7-day probability of development.
    """
    url = GTWO_KMZ.get(basin)
    if not url:
        return []

    log.info("  Fetching GTWO KMZ for %s …", basin)
    raw = fetch(url)
    if not raw:
        return []

    kml_bytes = kmz_to_kml_bytes(raw)
    if not kml_bytes:
        return []

    features = kml_placemarks_to_features(kml_bytes)
    # Tag all GTWO features with their product type
    for f in features:
        f["properties"]["product_type"] = "gtwo_outlook"
        f["properties"]["basin"] = basin
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Shapefile fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cone_features(storm: dict) -> list[dict]:
    """Download forecast ZIP and extract cone polygon features."""
    url = storm.get("forecast_zip_url")
    if not url:
        return []
    log.info("    Fetching cone/track shapefiles: %s", url)
    raw = fetch(url)
    if not raw:
        return []
    feats = shapefile_zip_to_features(raw, SHP_CONE_SUFFIX)
    for f in feats:
        f["properties"]["nhc_product"] = "cone_of_uncertainty"
        f["properties"]["atcf"] = storm["meta"]["atcf"]
        f["properties"]["storm_name"] = storm["meta"]["name"]
        f["properties"]["storm_id"] = storm["meta"]["storm_id"]
    return feats


def fetch_track_features(storm: dict) -> list[dict]:
    """Download forecast ZIP and extract track point features."""
    url = storm.get("forecast_zip_url")
    if not url:
        return []
    raw = fetch(url)
    if not raw:
        return []
    feats = shapefile_zip_to_features(raw, SHP_TRACK_PTS_SUFFIX)
    for f in feats:
        f["properties"]["nhc_product"] = "forecast_track_points"
        f["properties"]["atcf"] = storm["meta"]["atcf"]
        f["properties"]["storm_name"] = storm["meta"]["name"]
        f["properties"]["storm_id"] = storm["meta"]["storm_id"]
    return feats


def fetch_wind_radii_features(storm: dict) -> list[dict]:
    """Download wind-field ZIP and extract wind radii polygon features."""
    url = storm.get("wind_field_zip_url")
    if not url:
        return []
    log.info("    Fetching wind radii shapefiles: %s", url)
    raw = fetch(url)
    if not raw:
        return []
    # The wind-field ZIP contains multiple layers; grab all .shp files
    feats = shapefile_zip_to_features(raw, "")  # empty suffix → all layers
    for f in feats:
        f["properties"]["nhc_product"] = "wind_radii"
        f["properties"]["atcf"] = storm["meta"]["atcf"]
        f["properties"]["storm_name"] = storm["meta"]["name"]
        f["properties"]["storm_id"] = storm["meta"]["storm_id"]
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Wind speed probability — basin-wide KMZ
# ─────────────────────────────────────────────────────────────────────────────

def _wsp_kmz_url(basin: str, threshold_kt: int) -> Optional[str]:
    """
    Attempt to construct the current wind-speed probability KMZ URL.
    NHC publishes these at times aligned to synoptic hours (00, 06, 12, 18 UTC).
    We try the last four synoptic hours.
    """
    now = datetime.now(timezone.utc)
    # Map basin to NHC abbreviation used in the WSP filename
    basin_abbr = {"atlantic": "atl", "epac": "epac", "cpac": "cpac"}.get(basin, "atl")
    # NHC WSP KMZ naming: {YYYYMMDDHH}_wsp{threshold}knt120hr_5km.kmz
    for offset_hours in [0, 6, 12, 18]:
        hour = (now.hour // 6) * 6 - offset_hours
        dt = now.replace(hour=max(hour, 0), minute=0, second=0, microsecond=0)
        stamp = dt.strftime("%Y%m%d%H")
        url = (
            f"https://www.nhc.noaa.gov/gis/forecast/archive/"
            f"{stamp}_wsp{threshold_kt}knt120hr_5km.kmz"
        )
        # Quick HEAD check
        try:
            r = SESSION.head(url, timeout=10)
            if r.status_code == 200:
                return url
        except Exception:
            pass
    return None


def fetch_wsp_features(basin: str) -> list[dict]:
    """
    Fetch wind speed probability KMZ files (34, 50, 64 kt) for a basin.
    Returns GeoJSON features tagged with the probability threshold.
    """
    all_feats = []
    for kt in [34, 50, 64]:
        url = _wsp_kmz_url(basin, kt)
        if not url:
            log.info("    No WSP KMZ found for %s %dkt", basin, kt)
            continue
        log.info("    Fetching WSP KMZ %dkt for %s: %s", kt, basin, url)
        raw = fetch(url)
        if not raw:
            continue
        kml_bytes = kmz_to_kml_bytes(raw)
        if not kml_bytes:
            continue
        feats = kml_placemarks_to_features(kml_bytes)
        for f in feats:
            f["properties"]["nhc_product"] = "wind_speed_probability"
            f["properties"]["wind_threshold_kt"] = kt
            f["properties"]["basin"] = basin
        all_feats.extend(feats)
    return all_feats


# ─────────────────────────────────────────────────────────────────────────────
# Storm surge — per storm, issued when threatening coasts
# ─────────────────────────────────────────────────────────────────────────────

def fetch_surge_features(storm: dict) -> list[dict]:
    """
    Attempt to fetch storm surge shapefiles for a storm.
    NHC publishes these under:
      https://www.nhc.noaa.gov/gis/storm_surge/{atcf}_surge.zip
    These only exist when a surge advisory has been issued.
    """
    atcf = storm["meta"]["atcf"]
    url = f"https://www.nhc.noaa.gov/gis/storm_surge/{atcf}_surge.zip"
    log.info("    Checking surge ZIP for %s …", atcf)
    try:
        head = SESSION.head(url, timeout=10)
        if head.status_code != 200:
            return []
    except Exception:
        return []

    raw = fetch(url)
    if not raw:
        return []

    feats = shapefile_zip_to_features(raw, "")
    for f in feats:
        f["properties"]["nhc_product"] = "storm_surge"
        f["properties"]["atcf"] = atcf
        f["properties"]["storm_name"] = storm["meta"]["name"]
        f["properties"]["storm_id"] = storm["meta"]["storm_id"]
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Merge xWeather + NHC data into final Feature structure
# ─────────────────────────────────────────────────────────────────────────────

def build_storm_feature(nhc_storm: dict, xw_record: Optional[dict]) -> dict:
    """
    Build a single GeoJSON Feature for a storm's current position,
    merging NHC meta with xWeather data where available.
    """
    meta = nhc_storm["meta"]
    lat = meta.get("center_lat")
    lon = meta.get("center_lon")

    # Prefer xWeather coordinates if NHC meta is missing
    if (lat is None or lon is None) and xw_record:
        lat = xw_record.get("lat")
        lon = xw_record.get("lon")

    geom = {"type": "Point", "coordinates": [lon, lat]} if (lat is not None and lon is not None) else None

    props = {
        "nhc_product":      "storm_position",
        "storm_id":         meta.get("storm_id"),
        "atcf":             meta.get("atcf"),
        "wallet":           meta.get("wallet"),
        "name":             meta.get("name"),
        "type":             meta.get("type"),
        "wind_mph":         meta.get("wind_mph"),
        "pressure_mb":      meta.get("pressure_mb"),
        "movement":         meta.get("movement"),
        "headline":         meta.get("headline"),
        "datetime_str":     meta.get("datetime_str"),
        "advisory_number":  meta.get("advisory_number"),
    }

    # Enrich with xWeather data
    if xw_record:
        props["xweather_id"]    = xw_record.get("xweather_id")
        props["max_category"]   = xw_record.get("max_cat")
        props["wind_radii_nm"]  = xw_record.get("wind_radii_nm")
        props["movement_mph"]   = (xw_record.get("movement") or {}).get("speedMPH")
        props["movement_dir"]   = (xw_record.get("movement") or {}).get("directionDEG")

    return {"type": "Feature", "geometry": geom, "properties": props}


def build_track_feature(xw_record: dict) -> dict:
    """Build a LineString feature for the historical track from xWeather data."""
    track = xw_record.get("track") or []
    coords = [[pt["lon"], pt["lat"]] for pt in track if pt.get("lat") and pt.get("lon")]
    if not coords:
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "nhc_product": "historical_track",
            "atcf": None,
            "xweather_id": xw_record.get("xweather_id"),
            "name": xw_record.get("name"),
            "track_points": track,
        },
    }


def build_forecast_line_feature(xw_record: dict) -> dict:
    """Build a LineString feature for the forecast track from xWeather data."""
    forecast = xw_record.get("forecast") or []
    coords = [[pt["lon"], pt["lat"]] for pt in forecast if pt.get("lat") and pt.get("lon")]
    if not coords:
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "nhc_product": "forecast_track_line",
            "xweather_id": xw_record.get("xweather_id"),
            "name": xw_record.get("name"),
            "forecast_points": forecast,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Match NHC storms ↔ xWeather storms
# ─────────────────────────────────────────────────────────────────────────────

def match_xweather(nhc_storm: dict, xw_list: list[dict]) -> Optional[dict]:
    """
    Try to match a NHC storm to an xWeather record by name or rough position.
    """
    name = (nhc_storm["meta"].get("name") or "").lower().strip()
    lat = nhc_storm["meta"].get("center_lat")
    lon = nhc_storm["meta"].get("center_lon")

    for xw in xw_list:
        xw_name = (xw.get("name") or "").lower().strip()
        if name and xw_name and name == xw_name:
            return xw
        # Fallback: position proximity (~200 km)
        if lat and lon and xw.get("lat") and xw.get("lon"):
            dlat = abs(lat - xw["lat"])
            dlon = abs(lon - xw["lon"])
            dist = math.sqrt(dlat**2 + dlon**2)
            if dist < 2.0:  # ~200km
                return xw
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main basin builder
# ─────────────────────────────────────────────────────────────────────────────

def build_basin_geojson(
    basin: str,
    xweather_client_id: str,
    xweather_client_secret: str,
    xw_by_basin: dict,
) -> dict:
    """
    Build the complete GeoJSON FeatureCollection for one basin.
    """
    log.info("=" * 60)
    log.info("Building %s GeoJSON …", basin.upper())
    features = []

    # ── 1. GTWO development areas ─────────────────────────────────────────────
    gtwo_feats = fetch_gtwo_features(basin)
    log.info("  GTWO features: %d", len(gtwo_feats))
    features.extend(gtwo_feats)

    # ── 2. Wind Speed Probabilities (basin-wide) ──────────────────────────────
    wsp_feats = fetch_wsp_features(basin)
    log.info("  WSP features: %d", len(wsp_feats))
    features.extend(wsp_feats)

    # ── 3. Active storms (NHC RSS + xWeather) ────────────────────────────────
    rss_url = NHC_RSS[basin]
    log.info("  Fetching RSS: %s", rss_url)
    rss_bytes = fetch(rss_url)
    nhc_storms = []
    if rss_bytes:
        nhc_storms = parse_nhc_rss(basin, rss_bytes)
    log.info("  Active NHC storms: %d", len(nhc_storms))

    xw_list = xw_by_basin.get(basin, [])

    # Also handle xWeather-only storms (not yet in RSS but active)
    matched_xw_ids = set()

    for storm in nhc_storms:
        atcf = storm["meta"]["atcf"]
        log.info("  Processing storm: %s (%s)", storm["meta"]["name"], atcf)

        xw = match_xweather(storm, xw_list)
        if xw:
            matched_xw_ids.add(xw.get("xweather_id"))

        # Current position feature
        pos_feat = build_storm_feature(storm, xw)
        features.append(pos_feat)

        # Historical + forecast track from xWeather
        if xw:
            track_feat = build_track_feature(xw)
            if track_feat:
                track_feat["properties"]["atcf"] = atcf
                track_feat["properties"]["storm_id"] = storm["meta"]["storm_id"]
                features.append(track_feat)
            fcast_feat = build_forecast_line_feature(xw)
            if fcast_feat:
                fcast_feat["properties"]["atcf"] = atcf
                fcast_feat["properties"]["storm_id"] = storm["meta"]["storm_id"]
                features.append(fcast_feat)

        # Cone of uncertainty (NHC shapefile)
        cone_feats = fetch_cone_features(storm)
        log.info("    Cone features: %d", len(cone_feats))
        features.extend(cone_feats)

        # Forecast track points (NHC shapefile)
        track_pts_feats = fetch_track_features(storm)
        log.info("    Track point features: %d", len(track_pts_feats))
        features.extend(track_pts_feats)

        # Wind radii (NHC shapefile)
        wind_feats = fetch_wind_radii_features(storm)
        log.info("    Wind radii features: %d", len(wind_feats))
        features.extend(wind_feats)

        # Storm surge (only if advisory issued)
        surge_feats = fetch_surge_features(storm)
        log.info("    Storm surge features: %d", len(surge_feats))
        features.extend(surge_feats)

    # xWeather-only storms (appear in API but maybe not yet in RSS)
    for xw in xw_list:
        if xw.get("xweather_id") in matched_xw_ids:
            continue
        log.info("  xWeather-only storm: %s (%s)", xw.get("name"), xw.get("xweather_id"))
        stub = {
            "meta": {
                "storm_id": "",
                "atcf": "",
                "wallet": "",
                "name": xw.get("name", ""),
                "type": xw.get("storm_type", ""),
                "center_lat": xw.get("lat"),
                "center_lon": xw.get("lon"),
                "wind_mph": str(xw.get("wind_speed_mph", "")),
                "pressure_mb": str(xw.get("pressure_mb", "")),
                "movement": "",
                "headline": "",
                "datetime_str": "",
                "advisory_number": None,
            },
            "forecast_zip_url": None,
            "wind_field_zip_url": None,
            "best_track_zip_url": None,
        }
        pos_feat = build_storm_feature(stub, xw)
        features.append(pos_feat)
        track_feat = build_track_feature(xw)
        if track_feat:
            features.append(track_feat)
        fcast_feat = build_forecast_line_feature(xw)
        if fcast_feat:
            features.append(fcast_feat)

    # ── 4. Metadata feature ───────────────────────────────────────────────────
    features.append({
        "type": "Feature",
        "geometry": None,
        "properties": {
            "nhc_product":    "metadata",
            "basin":          basin,
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "active_storms":  len(nhc_storms),
            "gtwo_areas":     len(gtwo_feats),
        },
    })

    return {
        "type": "FeatureCollection",
        "features": [f for f in features if f is not None],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    client_id = os.environ.get("XWEATHER_CLIENT_ID", "")
    client_secret = os.environ.get("XWEATHER_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        log.error("XWEATHER_CLIENT_ID and XWEATHER_CLIENT_SECRET must be set.")
        sys.exit(1)

    # Fetch xWeather data once (all basins)
    log.info("Fetching xWeather tropical cyclone data …")
    xw_by_basin = fetch_xweather_storms(client_id, client_secret)
    log.info(
        "  xWeather storms — AT:%d  EP:%d  CP:%d",
        len(xw_by_basin["atlantic"]),
        len(xw_by_basin["epac"]),
        len(xw_by_basin["cpac"]),
    )

    # Output directory
    out_dir = Path("docs/tropical_api")
    out_dir.mkdir(parents=True, exist_ok=True)

    basin_files = {
        "atlantic": "atlantic_nhc.geojson",
        "epac":     "epac_nhc.geojson",
        "cpac":     "cpac_nhc.geojson",
    }

    for basin, filename in basin_files.items():
        geojson = build_basin_geojson(basin, client_id, client_secret, xw_by_basin)
        out_path = out_dir / filename
        out_path.write_text(json.dumps(geojson, indent=2, default=str), encoding="utf-8")
        log.info(
            "Wrote %s  (%d features)",
            out_path,
            len(geojson["features"]),
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
