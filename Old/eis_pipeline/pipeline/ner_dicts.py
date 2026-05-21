"""
Dictionary-based NER for high-value entity categories that we can name
deterministically:

  - Federally recognized tribes (seeded from the BIA published list)
  - Environmental NGOs (curated for the 1970s–90s EIS era)

These complement spaCy NER. Dictionary matches bypass downstream Haiku triage
because their identity as stakeholders is already established by membership in
the list.

Both lists are seeds — extend as you encounter entities the pipeline misses.
Match is case-insensitive, word-boundary anchored, and respects multi-word
phrases (so "Nature Conservancy" won't match "natured to conserve").
"""

from __future__ import annotations

import re
from typing import NamedTuple


class DictMatch(NamedTuple):
    name: str          # canonical name as it should appear in output
    category: str      # "tribe" | "ngo"
    first_offset: int  # first character offset where it was found


# ---------------------------------------------------------------------------
# Federally recognized tribes
# Source: Bureau of Indian Affairs, "Indian Entities Recognized by and Eligible
# to Receive Services from the United States Bureau of Indian Affairs"
# (Federal Register annual notice). This is a curated subset of the full ~574
# entries focused on tribes most likely to appear in EIS documents from the
# 1970s–90s (land use, water rights, mining, and infrastructure disputes).
# Expand as needed — the full list is public and stable year-to-year.
# ---------------------------------------------------------------------------
TRIBES: list[str] = [
    # Sioux / Lakota / Dakota
    "Oglala Sioux Tribe",
    "Rosebud Sioux Tribe",
    "Standing Rock Sioux Tribe",
    "Cheyenne River Sioux Tribe",
    "Crow Creek Sioux Tribe",
    "Lower Brule Sioux Tribe",
    "Yankton Sioux Tribe",
    "Sisseton-Wahpeton Oyate",
    "Spirit Lake Tribe",
    "Flandreau Santee Sioux Tribe",

    # Navajo / Hopi / Zuni / Pueblo
    "Navajo Nation",
    "Hopi Tribe",
    "Zuni Tribe",
    "Pueblo of Acoma",
    "Pueblo of Laguna",
    "Pueblo of Zia",
    "Pueblo of Jemez",
    "Pueblo of Santa Ana",
    "Pueblo of Santo Domingo",
    "Pueblo of San Felipe",
    "Pueblo of Cochiti",
    "Pueblo of Sandia",
    "Pueblo of Isleta",
    "Pueblo of Taos",
    "Pueblo of Picuris",
    "Pueblo of Santa Clara",
    "Pueblo of San Ildefonso",
    "Pueblo of Pojoaque",
    "Pueblo of Nambe",
    "Pueblo of Tesuque",
    "Ohkay Owingeh",

    # Apache
    "Jicarilla Apache Nation",
    "Mescalero Apache Tribe",
    "White Mountain Apache Tribe",
    "San Carlos Apache Tribe",
    "Tonto Apache Tribe",
    "Yavapai-Apache Nation",
    "Fort Sill Apache Tribe",

    # Cherokee / Choctaw / Chickasaw / Creek / Seminole (Five Tribes)
    "Cherokee Nation",
    "Eastern Band of Cherokee Indians",
    "United Keetoowah Band of Cherokee Indians",
    "Choctaw Nation of Oklahoma",
    "Mississippi Band of Choctaw Indians",
    "Chickasaw Nation",
    "Muscogee (Creek) Nation",
    "Seminole Nation of Oklahoma",
    "Seminole Tribe of Florida",
    "Miccosukee Tribe of Indians of Florida",

    # Northern Plains
    "Blackfeet Nation",
    "Crow Tribe",
    "Northern Cheyenne Tribe",
    "Northern Arapaho Tribe",
    "Eastern Shoshone Tribe",
    "Fort Belknap Indian Community",
    "Fort Peck Assiniboine and Sioux Tribes",
    "Chippewa Cree Tribe",
    "Little Shell Tribe of Chippewa Indians",
    "Confederated Salish and Kootenai Tribes",
    "Mandan, Hidatsa, and Arikara Nation",
    "Turtle Mountain Band of Chippewa Indians",

    # Pacific Northwest
    "Confederated Tribes of the Warm Springs Reservation",
    "Confederated Tribes of the Umatilla Indian Reservation",
    "Confederated Tribes of Siletz Indians",
    "Confederated Tribes of the Grand Ronde Community",
    "Confederated Tribes of the Colville Reservation",
    "Confederated Tribes and Bands of the Yakama Nation",
    "Yakama Nation",
    "Spokane Tribe of Indians",
    "Coeur d'Alene Tribe",
    "Nez Perce Tribe",
    "Shoshone-Bannock Tribes",
    "Kalispel Indian Community",
    "Quileute Tribe",
    "Quinault Indian Nation",
    "Makah Tribe",
    "Hoh Indian Tribe",
    "Lummi Nation",
    "Nooksack Indian Tribe",
    "Swinomish Indian Tribal Community",
    "Tulalip Tribes",
    "Stillaguamish Tribe of Indians",
    "Suquamish Tribe",
    "Skokomish Indian Tribe",
    "Squaxin Island Tribe",
    "Puyallup Tribe of Indians",
    "Muckleshoot Indian Tribe",
    "Nisqually Indian Tribe",
    "Cowlitz Indian Tribe",
    "Chehalis Indian Tribe",
    "Confederated Tribes of the Chehalis Reservation",
    "Coquille Indian Tribe",
    "Klamath Tribes",
    "Cow Creek Band of Umpqua Tribe of Indians",
    "Burns Paiute Tribe",

    # California
    "Yurok Tribe",
    "Hoopa Valley Tribe",
    "Karuk Tribe",
    "Tule River Indian Tribe",
    "Pala Band of Mission Indians",
    "Pechanga Band of Luiseno Mission Indians",
    "Cabazon Band of Mission Indians",
    "Morongo Band of Mission Indians",
    "Agua Caliente Band of Cahuilla Indians",
    "Soboba Band of Luiseno Indians",
    "Sycuan Band of the Kumeyaay Nation",
    "Viejas Band of Kumeyaay Indians",

    # Great Basin / Southwest
    "Ute Mountain Ute Tribe",
    "Southern Ute Indian Tribe",
    "Ute Indian Tribe of the Uintah and Ouray Reservation",
    "Paiute Indian Tribe of Utah",
    "San Juan Southern Paiute Tribe",
    "Pyramid Lake Paiute Tribe",
    "Walker River Paiute Tribe",
    "Fallon Paiute-Shoshone Tribe",
    "Te-Moak Tribe of Western Shoshone Indians",
    "Duckwater Shoshone Tribe",
    "Yomba Shoshone Tribe",
    "Las Vegas Tribe of Paiute Indians",
    "Moapa Band of Paiutes",
    "Havasupai Tribe",
    "Hualapai Tribe",
    "Kaibab Band of Paiute Indians",
    "Colorado River Indian Tribes",
    "Cocopah Indian Tribe",
    "Quechan Tribe",
    "Tohono O'odham Nation",
    "Gila River Indian Community",
    "Salt River Pima-Maricopa Indian Community",
    "Ak-Chin Indian Community",
    "Pascua Yaqui Tribe",

    # Midwest / Great Lakes
    "Red Lake Band of Chippewa Indians",
    "White Earth Band of Ojibwe",
    "Leech Lake Band of Ojibwe",
    "Fond du Lac Band of Lake Superior Chippewa",
    "Bois Forte Band of Chippewa",
    "Mille Lacs Band of Ojibwe",
    "Grand Portage Band of Chippewa",
    "Bad River Band of Lake Superior Chippewa",
    "Lac du Flambeau Band of Lake Superior Chippewa",
    "Lac Courte Oreilles Band of Lake Superior Chippewa",
    "Red Cliff Band of Lake Superior Chippewa",
    "St. Croix Chippewa Indians of Wisconsin",
    "Sokaogon Chippewa Community",
    "Forest County Potawatomi Community",
    "Hannahville Indian Community",
    "Keweenaw Bay Indian Community",
    "Lac Vieux Desert Band of Lake Superior Chippewa",
    "Little Traverse Bay Bands of Odawa Indians",
    "Little River Band of Ottawa Indians",
    "Saginaw Chippewa Indian Tribe",
    "Sault Ste. Marie Tribe of Chippewa Indians",
    "Pokagon Band of Potawatomi Indians",
    "Match-E-Be-Nash-She-Wish Band of Pottawatomi",
    "Nottawaseppi Huron Band of the Potawatomi",
    "Ho-Chunk Nation",
    "Oneida Nation",
    "Menominee Indian Tribe of Wisconsin",
    "Stockbridge-Munsee Community Band of Mohican Indians",
    "Forest County Potawatomi",

    # Northeast
    "Penobscot Indian Nation",
    "Passamaquoddy Tribe",
    "Houlton Band of Maliseet Indians",
    "Aroostook Band of Micmacs",
    "Mashantucket Pequot Tribe",
    "Mohegan Tribe",
    "Narragansett Indian Tribe",
    "Wampanoag Tribe of Gay Head (Aquinnah)",
    "Mashpee Wampanoag Tribe",
    "Saint Regis Mohawk Tribe",
    "Cayuga Nation",
    "Oneida Indian Nation",
    "Onondaga Nation",
    "Seneca Nation of Indians",
    "Tonawanda Band of Seneca",
    "Tuscarora Nation",
    "Shinnecock Indian Nation",

    # Alaska Native (a small sample — there are hundreds of villages)
    "Native Village of Barrow",
    "Native Village of Kotzebue",
    "Sitka Tribe of Alaska",
    "Central Council of the Tlingit and Haida Indian Tribes",
    "Metlakatla Indian Community",
    "Yupiit of Andreafski",
    "Kenaitze Indian Tribe",

    # Misc commonly-referenced
    "Iowa Tribe of Kansas and Nebraska",
    "Iowa Tribe of Oklahoma",
    "Kickapoo Tribe of Oklahoma",
    "Kickapoo Tribe in Kansas",
    "Otoe-Missouria Tribe of Indians",
    "Sac and Fox Nation",
    "Pawnee Nation of Oklahoma",
    "Ponca Tribe of Oklahoma",
    "Ponca Tribe of Nebraska",
    "Osage Nation",
    "Omaha Tribe of Nebraska",
    "Santee Sioux Nation",
    "Winnebago Tribe of Nebraska",
    "Kiowa Indian Tribe of Oklahoma",
    "Comanche Nation",
    "Wichita and Affiliated Tribes",
    "Caddo Nation of Oklahoma",
    "Delaware Nation",
    "Delaware Tribe of Indians",
    "Shawnee Tribe",
    "Eastern Shawnee Tribe of Oklahoma",
    "Absentee-Shawnee Tribe of Indians of Oklahoma",
    "Wyandotte Nation",
    "Modoc Nation",
    "Ottawa Tribe of Oklahoma",
    "Peoria Tribe of Indians of Oklahoma",
    "Quapaw Nation",
    "Tunica-Biloxi Tribe of Louisiana",
    "Chitimacha Tribe of Louisiana",
    "Coushatta Tribe of Louisiana",
    "Jena Band of Choctaw Indians",
    "Alabama-Coushatta Tribe of Texas",
    "Ysleta del Sur Pueblo",
    "Kickapoo Traditional Tribe of Texas",
    "Catawba Indian Nation",
    "Lumbee Tribe of North Carolina",
    "Poarch Band of Creek Indians",
]


# ---------------------------------------------------------------------------
# Environmental NGOs
# Focused on organizations that were active and frequently named in EIS
# documents during the 1970s–90s. Extend with regional groups as encountered.
# ---------------------------------------------------------------------------
ENV_NGOS: list[str] = [
    # Major national orgs
    "Sierra Club",
    "Natural Resources Defense Council",
    "NRDC",
    "Environmental Defense Fund",
    "Environmental Defense",
    "National Audubon Society",
    "Audubon Society",
    "National Wildlife Federation",
    "The Wilderness Society",
    "The Nature Conservancy",
    "Nature Conservancy",
    "Defenders of Wildlife",
    "Friends of the Earth",
    "Greenpeace",
    "Earthjustice",
    "Sierra Club Legal Defense Fund",
    "World Wildlife Fund",
    "Conservation International",
    "League of Conservation Voters",
    "Izaak Walton League of America",
    "Izaak Walton League",
    "Center for Biological Diversity",
    "Public Citizen",
    "Earth First!",
    "Earth Island Institute",
    "Wilderness Watch",
    "Western Watersheds Project",
    "Ocean Conservancy",
    "American Rivers",
    "Trust for Public Land",
    "Conservation Fund",
    "Trout Unlimited",
    "Ducks Unlimited",
    "Pheasants Forever",
    "National Parks Conservation Association",
    "Wildlife Conservation Society",

    # Indigenous / Tribal advocacy
    "Native American Rights Fund",
    "NARF",
    "American Indian Movement",

    # Regional and topical
    "Rocky Mountain Institute",
    "Conservation Law Foundation",
    "Southern Environmental Law Center",
    "Western Resource Advocates",
    "Save Our Wild Salmon",
    "Pacific Coast Federation of Fishermen's Associations",
    "Center for Marine Conservation",
    "Oceana",
    "Surfrider Foundation",
    "Waterkeeper Alliance",
    "Riverkeeper",
    "Chesapeake Bay Foundation",
    "Save the Bay",

    # Air / climate
    "Clean Air Task Force",
    "Climate Action Network",
    "Environmental Working Group",

    # Older / historical
    "Save-the-Redwoods League",
    "Save the Redwoods League",
    "Cousteau Society",
    "Natural Resources Council of America",
]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _build_pattern(names: list[str]) -> re.Pattern[str]:
    """
    Compile a single alternation regex from a list of names.
    Longer names listed first so "Confederated Tribes of the Yakama Nation"
    is preferred over "Yakama Nation".
    """
    sorted_names = sorted(set(names), key=len, reverse=True)
    escaped = [re.escape(n) for n in sorted_names]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_TRIBES_PATTERN = _build_pattern(TRIBES)
_NGOS_PATTERN = _build_pattern(ENV_NGOS)

_TRIBES_CANONICAL: dict[str, str] = {n.lower(): n for n in TRIBES}
_NGOS_CANONICAL: dict[str, str] = {n.lower(): n for n in ENV_NGOS}


def find_tribes(text: str) -> list[DictMatch]:
    """Find federally recognized tribes mentioned in text. Returns first-appearance order."""
    return _find(text, _TRIBES_PATTERN, _TRIBES_CANONICAL, category="tribe")


def find_ngos(text: str) -> list[DictMatch]:
    """Find known environmental NGOs mentioned in text. Returns first-appearance order."""
    return _find(text, _NGOS_PATTERN, _NGOS_CANONICAL, category="ngo")


def _find(
    text: str,
    pattern: re.Pattern[str],
    canonical: dict[str, str],
    category: str,
) -> list[DictMatch]:
    """Scan text, return DictMatch list in first-appearance order, deduplicated by canonical name."""
    seen: dict[str, DictMatch] = {}
    for m in pattern.finditer(text):
        matched = m.group(0)
        canon = canonical.get(matched.lower(), matched)
        if canon not in seen:
            seen[canon] = DictMatch(name=canon, category=category, first_offset=m.start())
    return sorted(seen.values(), key=lambda d: d.first_offset)
