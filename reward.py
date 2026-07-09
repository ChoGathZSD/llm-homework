"""Programmatic reward for detailed-prompt -> SVG-logo generation.

The reward is intentionally heuristic.  It measures whether an output is a
loadable, bounded, simple SVG logo that roughly follows visual words in the
prompt.  It is a training/evaluation proxy, not a substitute for human visual
judgement.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GRAPHIC_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}
SUPPORT_TAGS = {
    "svg",
    "defs",
    "g",
    "linearGradient",
    "radialGradient",
    "stop",
    "clipPath",
    "mask",
    "title",
    "desc",
}
BANNED_TAGS = {"script", "foreignObject", "image", "iframe", "audio", "video", "canvas"}
COLOR_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b|rgba?\([^)]+\)|\b[a-zA-Z]+\b")
HEX_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)


NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 38, 38),
    "orange": (245, 158, 11),
    "yellow": (234, 179, 8),
    "green": (34, 197, 94),
    "teal": (20, 184, 166),
    "cyan": (6, 182, 212),
    "blue": (37, 99, 235),
    "navy": (20, 40, 80),
    "purple": (147, 51, 234),
    "pink": (236, 72, 153),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "gold": (242, 169, 59),
    "golden": (242, 169, 59),
    "brown": (120, 72, 35),
}


COLOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "red": ("red", "crimson", "ruby", "scarlet"),
    "orange": ("orange", "amber", "copper"),
    "yellow": ("yellow", "gold", "golden"),
    "green": ("green", "emerald", "lime"),
    "teal": ("teal", "turquoise", "aqua"),
    "blue": ("blue", "navy", "azure", "sky"),
    "purple": ("purple", "violet", "indigo"),
    "pink": ("pink", "rose", "magenta"),
    "black": ("black", "dark", "charcoal"),
    "white": ("white", "ivory"),
    "gray": ("gray", "grey", "silver"),
    "brown": ("brown", "wood", "earth"),
}


SHAPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "circle": ("circle", "circular", "badge", "coin", "ring", "dot", "orb"),
    "rect": ("square", "rectangle", "block", "panel"),
    "line": ("line", "ray", "staff", "stripe", "tick"),
    "polygon": ("triangle", "diamond", "star", "hexagon", "shield"),
    "path": (
        "leaf",
        "curve",
        "wave",
        "swoosh",
        "flame",
        "mountain",
        "bird",
        "animal",
        "note",
        "music",
        "plant",
        "sprout",
    ),
}


@dataclass
class ParsedSvg:
    raw: str
    candidate: str
    root: ET.Element | None
    parse_error: str | None
    had_wrapping_text: bool


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_svg(text: str) -> tuple[str, bool]:
    stripped = (text or "").strip()
    match = SVG_RE.search(stripped)
    if match:
        candidate = match.group(0).strip()
        return candidate, candidate != stripped
    start = stripped.lower().find("<svg")
    if start >= 0:
        return stripped[start:].strip(), start != 0
    return stripped, False


def parse_svg(text: str) -> ParsedSvg:
    candidate, had_wrapping_text = extract_svg(text)
    try:
        root = ET.fromstring(candidate)
        return ParsedSvg(text, candidate, root, None, had_wrapping_text)
    except ET.ParseError as exc:
        return ParsedSvg(text, candidate, None, str(exc), had_wrapping_text)


def _iter_elements(root: ET.Element | None) -> Iterable[ET.Element]:
    if root is None:
        return []
    return root.iter()


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    value = value.strip()
    if not HEX_RE.fullmatch(value):
        return None
    digits = value[1:]
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_func_to_rgb(value: str) -> tuple[int, int, int] | None:
    if not value.lower().startswith("rgb"):
        return None
    nums = [float(x) for x in NUMBER_RE.findall(value)]
    if len(nums) < 3:
        return None
    return tuple(max(0, min(255, int(round(x)))) for x in nums[:3])  # type: ignore[return-value]


def _extract_color_values(root: ET.Element | None) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    if root is None:
        return colors

    for elem in root.iter():
        for key, value in elem.attrib.items():
            local = _strip_ns(key).lower()
            if local not in {"fill", "stroke", "stop-color", "color", "style"}:
                continue
            if value.strip().lower() in {"none", "transparent", "currentcolor"}:
                continue
            candidates = COLOR_RE.findall(value)
            for cand in candidates:
                low = cand.lower()
                rgb = _hex_to_rgb(cand) or _rgb_func_to_rgb(cand) or NAMED_COLORS.get(low)
                if rgb is not None:
                    colors.append(rgb)
    return colors


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta == 0:
        hue = 0.0
    elif mx == r:
        hue = (60 * ((g - b) / delta) + 360) % 360
    elif mx == g:
        hue = 60 * ((b - r) / delta) + 120
    else:
        hue = 60 * ((r - g) / delta) + 240
    sat = 0.0 if mx == 0 else delta / mx
    return hue, sat, mx


def _matches_color_group(rgb: tuple[int, int, int], group: str) -> bool:
    hue, sat, val = _rgb_to_hsv(rgb)
    if group == "black":
        return val < 0.28
    if group == "white":
        return val > 0.86 and sat < 0.18
    if group == "gray":
        return sat < 0.18 and 0.25 <= val <= 0.86
    if group == "red":
        return sat > 0.35 and (hue < 18 or hue >= 345)
    if group == "orange":
        return sat > 0.30 and 18 <= hue < 45
    if group == "yellow":
        return sat > 0.25 and 45 <= hue < 75
    if group == "green":
        return sat > 0.25 and 75 <= hue < 155
    if group == "teal":
        return sat > 0.25 and 155 <= hue < 195
    if group == "blue":
        return sat > 0.25 and 195 <= hue < 255
    if group == "purple":
        return sat > 0.25 and 255 <= hue < 300
    if group == "pink":
        return sat > 0.25 and 300 <= hue < 345
    if group == "brown":
        return sat > 0.25 and 15 <= hue < 55 and val < 0.75
    return False


def _numbers_from_svg(root: ET.Element | None) -> list[float]:
    if root is None:
        return []
    relevant = {
        "x",
        "y",
        "x1",
        "x2",
        "y1",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "width",
        "height",
        "points",
        "d",
        "viewBox",
        "stroke-width",
    }
    nums: list[float] = []
    for elem in root.iter():
        for key, value in elem.attrib.items():
            if _strip_ns(key) in relevant:
                nums.extend(float(x) for x in NUMBER_RE.findall(value))
    return nums


def _score_range(value: float, low: float, high: float, soft_low: float, soft_high: float) -> float:
    if low <= value <= high:
        return 1.0
    if soft_low <= value < low:
        return max(0.0, (value - soft_low) / max(1e-9, low - soft_low))
    if high < value <= soft_high:
        return max(0.0, (soft_high - value) / max(1e-9, soft_high - high))
    return 0.0


def validity_score(parsed: ParsedSvg) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if not parsed.candidate:
        return 0.0, ["empty output"]
    has_svg_bounds = parsed.candidate.lstrip().lower().startswith("<svg") and parsed.candidate.rstrip().lower().endswith("</svg>")
    if not has_svg_bounds:
        reasons.append("missing single svg envelope")
    if parsed.parse_error:
        return 0.05 if "<svg" in parsed.candidate.lower() else 0.0, [f"xml parse error: {parsed.parse_error}"]
    root_tag = _strip_ns(parsed.root.tag) if parsed.root is not None else ""
    score = 1.0
    if root_tag != "svg":
        score -= 0.45
        reasons.append("root is not svg")
    if parsed.had_wrapping_text:
        score -= 0.15
        reasons.append("extra text around svg")
    if "```" in parsed.raw:
        score -= 0.10
        reasons.append("markdown fence")
    if "xmlns" not in parsed.candidate[:300]:
        score -= 0.06
        reasons.append("missing xmlns")
    if not has_svg_bounds:
        score -= 0.15
    return max(0.0, score), reasons


def structure_score(root: ET.Element | None) -> tuple[float, dict[str, Any], list[str]]:
    if root is None:
        return 0.0, {}, ["no parsed tree"]

    tags = [_strip_ns(elem.tag) for elem in root.iter()]
    tag_counts = {tag: tags.count(tag) for tag in sorted(set(tags))}
    graphic_count = sum(1 for tag in tags if tag in GRAPHIC_TAGS)
    banned = sorted(set(tags).intersection(BANNED_TAGS))
    unsupported = sorted(set(tags).difference(GRAPHIC_TAGS | SUPPORT_TAGS))
    viewbox = root.attrib.get("viewBox") or root.attrib.get("viewbox") or ""

    score_parts = []
    reasons: list[str] = []

    if viewbox:
        nums = [float(x) for x in NUMBER_RE.findall(viewbox)]
        if len(nums) == 4:
            target = [0.0, 0.0, 256.0, 256.0]
            viewbox_score = sum(max(0.0, 1.0 - abs(a - b) / 256.0) for a, b in zip(nums, target)) / 4
        else:
            viewbox_score = 0.35
            reasons.append("malformed viewBox")
    else:
        viewbox_score = 0.0
        reasons.append("missing viewBox")
    score_parts.append(viewbox_score)

    count_score = _score_range(graphic_count, 4, 55, 1, 100)
    if graphic_count < 4:
        reasons.append("too few graphic elements")
    if graphic_count > 55:
        reasons.append("too many graphic elements")
    score_parts.append(count_score)

    diversity = len(set(tag for tag in tags if tag in GRAPHIC_TAGS))
    score_parts.append(min(1.0, diversity / 3.0))

    if banned:
        reasons.append(f"banned tags: {', '.join(banned)}")
    if unsupported:
        reasons.append(f"unsupported tags: {', '.join(unsupported[:5])}")
    tag_safety = 1.0 - min(0.8, 0.35 * len(banned) + 0.10 * len(unsupported))
    score_parts.append(tag_safety)

    return sum(score_parts) / len(score_parts), {"tag_counts": tag_counts, "graphic_count": graphic_count}, reasons


def geometry_score(root: ET.Element | None) -> tuple[float, dict[str, Any], list[str]]:
    nums = _numbers_from_svg(root)
    if not nums:
        return 0.0, {"numeric_count": 0}, ["no geometry numbers"]

    finite = [x for x in nums if math.isfinite(x)]
    if len(finite) != len(nums):
        return 0.05, {"numeric_count": len(nums)}, ["non-finite numbers"]

    in_canvas = [x for x in finite if -32 <= x <= 288]
    sane = [x for x in finite if -512 <= x <= 512]
    bounded_ratio = len(in_canvas) / len(finite)
    sane_ratio = len(sane) / len(finite)
    huge_penalty = 1.0 if max(abs(x) for x in finite) <= 1024 else 0.25

    spread = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    spread_score = _score_range(spread, 8, 135, 0, 280)

    score = 0.45 * bounded_ratio + 0.25 * sane_ratio + 0.20 * spread_score + 0.10 * huge_penalty
    reasons: list[str] = []
    if bounded_ratio < 0.82:
        reasons.append("many coordinates outside canvas")
    if huge_penalty < 1:
        reasons.append("extreme coordinates")
    if spread < 8:
        reasons.append("geometry nearly collapsed")

    return min(1.0, score), {
        "numeric_count": len(finite),
        "bounded_ratio": round(bounded_ratio, 4),
        "max_abs_number": round(max(abs(x) for x in finite), 4),
    }, reasons


def palette_score(root: ET.Element | None) -> tuple[float, dict[str, Any], list[str]]:
    colors = _extract_color_values(root)
    if not colors:
        return 0.0, {"color_count": 0, "unique_colors": []}, ["no colors"]

    unique = sorted(set(colors))
    unique_count = len(unique)
    palette_size_score = _score_range(unique_count, 2, 8, 1, 18)

    hsvs = [_rgb_to_hsv(rgb) for rgb in unique]
    values = [v for _, _, v in hsvs]
    saturations = [s for _, s, _ in hsvs]
    contrast = (max(values) - min(values)) if len(values) > 1 else 0.0
    contrast_score = min(1.0, contrast / 0.45)
    saturation_score = min(1.0, statistics.mean(saturations) / 0.35) if saturations else 0.0

    transparent = 0
    total_paint = 0
    if root is not None:
        for elem in root.iter():
            for attr in ("fill", "stroke"):
                if attr in elem.attrib:
                    total_paint += 1
                    if elem.attrib[attr].strip().lower() in {"none", "transparent"}:
                        transparent += 1
    visible_paint_score = 1.0 if total_paint == 0 else 1.0 - min(0.8, transparent / max(total_paint, 1))

    score = 0.40 * palette_size_score + 0.25 * contrast_score + 0.25 * saturation_score + 0.10 * visible_paint_score
    reasons: list[str] = []
    if unique_count < 2:
        reasons.append("single-color palette")
    if unique_count > 8:
        reasons.append("palette too large")
    if contrast < 0.15:
        reasons.append("low tonal contrast")

    return min(1.0, score), {
        "color_count": len(colors),
        "unique_color_count": unique_count,
        "unique_colors": ["#%02X%02X%02X" % rgb for rgb in unique[:12]],
    }, reasons


def prompt_alignment_score(prompt: str, root: ET.Element | None, svg_text: str) -> tuple[float, dict[str, Any], list[str]]:
    prompt_low = (prompt or "").lower()
    tags = [_strip_ns(elem.tag) for elem in _iter_elements(root)]
    colors = _extract_color_values(root)

    requested_colors = [
        group
        for group, words in COLOR_KEYWORDS.items()
        if any(re.search(rf"\b{re.escape(word)}\b", prompt_low) for word in words)
    ]
    color_hits = 0
    for group in requested_colors:
        if any(_matches_color_group(rgb, group) for rgb in colors):
            color_hits += 1
    color_score = 1.0 if not requested_colors else color_hits / len(requested_colors)

    requested_shapes = [
        group
        for group, words in SHAPE_KEYWORDS.items()
        if any(re.search(rf"\b{re.escape(word)}\b", prompt_low) for word in words)
    ]
    shape_hits = 0
    tag_set = set(tags)
    for group in requested_shapes:
        if group == "circle" and tag_set.intersection({"circle", "ellipse"}):
            shape_hits += 1
        elif group == "rect" and "rect" in tag_set:
            shape_hits += 1
        elif group == "line" and tag_set.intersection({"line", "path", "rect"}):
            shape_hits += 1
        elif group == "polygon" and tag_set.intersection({"polygon", "path"}):
            shape_hits += 1
        elif group == "path" and "path" in tag_set:
            shape_hits += 1
    shape_score = 1.0 if not requested_shapes else shape_hits / len(requested_shapes)

    literal_hexes = sorted(set(HEX_RE.findall(prompt)))
    svg_low = svg_text.lower()
    hex_hits = sum(1 for hx in literal_hexes if hx.lower() in svg_low)
    literal_hex_score = 1.0 if not literal_hexes else hex_hits / len(literal_hexes)

    score = 0.45 * color_score + 0.40 * shape_score + 0.15 * literal_hex_score
    reasons: list[str] = []
    if requested_colors and color_score < 0.5:
        reasons.append("few requested colors matched")
    if requested_shapes and shape_score < 0.5:
        reasons.append("few requested shapes matched")
    if literal_hexes and literal_hex_score < 0.5:
        reasons.append("few explicit hex colors reused")

    return score, {
        "requested_colors": requested_colors,
        "color_hits": color_hits,
        "requested_shapes": requested_shapes,
        "shape_hits": shape_hits,
        "literal_hexes": literal_hexes,
        "literal_hex_hits": hex_hits,
    }, reasons


def degeneration_score(text: str, root: ET.Element | None) -> tuple[float, dict[str, Any], list[str]]:
    stripped = (text or "").strip()
    length = len(stripped)
    length_score = _score_range(length, 220, 6500, 60, 12000)

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    duplicate_line_score = 1.0
    if lines:
        duplicate_line_score = len(set(lines)) / len(lines)

    elem_strings: list[str] = []
    if root is not None:
        for elem in root.iter():
            tag = _strip_ns(elem.tag)
            if tag in GRAPHIC_TAGS:
                attrs = " ".join(f"{k}={v}" for k, v in sorted(elem.attrib.items()))
                elem_strings.append(f"{tag} {attrs}")
    duplicate_elem_score = 1.0 if not elem_strings else len(set(elem_strings)) / len(elem_strings)

    has_forbidden_ref = False
    if root is not None:
        for elem in root.iter():
            for key, value in elem.attrib.items():
                local = _strip_ns(key).lower()
                if local in {"href", "src"} or local.endswith(":href"):
                    if re.search(r"https?://|data:|javascript:", value, re.IGNORECASE):
                        has_forbidden_ref = True
                if local == "style" and re.search(r"url\(\s*['\"]?(?:https?://|data:|javascript:)", value, re.IGNORECASE):
                    has_forbidden_ref = True
    elif re.search(r"(?:href|src)\s*=\s*['\"](?:https?://|data:|javascript:)", stripped, re.IGNORECASE):
        has_forbidden_ref = True
    ref_score = 0.0 if has_forbidden_ref else 1.0

    score = 0.40 * length_score + 0.20 * duplicate_line_score + 0.25 * duplicate_elem_score + 0.15 * ref_score
    reasons: list[str] = []
    if length_score < 0.8:
        reasons.append("suspicious output length")
    if duplicate_elem_score < 0.6:
        reasons.append("repeated identical elements")
    if has_forbidden_ref:
        reasons.append("external or unsafe reference")
    return min(1.0, score), {
        "length": length,
        "duplicate_element_score": round(duplicate_elem_score, 4),
    }, reasons


def score_svg(svg_text: str, prompt: str = "") -> dict[str, Any]:
    parsed = parse_svg(svg_text)

    validity, validity_reasons = validity_score(parsed)
    structure, structure_meta, structure_reasons = structure_score(parsed.root)
    geometry, geometry_meta, geometry_reasons = geometry_score(parsed.root)
    palette, palette_meta, palette_reasons = palette_score(parsed.root)
    alignment, alignment_meta, alignment_reasons = prompt_alignment_score(prompt, parsed.root, parsed.candidate)
    degeneration, degeneration_meta, degeneration_reasons = degeneration_score(svg_text, parsed.root)

    components = {
        "validity": validity,
        "structure": structure,
        "geometry": geometry,
        "palette": palette,
        "prompt_alignment": alignment,
        "anti_degenerate": degeneration,
    }
    weights = {
        "validity": 0.30,
        "structure": 0.20,
        "geometry": 0.18,
        "palette": 0.14,
        "prompt_alignment": 0.13,
        "anti_degenerate": 0.05,
    }
    raw_total = sum(components[name] * weights[name] for name in components)
    if validity < 0.20:
        raw_total *= 0.25
    elif validity < 0.60:
        raw_total *= 0.65

    reasons = {
        "validity": validity_reasons,
        "structure": structure_reasons,
        "geometry": geometry_reasons,
        "palette": palette_reasons,
        "prompt_alignment": alignment_reasons,
        "anti_degenerate": degeneration_reasons,
    }
    metadata = {
        "parse_error": parsed.parse_error,
        "had_wrapping_text": parsed.had_wrapping_text,
        "structure": structure_meta,
        "geometry": geometry_meta,
        "palette": palette_meta,
        "prompt_alignment": alignment_meta,
        "anti_degenerate": degeneration_meta,
    }
    return {
        "score": round(raw_total * 100.0, 4),
        "components": {k: round(v * 100.0, 4) for k, v in components.items()},
        "reasons": reasons,
        "metadata": metadata,
    }


def reward(prompt: str, svg_text: str) -> float:
    """Compatibility helper for simple callers."""
    return float(score_svg(svg_text, prompt)["score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", type=Path, help="SVG text file to score")
    parser.add_argument("--prompt", default="", help="Optional prompt text")
    parser.add_argument("--jsonl", type=Path, help="Score assistant SVGs from chat-format JSONL")
    args = parser.parse_args()

    if args.jsonl:
        rows = []
        for line in args.jsonl.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            messages = obj["messages"]
            rows.append(score_svg(messages[-1]["content"], messages[1]["content"]))
        mean_score = statistics.mean(row["score"] for row in rows) if rows else 0.0
        print(json.dumps({"count": len(rows), "mean_score": mean_score, "rows": rows}, indent=2))
        return

    if not args.svg:
        parser.error("--svg or --jsonl is required")
    print(json.dumps(score_svg(args.svg.read_text(encoding="utf-8"), args.prompt), indent=2))


if __name__ == "__main__":
    main()
