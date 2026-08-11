"""Fixed media library for curated Neuromancer reactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_MEDIA_TYPES = {"photo", "video", "sticker", "animation", "document"}
SUPPORTED_SEND_MODES = {"with_optional_text", "media_only", "caption"}


class MediaLibraryError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaAsset:
    id: str
    path: Path
    type: str
    description: str
    use_when: tuple[str, ...]
    avoid_when: tuple[str, ...]
    send_mode: str


@dataclass(frozen=True)
class MediaLibrary:
    root: Path
    assets: tuple[MediaAsset, ...]

    @classmethod
    def load(cls, index_path: Path) -> "MediaLibrary":
        if not index_path.exists():
            return cls(root=index_path.parent, assets=())

        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MediaLibraryError(f"Invalid media index JSON: {index_path}") from exc

        root = index_path.parent.parent if index_path.name == "index.json" else index_path.parent
        assets = tuple(parse_asset(item, root=root) for item in raw.get("assets", []))
        seen: set[str] = set()
        for asset in assets:
            if asset.id in seen:
                raise MediaLibraryError(f"Duplicate media asset id: {asset.id}")
            seen.add(asset.id)
        return cls(root=root, assets=assets)

    def get(self, asset_id: str | None) -> MediaAsset | None:
        if not asset_id:
            return None
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None

    def has(self, asset_id: str | None) -> bool:
        return self.get(asset_id) is not None

    def prompt_summary(self) -> str:
        if not self.assets:
            return "No fixed media assets are currently available."

        lines: list[str] = []
        for asset in self.assets:
            lines.append(f"- id: {asset.id}")
            lines.append(f"  type: {asset.type}")
            lines.append(f"  send_mode: {asset.send_mode}")
            lines.append(f"  description: {asset.description}")
            if asset.use_when:
                lines.append("  use_when:")
                lines.extend(f"    - {item}" for item in asset.use_when)
            if asset.avoid_when:
                lines.append("  avoid_when:")
                lines.extend(f"    - {item}" for item in asset.avoid_when)
        return "\n".join(lines)


def parse_asset(item: dict[str, Any], *, root: Path) -> MediaAsset:
    asset_id = required_text(item, "id")
    raw_path = required_text(item, "path")
    media_type = required_text(item, "type").casefold()
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise MediaLibraryError(f"Unsupported media type for {asset_id}: {media_type}")

    send_mode = str(item.get("send_mode") or "with_optional_text").casefold()
    if send_mode not in SUPPORTED_SEND_MODES:
        raise MediaLibraryError(f"Unsupported send_mode for {asset_id}: {send_mode}")

    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise MediaLibraryError(f"Media asset file does not exist for {asset_id}: {path}")

    return MediaAsset(
        id=asset_id,
        path=path,
        type=media_type,
        description=str(item.get("description") or ""),
        use_when=tuple(str(value) for value in item.get("use_when") or ()),
        avoid_when=tuple(str(value) for value in item.get("avoid_when") or ()),
        send_mode=send_mode,
    )


def required_text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MediaLibraryError(f"Media asset is missing required field: {key}")
    return value.strip()
