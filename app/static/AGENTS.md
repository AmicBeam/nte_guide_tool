# Static Module Guidelines

- Character avatars and default portraits should be sourced from the latest NTEData assets first, then Nanoka when needed. Preserve character scale, head alignment, and framing. Read `docs/character-image-assets.md` before importing or replacing images; reuse the shared avatar/portrait paths and existing WebP files. Record the actual source path and version for each new asset.
- `common/css/app.css` is the stylesheet entrypoint only. Keep it as an ordered list of `@import` statements and do not add page rules there.
- Put shared CSS in `common/css/`:
  - `foundation.css`: tokens, reset, shared cards, buttons, forms, and global shell layout.
  - `catalog-cards.css`: reusable character, build, and item card primitives.
  - `responsive.css`: shared media queries and viewport-specific overrides.
- Put card game page CSS in `app/modules/card_game/static/css/`:
  - `build-page.css`, `codex-page.css`, `analytics-page.css`: page-specific layout and states.
  - `duel-table*.css`: table page layout, board, sidebar, overlays, and table-specific states.
- Put Kongmu page CSS in `app/modules/kongmu/static/css/` and shaft page CSS in `app/modules/shaft/static/css/`.
- Put shared JavaScript in `common/js/`; page-owned JavaScript belongs to `app/modules/<module>/static/js/`.
- `common/css/app.css` imports shared styles only. Each module template loads its module-owned styles through that module's Flask blueprint.
- Preserve import order when moving rules. Later modules may intentionally override earlier shared primitives.
- Keep high-frequency map interactions cheap: prefer transforms, CSS variables, stable dimensions, and lightweight overlays over expensive shadows or layout-triggering changes.
- Do not encode game rules in CSS. CSS should only express presentation and interaction state returned by the backend or table UI code.
