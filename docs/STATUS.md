# Status — aiodoo-colab

**Living document.** `main` is the only source of truth.  
**Permanent branch:** `main`  
**Git tags / GitHub Releases:** none (metadata reset)  
**Historical evidence:** `docs/archive/`

## Purpose

Colab / Drive orchestration for training launches (subprocess). Does not own training logic or product composition.

## Current implementation (on main)

| Item | Status |
|------|--------|
| Training orchestration launcher | Shipped |
| Product composition | Out of scope |
| Full multi-repo product pipeline in Colab | **Not shipped** — optional Future Work |
| Required for Running System (ECO-1) | **No** — Training infrastructure only |

## Living docs

- `README.md`, `CHANGELOG.md`
- `docs/STATUS.md`, `docs/archive/`
