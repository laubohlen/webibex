# CR: fix landmark click-coordinate scaling for images narrower than LANDMARK_IMAGE_WIDTH

**What changed:**
- Modified: `static/css/tailwind.css` — `.imageToLandmark` gained `width: 100%; height: auto;` (previously had no width rule at all).
- Rebuilt: `static/css/style.css` (`tailwind build`), `static/css/style.min.css` (`cleancss`), and their `staticfiles/css/` collected copies (`manage.py collectstatic`) — same three files, source + collected.

**Follow-up action:** none — self-contained CSS fix, already built and collected.

**Do NOT:**
- Re-run `manage.py collectstatic` broadly without scoping the diff first. Doing so this session also touched unrelated `staticfiles/admin/*` and `staticfiles/filer/*` files (real content drift — Django 5.2.15's own admin CSS/JS differs from whatever version last generated the committed `staticfiles/` tree, likely stale since the `480607b` Django version bump never re-ran `collectstatic`). That drift was reverted (`git checkout -- staticfiles/admin/ staticfiles/filer/`) and is **out of scope** for this fix — it's a separate, pre-existing gap (tracked as a TODO in `docs/security-remediation-plan.md`), not something to silently fold in here.
- Assume this is TF-migration-related. It isn't — reproduces identically regardless of which backend serves embeddings (confirmed: the bug is entirely in `core/utils.py:scale_coordinate()` + this CSS, both of which run before any embedding call).

**Trigger:** any change to the landmarking UI (`templates/simple_landmarks/multi_landmarking.html`) or to `settings.LANDMARK_IMAGE_WIDTH` — re-verify the `<img>` actually renders at that width for images both narrower and wider than it.

**Why:** `templates/simple_landmarks/multi_landmarking.html`'s click handler uses `event.target.getBoundingClientRect()` on the `<img>` element to capture `horn_x`/`horn_y`/`eye_x`/`eye_y`, and `core/views.py`'s `save_landmarks_view` scales those back to real image coordinates via `core/utils.py:scale_coordinate(x, y, image.width, settings.LANDMARK_IMAGE_WIDTH)` — an unconditional assumption that the image rendered at exactly `LANDMARK_IMAGE_WIDTH` (1600px). But `.imageToLandmark` had no width rule, and the parent `<main class="fixedContainer">`'s inline `width: 1600px` doesn't force the child `<img>` to fill it — browsers don't upscale `<img>` elements to a wider parent by default. Any uploaded image narrower than 1600px (confirmed with real ~1024px stock photos) rendered at its own natural size, so click coordinates were already in real-image-space, then got needlessly shrunk again by the `image.width/1600` factor — landmarks ended up shrunk toward the image's top-left corner, in one case landing entirely off the animal (blank sky/clouds), producing a garbage crop.

Discovered and diagnosed during the 2026-07-23 TF2210 e2e manual test (see `docs/tf1-to-tf2-migration-plan.md`) — unrelated to that migration itself, confirmed by code-path analysis (landmark save happens before any embedding call) and reproduced with annotated before/after screenshots.

**Verify:** upload an image narrower than 1600px, click horn tip + eye corner, confirm the saved `LandmarkItem` coordinates land on the actual horn/eye when plotted back onto the source image (`cv2.circle` at the stored `(x, y)`) — not near the top-left corner or off-image. Confirmed working this session across multiple ~1024px test images after the fix.

**Rollback:** `git revert` this commit, or manually drop the `width: 100%; height: auto;` lines from `.imageToLandmark` and rebuild (`tailwind build` + `cleancss` + `collectstatic`).
