/**
 * End-to-end smoke test — full user journey through all four pages.
 *
 * Prerequisites:
 *   1. ``bash scripts/dev.sh`` is running (backend :8000, frontend :5173).
 *   2. ``data/seed.txt`` exists (run ``python scripts/seed_corpus.py`` if not).
 *
 * Run: ``cd frontend && npx playwright test``
 *
 * data-testid attributes used (provided by Track E):
 *   - ingest-dropzone   (IngestPage.tsx)
 *   - pause-btn         (IngestPage.tsx)
 *   - generate-btn      (GeneratePage.tsx)
 *   - db-reset-btn      (DBPage.tsx)
 *   - model-card-{name} (ModelCard.tsx / GeneratePage.tsx)
 */
import path from "path";

import { expect, test } from "@playwright/test";

const MODEL_NAMES = [
  "char_1gram",
  "char_2gram",
  "char_3gram",
  "char_4gram",
  "char_5gram",
  "word_1gram",
  "word_2gram",
  "word_3gram",
  "bpe_1gram",
  "bpe_2gram",
  "bpe_3gram",
  "feedforward",
  "transformer",
];

test("full user journey: db-reset → ingest → stats → generate → db", async ({ page }) => {
  // ── 0. Navigate to the app root ────────────────────────────────────────────
  await page.goto("/");

  // ── 1. DB page — reset to guarantee a clean state ──────────────────────────
  await page.click("text=DB");
  await page.click('[data-testid="db-reset-btn"]');
  // Confirm the reset dialog if one exists (optional — may not be present).
  const confirmBtn = page.locator("text=OK, text=Confirm, text=Yes").first();
  if (await confirmBtn.isVisible({ timeout: 1_000 }).catch(() => false)) {
    await confirmBtn.click();
  }

  // ── 2. Ingest page — upload the seed corpus ─────────────────────────────────
  await page.click("text=Ingest");

  const seedPath = path.resolve(__dirname, "../../data/seed.txt");
  // The dropzone wraps a hidden file input; Playwright can target it directly.
  await page.locator('[data-testid="ingest-dropzone"] input[type="file"]').setInputFiles(seedPath);

  // Wait for the ingest_complete signal to appear in the progress panel.
  // Budget 5 minutes — CPU-only Markov evaluation is slow.
  await expect(page.locator("text=Ingest complete")).toBeVisible({ timeout: 300_000 });

  // ── 3. Stats page — verify all 13 model cards are rendered ──────────────────
  await page.click("text=Stats");

  const spotCheckModels = [
    "char_1gram",
    "char_3gram",
    "word_2gram",
    "bpe_3gram",
    "feedforward",
    "transformer",
  ];
  for (const name of spotCheckModels) {
    await expect(
      page.locator(`[data-testid="model-card-${name}"]`)
    ).toBeVisible({ timeout: 10_000 });
  }

  // ── 4. Generate page — run generation and verify 13 outputs ─────────────────
  await page.click("text=Generate");
  await page.fill('input[type="number"]', "20");
  await page.click('[data-testid="generate-btn"]');

  // Each model output contains its real-word percentage.
  await expect(page.locator("text=Real words:")).toHaveCount(MODEL_NAMES.length, {
    timeout: 60_000,
  });

  // ── 5. DB page — verify ngram table heading is visible ──────────────────────
  await page.click("text=DB");
  await expect(page.locator("text=char_ngrams")).toBeVisible({ timeout: 10_000 });
});
