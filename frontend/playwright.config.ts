import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E configuration.
 *
 * Requires the full dev stack to be running via ``scripts/dev.sh`` before
 * executing: ``cd frontend && npx playwright test``.
 */
export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",

  // Fail fast — these tests are slow; one failure is enough signal.
  fullyParallel: false,
  workers: 1,

  // Generous timeouts because ingest can take several minutes.
  timeout: 360_000,
  expect: { timeout: 10_000 },

  reporter: [["list"], ["html", { open: "never" }]],

  use: {
    baseURL: "http://localhost:5173",
    // Capture screenshots and traces on failure for post-mortem debugging.
    screenshot: "only-on-failure",
    trace: "on-first-retry",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
