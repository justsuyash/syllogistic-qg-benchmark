/**
 * 1-Command Reproducibility Verifier (`npm run verify`) for the 400-Run (80 Cases x 5 Arms)
 * Contrastive Boundary Pinning (CBP) & Factored 2x2 Diagnostic Question-Generation Benchmark:
 * Recomputes Table 2 (Combined 400-Run Leaderboard), Table 3 (4-Subtrack Breakdown),
 * and Table 4 (5x5 Cyclic Latin-Square Balance & Positional Bias) directly from
 * `data/runs-400-five-arm.json` in < 1 second and asserts exact numerical agreement.
 */
import fs from "node:fs"
import path from "node:path"
import {
  compareArmPairCuped,
  PUBLIC_ARM_IDS,
  summarizeArmMetrics,
  verifyLatinSquareBalance,
  type ArmSummaryMetrics,
  type BenchmarkCaseRecord,
} from "./metrics"

function assertEqual(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `[Verification Failure] ${label}: expected ${JSON.stringify(
        expected
      )}, got ${JSON.stringify(actual)}`
    )
  }
}

function formatArmMarkdownRow(m: ArmSummaryMetrics): string {
  return `| \`${m.armId}\` | ${m.meanJudgeScore.toFixed(2)} | ${m.firstPlaceWins}/${
    m.cases
  } (${m.winRatePct}%) | ${
    m.validKeyRatePct != null ? `${m.validKeyRatePct}%` : "—"
  } | ${
    m.noviceFoilBiteRatePct != null ? `${m.noviceFoilBiteRatePct}%` : "—"
  } | ${
    m.meanDiscriminationD != null ? m.meanDiscriminationD.toFixed(3) : "—"
  } | ${m.meanEverydayClarityScore} | ${
    m.debativeStemRatePct != null ? `${m.debativeStemRatePct}%` : "100%"
  } | ${
    m.predictionOnlyStemRatePct != null
      ? `${m.predictionOnlyStemRatePct}%`
      : "100%"
  } | ${m.p50LatencyMs}ms / ${m.p95LatencyMs}ms | $${m.totalCostUsd.toFixed(
    4
  )} | ${m.validHighDPerDollar != null ? m.validHighDPerDollar : "—"} |`
}

export function runPaper1Verification(
  repoDir = path.resolve(__dirname, "..")
) {
  const runsPath = path.join(repoDir, "data/runs-400-five-arm.json")
  const raw = JSON.parse(fs.readFileSync(runsPath, "utf-8")) as {
    totalCases: number
    totalArms: number
    totalGenerations: number
    cases: BenchmarkCaseRecord[]
  }

  const blocks = raw.cases
  assertEqual("Total Verbatim Cases", blocks.length, 80)
  const totalArmRuns = blocks.reduce(
    (s, b) => s + Object.keys(b.arms).length,
    0
  )
  assertEqual("Total Live Arm Generations", totalArmRuns, 400)

  // 1. Verify exact 5x5 Cyclic Latin-Square Balance [16, 16, 16, 16, 16] overall
  const latin = verifyLatinSquareBalance(blocks)
  assertEqual("Exact 5x5 Uniform Latin Square", latin.isExactUniformLatinSquare, true)
  for (const armId of PUBLIC_ARM_IDS) {
    assertEqual(
      `Latin Square Slot Exposure for ${armId}`,
      latin.armSlotExposure[armId],
      [16, 16, 16, 16, 16]
    )
  }
  assertEqual("Slot 1 + Slot 2 First-Place Wins", latin.top2SlotWins, 54)
  assertEqual("Slot 1 + Slot 2 Win Share %", latin.top2SlotWinSharePct, 67.5)

  // Verify [4, 4, 4, 4, 4] balance per 20-case sub-track
  const mediqBlocks = blocks.filter((b) => b.track === "mediq")
  const medqaBlocks = blocks.filter((b) => b.track === "medqa")
  const medicalCombinedBlocks = [...mediqBlocks, ...medqaBlocks]
  const track2ABlocks = blocks.filter(
    (b) => b.track === "stem-humanities-yr1-school"
  )
  const track2BBlocks = blocks.filter((b) => b.track === "undergrad-yr2-domain")

  for (const [trackName, subBlocks] of [
    ["mediq", mediqBlocks],
    ["medqa", medqaBlocks],
    ["stem-humanities-yr1-school", track2ABlocks],
    ["undergrad-yr2-domain", track2BBlocks],
  ] as const) {
    const subLatin = verifyLatinSquareBalance(subBlocks)
    for (const armId of PUBLIC_ARM_IDS) {
      assertEqual(
        `Sub-track ${trackName} 5x5 Slot Exposure for ${armId}`,
        subLatin.armSlotExposure[armId],
        [4, 4, 4, 4, 4]
      )
    }
  }

  // 2. Recompute Combined 80-Case (400-Run) Leaderboard
  const overallByArm = Object.fromEntries(
    PUBLIC_ARM_IDS.map((id) => [id, summarizeArmMetrics(blocks, id)])
  ) as Record<(typeof PUBLIC_ARM_IDS)[number], ArmSummaryMetrics>

  assertEqual(
    "Arm 0 Overall Valid Key %",
    overallByArm["arm-0-single-shot"].validKeyRatePct,
    76.7
  )
  assertEqual(
    "Arm 1 ALFA Overall Valid Key %",
    overallByArm["arm-1-alfa-checklist"].validKeyRatePct,
    88.3
  )
  assertEqual(
    "Arm 2 Unguarded Pass-2 Overall Valid Key %",
    overallByArm["arm-2-checklist-unguarded-pass2"].validKeyRatePct,
    85.0
  )
  assertEqual(
    "Arm 3 CBP Unfactored Rationale Stem Overall Valid Key %",
    overallByArm["arm-3-cbp-rationale-stem"].validKeyRatePct,
    95.0
  )
  assertEqual(
    "Arm 4 CBP Factored 2x2 Overall Valid Key %",
    overallByArm["arm-4-cbp-factored-2x2"].validKeyRatePct,
    91.7
  )
  assertEqual(
    "Arm 3 CBP Rationale Stem Novice Foil Bite %",
    overallByArm["arm-3-cbp-rationale-stem"].noviceFoilBiteRatePct,
    8.3
  )
  assertEqual(
    "Arm 4 CBP Factored 2x2 Novice Foil Bite %",
    overallByArm["arm-4-cbp-factored-2x2"].noviceFoilBiteRatePct,
    15.0
  )
  assertEqual(
    "Arm 3 Discrimination D",
    overallByArm["arm-3-cbp-rationale-stem"].meanDiscriminationD,
    0.033
  )
  assertEqual(
    "Arm 4 Discrimination D (Highest Overall)",
    overallByArm["arm-4-cbp-factored-2x2"].meanDiscriminationD,
    0.067
  )
  assertEqual(
    "Arm 3 Factored 2x2 Compliance %",
    overallByArm["arm-3-cbp-rationale-stem"].predictionOnlyStemRatePct,
    0
  )
  assertEqual(
    "Arm 4 Factored 2x2 Compliance %",
    overallByArm["arm-4-cbp-factored-2x2"].predictionOnlyStemRatePct,
    100
  )
  assertEqual(
    "Arm 4 STE Clarity Score (Highest Overall)",
    overallByArm["arm-4-cbp-factored-2x2"].meanEverydayClarityScore,
    97.4
  )

  // 3. Recompute 4 Sub-Tracks
  const medqaArm1 = summarizeArmMetrics(medqaBlocks, "arm-1-alfa-checklist")
  const medqaArm2 = summarizeArmMetrics(
    medqaBlocks,
    "arm-2-checklist-unguarded-pass2"
  )
  const medqaArm3 = summarizeArmMetrics(
    medqaBlocks,
    "arm-3-cbp-rationale-stem"
  )
  const medqaArm4 = summarizeArmMetrics(
    medqaBlocks,
    "arm-4-cbp-factored-2x2"
  )
  assertEqual("MedQA Arm 1 ALFA Valid Key %", medqaArm1.validKeyRatePct, 80.0)
  assertEqual("MedQA Arm 2 Unguarded Pass-2 Valid Key %", medqaArm2.validKeyRatePct, 65.0)
  assertEqual("MedQA Arm 3 CBP Rationale Stem Valid Key %", medqaArm3.validKeyRatePct, 100.0)
  assertEqual("MedQA Arm 4 CBP Factored 2x2 Valid Key %", medqaArm4.validKeyRatePct, 95.0)

  const track2AArm3 = summarizeArmMetrics(
    track2ABlocks,
    "arm-3-cbp-rationale-stem"
  )
  const track2AArm4 = summarizeArmMetrics(
    track2ABlocks,
    "arm-4-cbp-factored-2x2"
  )
  assertEqual("Track 2A Arm 3 Unfactored Foil Bite %", track2AArm3.noviceFoilBiteRatePct, 0.0)
  assertEqual("Track 2A Arm 4 Factored 2x2 Foil Bite %", track2AArm4.noviceFoilBiteRatePct, 25.0)
  assertEqual("Track 2A Arm 4 Discrimination D", track2AArm4.meanDiscriminationD, 0.1)

  const track2BArm4 = summarizeArmMetrics(
    track2BBlocks,
    "arm-4-cbp-factored-2x2"
  )
  assertEqual("Track 2B Arm 4 Valid Key %", track2BArm4.validKeyRatePct, 95.0)
  assertEqual("Track 2B Arm 4 Discrimination D", track2BArm4.meanDiscriminationD, 0.1)

  // 4. Recompute CUPED Pairwise Contrasts
  const cupedArm4VsArm3 = compareArmPairCuped(
    blocks,
    "arm-4-cbp-factored-2x2",
    "arm-3-cbp-rationale-stem",
    "Factored 2x2 Stem (Arm 4) vs. Unfactored Rationale Stem (Arm 3)"
  )
  assertEqual("CUPED Arm 4 vs Arm 3 Delta", cupedArm4VsArm3.cupedDelta, -1.129)

  return {
    totalCases: blocks.length,
    totalArmRuns,
    latin,
    overallByArm,
    medicalCombinedByArm: Object.fromEntries(
      PUBLIC_ARM_IDS.map((id) => [
        id,
        summarizeArmMetrics(medicalCombinedBlocks, id),
      ])
    ),
    track2AByArm: Object.fromEntries(
      PUBLIC_ARM_IDS.map((id) => [id, summarizeArmMetrics(track2ABlocks, id)])
    ),
    track2BByArm: Object.fromEntries(
      PUBLIC_ARM_IDS.map((id) => [id, summarizeArmMetrics(track2BBlocks, id)])
    ),
  }
}

if (require.main === module) {
  const res = runPaper1Verification()
  console.log(
    "================================================================================"
  )
  console.log(
    " CBP & FACTORED 2x2 REPRODUCIBILITY VERIFIER — 400 RUNS (80 CASES x 5 ARMS, 5x5 LATIN SQUARE)"
  )
  console.log(
    "================================================================================\n"
  )
  console.log(
    `[PASS] Total Verbatim Cases = ${res.totalCases} | Total Generations = ${res.totalArmRuns}`
  )
  console.log(
    `[PASS] Exact 5x5 Cyclic Latin-Square Balance = [16,16,16,16,16] overall and [4,4,4,4,4] per 20-case sub-track (max deviation = ${res.latin.maxDeviationFromUniform})`
  )
  console.log(
    `[PASS] Multi-Candidate Judge Positional Bias Caught = Slots 1+2 won ${res.latin.top2SlotWins}/${res.totalCases} (${res.latin.top2SlotWinSharePct}%) of 1st-place ranks\n`
  )
  console.log(
    "### Combined 80-Case Leaderboard (400 Runs · 5x5 Cyclic Latin Square)"
  )
  console.log(
    "| Arm Slug | Rubric | 1st Wins | Valid Key % | Foil Bite % | Discrim. D | Clarity | Debative % | Factored 2x2 % | P50 / P95 | Cost | High-D / $1 |"
  )
  console.log(
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
  )
  for (const armId of PUBLIC_ARM_IDS) {
    console.log(formatArmMarkdownRow(res.overallByArm[armId]))
  }
  console.log("\n[ALL STATISTICAL, LATIN-SQUARE & PSYCHOMETRIC ASSERTIONS PASSED]")
}
