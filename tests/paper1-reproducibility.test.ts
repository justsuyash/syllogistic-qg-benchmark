import { describe, expect, it } from "vitest"
import {
  auditPredictionOnly2x2Stem,
  computeEverydayClarity,
  PUBLIC_ARM_IDS,
} from "../src/metrics"
import { shuffleFourOptionsDeterministic } from "../src/solver-harness"
import {
  buildArm0SingleShotMcqPrompt,
  buildArm1AlfaChecklistMcqPrompt,
  buildArm3CbpRationaleStemMcqPrompt,
  buildArm4CbpFactored2x2McqPrompt,
  buildPlainEnglishMediqPrompt,
  DECODING_HYPERPARAMETERS,
  evaluateMonotonicSolverBGate,
  FIVE_PLAIN_MISTAKE_TYPES,
  PLAIN_HUMAN_PLAN_SLOT_DESCRIPTIONS,
  PlainHumanPlanSchema,
  sanitizePredictionOnlyStem,
  validatePredictionOnlyDebativeStemContract,
} from "../src/reference-prompts"
import { runPaper1Verification } from "../src/verify-paper1"
import fs from "node:fs"
import path from "node:path"

describe("400-Run, 5-Arm 5x5 Cyclic Latin-Square Contrastive Boundary Pinning (CBP) Benchmark Reproducibility", () => {
  it("verifies exact 400 runs, [16,16,16,16,16] Latin Square ([4,4,4,4,4] per sub-track), Valid Key %, Novice Foil Bite %, and CUPED math", () => {
    const res = runPaper1Verification(path.resolve(__dirname, ".."))
    expect(res.totalCases).toBe(80)
    expect(res.totalArmRuns).toBe(400)
    expect(res.latin.isExactUniformLatinSquare).toBe(true)
    expect(res.latin.maxDeviationFromUniform).toBe(0)
    expect(res.latin.top2SlotWins).toBe(54)
    expect(res.latin.top2SlotWinSharePct).toBe(67.5)

    // Overall 60-MCQ Valid Key % progression
    expect(res.overallByArm["arm-0-single-shot"].validKeyRatePct).toBe(76.7)
    expect(res.overallByArm["arm-1-alfa-checklist"].validKeyRatePct).toBe(88.3)
    expect(res.overallByArm["arm-2-checklist-unguarded-pass2"].validKeyRatePct).toBe(85.0)
    expect(res.overallByArm["arm-3-cbp-rationale-stem"].validKeyRatePct).toBe(95.0)
    expect(res.overallByArm["arm-4-cbp-factored-2x2"].validKeyRatePct).toBe(91.7)

    // Rationale Leakage Mitigation (Arm 3 -> Arm 4 lifts Foil Bite % from 8.3% to 15.0% and doubles D from +0.033 to +0.067)
    expect(res.overallByArm["arm-3-cbp-rationale-stem"].noviceFoilBiteRatePct).toBe(8.3)
    expect(res.overallByArm["arm-4-cbp-factored-2x2"].noviceFoilBiteRatePct).toBe(15.0)
    expect(res.overallByArm["arm-3-cbp-rationale-stem"].meanDiscriminationD).toBe(0.033)
    expect(res.overallByArm["arm-4-cbp-factored-2x2"].meanDiscriminationD).toBe(0.067)
    expect(res.overallByArm["arm-4-cbp-factored-2x2"].predictionOnlyStemRatePct).toBe(100)

    // Sub-Tracks: MedQA, Track 2A, Track 2B
    expect(res.track2AByArm["arm-3-cbp-rationale-stem"].noviceFoilBiteRatePct).toBe(0.0)
    expect(res.track2AByArm["arm-4-cbp-factored-2x2"].noviceFoilBiteRatePct).toBe(25.0)
    expect(res.track2BByArm["arm-4-cbp-factored-2x2"].validKeyRatePct).toBe(95.0)
    expect(res.track2BByArm["arm-4-cbp-factored-2x2"].meanDiscriminationD).toBe(0.1)
  })

  it("audits PlainHumanPlanSchema (Register-Decoupled CBP), sanitizePredictionOnlyStem, Factored 2x2 stems, and monotonic Solver B gate", () => {
    const cleanAudit = auditPredictionOnly2x2Stem({
      questionStem:
        "During a code review of a recursive binary search, Speaker 1 predicts the base case triggers on an empty slice, while Speaker 2 predicts it causes an infinite recursion. Who is right, and why?",
      correctOption:
        "Speaker 1 is right because `high < low` terminates immediately when the sub-array becomes empty.",
      distractors: [
        "Speaker 1 is right because integer division rounds toward negative infinity on all positive indices.",
        "Speaker 2 is right because `mid` never moves past `low` when `low == high`.",
        "Speaker 2 is right because the call stack overflows before `high` is decremented.",
      ],
    })
    expect(cleanAudit.hasDebativeClash).toBe(true)
    expect(cleanAudit.zeroCausalLeakInStem).toBe(true)
    expect(cleanAudit.has2x2OptionBalance).toBe(true)
    expect(cleanAudit.isStrictPredictionOnly2x2).toBe(true)

    const sanitized = sanitizePredictionOnlyStem(
      "Two students test a circuit. Speaker 1 predicts the bulb brightens because resistance drops, while Speaker 2 predicts the bulb dims since current splits. Who is right, and why? Note that the battery is ideal."
    )
    expect(sanitized).toBe(
      "Two students test a circuit. Speaker 1 predicts the bulb brightens, while Speaker 2 predicts the bulb dims. Who is right, and why?"
    )

    const plan = PlainHumanPlanSchema({
      whyTwoAnswersLookRightAtFirst: "Both answers mention current and voltage.",
      whatActuallyHappens: "Current increases when resistance drops.",
      everydayExplanation: "A wider pipe lets more water flow at the same push.",
      oneFactInQuestionThatSettlesIt: "The resistor is halved to 4 ohms.",
      threeCommonMistakes: [
        {
          mistakeType: "REVERSED_THE_DIRECTION_OR_SIGN",
          whySomeonePicksIt: "Flips Ohm's law.",
          exactDetailInQuestionThatBreaksIt: "4 ohms is smaller than 8 ohms.",
        },
        {
          mistakeType: "MISSED_ONE_KEY_DETAIL_IN_SETUP",
          whySomeonePicksIt: "Misses the parallel branch.",
          exactDetailInQuestionThatBreaksIt: "Switch S is closed.",
        },
        {
          mistakeType: "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS",
          whySomeonePicksIt: "Assumes constant current source.",
          exactDetailInQuestionThatBreaksIt: "Ideal 12 V battery.",
        },
      ],
    })
    expect(plan.threeCommonMistakes.length).toBe(3)
    expect(new Set(plan.threeCommonMistakes.map((m) => m.mistakeType)).size).toBe(3)

    const clarity = computeEverydayClarity(
      "Two students test how a resistor network behaves when a parallel branch opens."
    )
    expect(clarity.everydayClarityScore).toBeGreaterThanOrEqual(90)

    const deck1 = shuffleFourOptionsDeterministic("Key", ["D1", "D2", "D3"], "case-01")
    const deck2 = shuffleFourOptionsDeterministic("Key", ["D1", "D2", "D3"], "case-01")
    expect(deck1).toEqual(deck2)
    expect(deck1.options[deck1.correctOptionId]).toBe("Key")

    const repoRoot = path.resolve(__dirname, "..")
    const manifest = JSON.parse(
      fs.readFileSync(path.join(repoRoot, "data/cases-80-manifest.json"), "utf-8")
    )
    const leaderboard = JSON.parse(
      fs.readFileSync(path.join(repoRoot, "data/leaderboard-summary.json"), "utf-8")
    )
    expect(manifest.cases.length).toBe(80)
    expect(Object.keys(leaderboard.overall.arms)).toEqual([...PUBLIC_ARM_IDS])

    // Verify reference prompt schemas & monotonic Solver B gate
    expect(DECODING_HYPERPARAMETERS.binomialStandardError60McqPct).toBe(3.2)
    expect(Object.keys(PLAIN_HUMAN_PLAN_SLOT_DESCRIPTIONS).length).toBe(5)
    expect(FIVE_PLAIN_MISTAKE_TYPES.length).toBe(5)

    const sampleInput = {
      specialty: "Clinical Medicine",
      sourcePassage: "A patient presents with acute shortness of breath and sharp chest pain.",
    }
    const p0 = buildArm0SingleShotMcqPrompt(sampleInput)
    const p1 = buildArm1AlfaChecklistMcqPrompt(sampleInput)
    const p3 = buildArm3CbpRationaleStemMcqPrompt(sampleInput)
    const p4 = buildArm4CbpFactored2x2McqPrompt(sampleInput)
    const pMediq = buildPlainEnglishMediqPrompt({
      specialty: "Internal Medicine",
      patientPresentation: "A 30-year-old arrives with fever and a sore knee.",
    })
    expect(p0.armAlias).toBe("arm-0-single-shot")
    expect(p1.armAlias).toBe("arm-1-alfa-checklist")
    expect(p3.armAlias).toBe("arm-3-cbp-rationale-stem")
    expect(p4.armAlias).toBe("arm-4-cbp-factored-2x2")
    expect(p4.system).toContain("whyTwoAnswersLookRightAtFirst")
    expect(p4.system).toContain("oneFactInQuestionThatSettlesIt")
    expect(pMediq.system).toContain("mostLikelyCondition")
    expect(pMediq.system).toContain("closestLookalikeCondition")
    expect(pMediq.system).toContain("oneSimpleQuestionToTellThemApart")

    expect(
      evaluateMonotonicSolverBGate({
        pass1ValidKey: true,
        pass1ParityDefects: 0,
        pass2ValidKey: true,
        pass2ParityDefects: 0,
      })
    ).toEqual({ selectedPass: "PASS_1", rolledBackToPass1: false })

    expect(
      evaluateMonotonicSolverBGate({
        pass1ValidKey: true,
        pass1ParityDefects: 1,
        pass2ValidKey: false,
        pass2ParityDefects: 0,
      })
    ).toEqual({ selectedPass: "PASS_1", rolledBackToPass1: true })

    expect(
      validatePredictionOnlyDebativeStemContract({
        questionStem:
          "Two engineers inspect a 3-strand pulley lifting a 72 kg load at 1.0 m/s^2. Engineer 1 predicts rope tension is 264 N, while Engineer 2 predicts rope tension is 240 N. Who is right, and why?",
        correctOption:
          "Engineer 1 is right because total upward force 72 * (10 + 1) = 792 N is shared equally across 3 supporting strands.",
        distractors: [
          "Engineer 1 is right because downward gravity subtracts from upward acceleration before dividing by 3.",
          "Engineer 2 is right because only static weight 720 N is divided across the 3 supporting strands.",
          "Engineer 2 is right because net upward acceleration reduces the tension required in each vertical strand.",
        ],
      }).passesContract
    ).toBe(true)
  })
})
