/**
 * Live Replication CLI (`npm run replicate:live`):
 * Allows external researchers and reviewers to test our exact Register-Decoupled Contrastive
 * Boundary Pinning (CBP) reference prompts (`arm-0-single-shot`, `arm-1-alfa-checklist`,
 * `arm-3-cbp-rationale-stem`, `arm-4-cbp-factored-2x2`) on any case using their own
 * `OPENROUTER_API_KEY` or `OPENAI_API_KEY`.
 *
 * Defaults to a zero-cost dry-run inspection (printing the exact prompts, decoding parameters,
 * and binomial standard error expectations) when no API key is set.
 */
import {
  buildArm0SingleShotMcqPrompt,
  buildArm1AlfaChecklistMcqPrompt,
  buildArm3CbpRationaleStemMcqPrompt,
  buildArm4CbpFactored2x2McqPrompt,
  DECODING_HYPERPARAMETERS,
  enforcePredictionOnly2x2Contract,
  ensureLeakyDebateStem,
  PlainHumanPlanSchema,
} from "./reference-prompts"
import { auditPredictionOnly2x2Stem, computeEverydayClarity } from "./metrics"
import { evaluateCandidateMcqPsychometrics } from "./solver-harness"

const SAMPLE_VERBATIM_CASE = {
  id: "sample-replication-case-01",
  specialty: "Introductory Physics & Mechanics",
  sourcePassage:
    "A block of mass m2 = 2.0 kg rests on top of a larger block of mass m1 = 6.0 kg on a frictionless horizontal table. The coefficient of static friction between the two blocks is mu_s = 0.25 (use g = 10 m/s^2). A horizontal force F is applied directly to the lower block m1. Because static friction is the only horizontal force accelerating the top block m2, the maximum acceleration before slipping is a_max = mu_s * g = 2.5 m/s^2, which corresponds to a maximum applied force F_max = (m1 + m2) * a_max = (6.0 + 2.0) * 2.5 = 20.0 N.",
}

async function main() {
  const apiKey =
    process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || ""
  const isLiveFlag = process.argv.includes("--live")

  const prompts = [
    buildArm0SingleShotMcqPrompt(SAMPLE_VERBATIM_CASE),
    buildArm1AlfaChecklistMcqPrompt(SAMPLE_VERBATIM_CASE),
    buildArm3CbpRationaleStemMcqPrompt(SAMPLE_VERBATIM_CASE),
    buildArm4CbpFactored2x2McqPrompt(SAMPLE_VERBATIM_CASE),
  ]

  console.log(
    "================================================================================"
  )
  console.log(
    " LIVE REPLICATION HARNESS (Arms 0, 1, 3, 4 Contrastive Boundary Pinning Prompts)"
  )
  console.log(
    "================================================================================"
  )
  console.log(
    `Producer Model: ${DECODING_HYPERPARAMETERS.producerModel} | Evaluator Model: ${DECODING_HYPERPARAMETERS.evaluatorModel} (temp=0)`
  )
  console.log(
    `Expected Binomial SE on N=60 MCQs at p=91.7%-95.0%: +/-${DECODING_HYPERPARAMETERS.binomialStandardError60McqPct}% (+/-1.9 items)\n`
  )

  if (!apiKey || !isLiveFlag) {
    console.log(
      "[Dry-Run Mode] No API call made ($0.00 spend). Loaded 4 reference prompt schemas:"
    )
    for (const p of prompts) {
      console.log(
        `  - ${p.armAlias} -> slug: ${p.armId} (temp=${p.temperature}, systemChars=${p.system.length})`
      )
    }
    console.log(
      "\nTo run a live API replication against OpenRouter or OpenAI, pass `--live` with your API key:"
    )
    console.log(
      "  OPENROUTER_API_KEY=sk-or-... npm run replicate:live -- --live"
    )
    return
  }

  const baseUrl = process.env.OPENROUTER_API_KEY
    ? "https://openrouter.ai/api/v1"
    : "https://api.openai.com/v1"
  const producerModel = process.env.OPENROUTER_API_KEY
    ? DECODING_HYPERPARAMETERS.producerModel
    : "gpt-4o-mini"
  const evaluatorModel = process.env.OPENROUTER_API_KEY
    ? DECODING_HYPERPARAMETERS.evaluatorModel
    : "gpt-4o-mini"

  for (const p of prompts) {
    console.log(`\n[Live Replication] Generating ${p.armAlias} (${p.armId})...`)
    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: producerModel,
        temperature: p.temperature,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: p.system },
          { role: "user", content: p.user },
        ],
      }),
    })
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`)
    }
    const payload = (await res.json()) as any
    const data = JSON.parse(
      String(payload?.choices?.[0]?.message?.content ?? "{}")
    )
    const plan = PlainHumanPlanSchema(data)
    let stem = String(data.questionStem || data.stem || "")
    let correctOption = String(data.correctOption || data.correctAnswer || "")
    let distractors: [string, string, string] = [
      String(
        data.distractors?.[0] ||
          plan.threeCommonMistakes[0].wrongChoiceText ||
          ""
      ),
      String(
        data.distractors?.[1] ||
          plan.threeCommonMistakes[1].wrongChoiceText ||
          ""
      ),
      String(
        data.distractors?.[2] ||
          plan.threeCommonMistakes[2].wrongChoiceText ||
          ""
      ),
    ]

    if (p.armId === "arm-3-cbp-rationale-stem") {
      stem = ensureLeakyDebateStem({
        questionStem: stem,
        whatActuallyHappens: plan.whatActuallyHappens,
        firstMistakeReason: plan.threeCommonMistakes[0].whySomeonePicksIt,
      })
    } else if (p.armId === "arm-4-cbp-factored-2x2") {
      const enforced = enforcePredictionOnly2x2Contract({
        questionStem: stem,
        speaker1PredictionOnly: String(data.speaker1PredictionOnly || ""),
        speaker2PredictionOnly: String(data.speaker2PredictionOnly || ""),
        correctOption,
        distractors,
        everydayExplanation: plan.everydayExplanation,
      })
      stem = enforced.questionStem
      correctOption = enforced.correctOption
      distractors = enforced.distractors
    }

    const audit = auditPredictionOnly2x2Stem({
      questionStem: stem,
      correctOption,
      distractors,
    })
    const clarity = computeEverydayClarity(stem)
    const psych = await evaluateCandidateMcqPsychometrics(
      {
        id: `${SAMPLE_VERBATIM_CASE.id}-${p.armId}`,
        sourcePassage: SAMPLE_VERBATIM_CASE.sourcePassage,
        questionStem: stem,
        correctOption,
        distractors,
      },
      { apiKey, baseUrl, model: evaluatorModel }
    )

    console.log(`  Stem: ${stem}`)
    console.log(
      `  ValidKey=${psych.coldSolve.validKey} | NoviceFoilBite=${psych.novice.noviceBitFoil} | D=${psych.discriminationD} | Factored2x2=${audit.isStrictPredictionOnly2x2} | Clarity=${clarity.everydayClarityScore}`
    )
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error("[replicate-live error]", err)
    process.exit(1)
  })
}
