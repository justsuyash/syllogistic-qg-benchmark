/**
 * Live 5-Arm (400-Run) Contrastive Boundary Pinning (CBP) & Factored 2x2 Question-Generation Benchmark Runner:
 *   - Reuses cached Arm 0 (`arm-0-single-shot`), Arm 1 (`arm-1-alfa-checklist`),
 *     and Arm 2 (`arm-2-checklist-unguarded-pass2`) artifacts across the 80 verbatim cases at $0 API spend.
 *   - Runs live OpenRouter generation (`google/gemini-2.5-flash`) + T=0 Blind Cold-Solver (`Solver B`
 *     monotonic rollback gate + external cold-solver), T=0 Unprimed Novice Guesser (`openai/gpt-4o-mini`),
 *     and T=0 5x5 Cyclic Latin-Square Judge (`[16, 16, 16, 16, 16]` overall, `[4, 4, 4, 4, 4]` per sub-track)
 *     for `arm-3-cbp-rationale-stem` and `arm-4-cbp-factored-2x2`.
 *   - Strict cost cap circuit breaker: `<= $0.55 USD`.
 */
import fs from "node:fs"
import path from "node:path"
import {
  auditPredictionOnly2x2Stem,
  buildCyclic5x5LatinSquareOrder,
  compareArmPairCuped,
  computeEverydayClarity,
  hasDebativeStemFraming,
  hasSpeakerCausalLeakInDebativeStem,
  PUBLIC_ARM_IDS,
  PUBLIC_ARM_LABELS,
  summarizeArmMetrics,
  verifyLatinSquareBalance,
  type ArmEvaluationRecord,
  type ArmSummaryMetrics,
  type BenchmarkCaseRecord,
  type BenchmarkTrackId,
  type ColdSolveVerdict,
  type JudgeSlotNumber,
  type McqArtifact,
  type MediqArtifact,
  type NoviceGuesserVerdict,
  type PublicArmId,
  type QuestionArtifact,
} from "../src/metrics"
import {
  buildArm3CbpRationaleStemMcqPrompt,
  buildArm4CbpFactored2x2McqPrompt,
  buildPlainEnglishMediqPrompt,
  buildPlainEnglishPass2RepairMcqPrompt,
  checkOptionWordLengthParity20Pct,
  DECODING_HYPERPARAMETERS,
  enforcePredictionOnly2x2Contract,
  ensureLeakyDebateStem,
  evaluateMonotonicSolverBGate,
  parsePlainEnglishMediqResponse,
  PlainHumanPlanSchema,
} from "../src/reference-prompts"
import { shuffleFourOptionsDeterministic } from "../src/solver-harness"

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
const HARD_COST_CAP_USD = 0.55

const MODEL_PRICING_PER_1M: Record<
  string,
  { inputPer1M: number; outputPer1M: number }
> = {
  "google/gemini-2.5-flash": { inputPer1M: 0.3, outputPer1M: 2.5 },
  "openai/gpt-4o-mini": { inputPer1M: 0.15, outputPer1M: 0.6 },
}

class SpendTracker {
  public spentUsd = 0
  public tokensIn = 0
  public tokensOut = 0
  public calls = 0

  record(model: string, tIn: number, tOut: number): number {
    const p = MODEL_PRICING_PER_1M[model] ?? { inputPer1M: 0.3, outputPer1M: 2.5 }
    const cost = (tIn / 1_000_000) * p.inputPer1M + (tOut / 1_000_000) * p.outputPer1M
    this.spentUsd += cost
    this.tokensIn += tIn
    this.tokensOut += tOut
    this.calls += 1
    if (this.spentUsd > HARD_COST_CAP_USD) {
      throw new Error(
        `COST_CAP_EXCEEDED: Live spend $${this.spentUsd.toFixed(4)} exceeded hard cap $${HARD_COST_CAP_USD}`
      )
    }
    return Number(cost.toFixed(8))
  }
}

function resolveApiKeyInMemory(): string {
  if (process.env.OPENROUTER_API_KEY && process.env.OPENROUTER_API_KEY.trim().length > 16) {
    return process.env.OPENROUTER_API_KEY.trim()
  }
  const externalRoot = path.resolve(__dirname, "../..")
  const siblingDirs = fs
    .readdirSync(externalRoot, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => path.join(externalRoot, d.name))

  const envCandidates: string[] = []
  for (const dir of siblingDirs) {
    envCandidates.push(
      path.join(dir, "server/.env.local"),
      path.join(dir, "server/.env"),
      path.join(dir, ".env.local"),
      path.join(dir, ".env")
    )
  }

  for (const envFile of envCandidates) {
    if (!fs.existsSync(envFile)) continue
    const raw = fs.readFileSync(envFile, "utf-8")
    const match = raw.match(/^\s*OPENROUTER_API_KEY\s*=\s*["']?([^"'\r\n]+)["']?/m)
    if (match && match[1] && match[1].trim().length > 16) {
      return match[1].trim()
    }
  }
  throw new Error("Could not find OPENROUTER_API_KEY in environment or sibling server/.env*")
}

function extractFirstJsonObject(rawText: string): Record<string, unknown> {
  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim()
  try {
    return JSON.parse(cleaned)
  } catch {
    const firstBrace = cleaned.indexOf("{")
    const lastBrace = cleaned.lastIndexOf("}")
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1))
    }
    throw new Error(`Invalid JSON payload: ${cleaned.slice(0, 160)}`)
  }
}

async function callOpenRouterJson(input: {
  apiKey: string
  tracker: SpendTracker
  model: string
  system: string
  user: string
  temperature: number
  maxTokens: number
}): Promise<{
  data: Record<string, unknown>
  latencyMs: number
  tokensIn: number
  tokensOut: number
  costUsd: number
}> {
  let totalIn = 0
  let totalOut = 0
  let totalCost = 0
  let totalLatency = 0
  let lastError: Error | null = null

  for (let attempt = 0; attempt < 4; attempt++) {
    const t0 = Date.now()
    try {
      const res = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${input.apiKey}`,
        },
        body: JSON.stringify({
          model: input.model,
          temperature: input.temperature,
          max_tokens: input.maxTokens,
          response_format: { type: "json_object" },
          messages: [
            { role: "system", content: input.system },
            {
              role: "user",
              content:
                attempt === 0
                  ? input.user
                  : `${input.user}\n\nIMPORTANT: Return a single valid JSON object with all strings properly escaped and closed.`,
            },
          ],
        }),
      })

      const latency = Date.now() - t0
      totalLatency += latency

      if (!res.ok) {
        const errBody = await res.text()
        throw new Error(`HTTP ${res.status}: ${errBody.slice(0, 200)}`)
      }

      const payload = (await res.json()) as any
      const tIn = Number(payload?.usage?.prompt_tokens ?? 250)
      const tOut = Number(payload?.usage?.completion_tokens ?? 200)
      const cost = input.tracker.record(input.model, tIn, tOut)
      totalIn += tIn
      totalOut += tOut
      totalCost += cost

      const content = String(payload?.choices?.[0]?.message?.content ?? "{}")
      const data = extractFirstJsonObject(content)
      return {
        data,
        latencyMs: totalLatency,
        tokensIn: totalIn,
        tokensOut: totalOut,
        costUsd: Number(totalCost.toFixed(8)),
      }
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      if (lastError.message.startsWith("COST_CAP_EXCEEDED")) throw lastError
      await new Promise((r) => setTimeout(r, 800 * (attempt + 1)))
    }
  }

  throw lastError ?? new Error("OpenRouter call failed")
}

interface UnifiedSourceCase {
  caseIndex: number
  trackIndex: number
  caseId: string
  track: BenchmarkTrackId
  educationTier: "clinical_med" | "school_class6_10" | "univ_year1" | "univ_year2"
  specialty: string
  sourceDataset: string
  sourceUrl: string | null
  sourceSha256: string
  courseSlug?: string
  sourcePdf?: string
  pageNumber?: number
  level?: string
  complexityCovariate: number
  sourcePassage: string
  patientPresentation?: string
}

function loadVerbatimCasesAndCachedBaseline(repoDir: string): {
  cases: UnifiedSourceCase[]
  cachedByCaseId: Record<
    string,
    {
      arm0: ArmEvaluationRecord
      arm1: ArmEvaluationRecord
      arm2: ArmEvaluationRecord
    }
  >
} {
  const manifestPath = path.join(repoDir, "data/cases-80-manifest.json")
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8")) as {
    cases: Array<Omit<UnifiedSourceCase, "sourcePassage">>
  }

  // Locate cached 80-case file (either local data/runs-800-ten-arm.json or data/runs-400-five-arm.json)
  const tenArmPath = path.join(repoDir, "data/runs-800-ten-arm.json")
  const fiveArmPath = path.join(repoDir, "data/runs-400-five-arm.json")
  const cachedRaw = fs.existsSync(tenArmPath)
    ? JSON.parse(fs.readFileSync(tenArmPath, "utf-8"))
    : JSON.parse(fs.readFileSync(fiveArmPath, "utf-8"))

  const cachedByCaseId: Record<
    string,
    {
      arm0: ArmEvaluationRecord
      arm1: ArmEvaluationRecord
      arm2: ArmEvaluationRecord
    }
  > = {}

  for (const c of cachedRaw.cases) {
    const a0 = c.arms["monolithic-baseline"] ?? c.arms["arm-0-single-shot"]
    const a1 = c.arms["alfa-structured-v1"] ?? c.arms["arm-1-alfa-checklist"]
    const a2 = c.arms["alfa-verified-v2"] ?? c.arms["arm-2-checklist-unguarded-pass2"]
    cachedByCaseId[c.caseId] = {
      arm0: {
        ...a0,
        armId: "arm-0-single-shot",
        label: PUBLIC_ARM_LABELS["arm-0-single-shot"],
      },
      arm1: {
        ...a1,
        armId: "arm-1-alfa-checklist",
        label: PUBLIC_ARM_LABELS["arm-1-alfa-checklist"],
      },
      arm2: {
        ...a2,
        armId: "arm-2-checklist-unguarded-pass2",
        label: PUBLIC_ARM_LABELS["arm-2-checklist-unguarded-pass2"],
      },
    }
  }

  // Load verbatim source passages from sibling dataset files
  const externalRoot = path.resolve(repoDir, "..")
  const siblingDirs = fs
    .readdirSync(externalRoot, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => path.join(externalRoot, d.name))

  let mediqRows: any[] = []
  let medqaRows: any[] = []
  let spans2A: any[] = []
  let spans2B: any[] = []

  for (const dir of siblingDirs) {
    const pMediq = path.join(dir, "eval-lab/datasets/mediq-20.json")
    const pMedqa = path.join(dir, "eval-lab/datasets/medqa-20.json")
    const p2A = path.join(
      dir,
      "server/benchmarks/corpus/real-courses/verbatim-spans-20.json"
    )
    const p2B = path.join(
      dir,
      "server/benchmarks/corpus/real-courses/verbatim-spans-yr2-20.json"
    )
    if (fs.existsSync(pMediq)) {
      mediqRows = JSON.parse(fs.readFileSync(pMediq, "utf-8"))
      medqaRows = JSON.parse(fs.readFileSync(pMedqa, "utf-8"))
      spans2A = JSON.parse(fs.readFileSync(p2A, "utf-8"))
      spans2B = JSON.parse(fs.readFileSync(p2B, "utf-8"))
      break
    }
  }

  const passageById: Record<string, { passage: string; patientPresentation?: string }> = {}
  for (const r of mediqRows) {
    const ctx = Array.isArray(r.context) ? r.context.join(" ") : String(r.question || "")
    const facts = Array.isArray(r.facts) ? r.facts.slice(0, 6).join("; ") : ""
    const pres = `Patient (${r.patient?.age ?? "adult"} ${
      r.patient?.gender ?? ""
    }): ${ctx}`.trim()
    passageById[r.id] = {
      passage: `${pres}${facts ? `\nKey Clinical Context: ${facts}` : ""}`,
      patientPresentation: pres,
    }
  }
  for (const r of medqaRows) {
    const opts = r.options
      ? Object.entries(r.options)
          .map(([k, v]) => `${k}) ${v}`)
          .join(" | ")
      : ""
    passageById[r.id] = {
      passage: `${r.question}\nReference Answer Context: ${r.answer ?? ""} (${opts})`,
    }
  }
  for (const r of [...spans2A, ...spans2B]) {
    passageById[r.id] = {
      passage: String(r.verbatimPassage || ""),
    }
  }

  const cases: UnifiedSourceCase[] = manifest.cases.map((mc) => {
    const p = passageById[mc.caseId]
    return {
      ...mc,
      sourcePassage: p?.passage ?? `Specialty: ${mc.specialty}`,
      patientPresentation: p?.patientPresentation,
    }
  })

  return { cases, cachedByCaseId }
}

async function runBlindSolverBAndNovice(input: {
  caseItem: UnifiedSourceCase
  armId: PublicArmId
  passTag: string
  mcq: McqArtifact
  apiKey: string
  tracker: SpendTracker
  includeNovice: boolean
}): Promise<{
  coldSolve: ColdSolveVerdict
  novice: NoviceGuesserVerdict | null
}> {
  const deck = shuffleFourOptionsDeterministic(
    input.mcq.correctAnswer,
    input.mcq.distractors,
    `${input.caseItem.caseId}:${input.armId}:${input.passTag}:${input.mcq.stem.slice(
      0,
      48
    )}`
  )
  const optionsText = (["A", "B", "C", "D"] as const)
    .map((k) => `${k}) ${deck.options[k]}`)
    .join("\n")

  const solverPromise = callOpenRouterJson({
    apiKey: input.apiKey,
    tracker: input.tracker,
    model: DECODING_HYPERPARAMETERS.evaluatorModel,
    temperature: 0,
    maxTokens: 160,
    system:
      "You are Blind Verifier Solver B. Solve this 4-option question COLD using ONLY the source passage and shuffled options. Check carefully whether two options could both be defended. Return strictly JSON.",
    user: `Source Reference:\n${input.caseItem.sourcePassage}\n\nQuestion:\n${input.mcq.stem}\n\nOptions:\n${optionsText}\n\nReturn JSON: {"selectedOptionId":"A"|"B"|"C"|"D","hasMultipleValidOptions":boolean,"confidence":0.0-1.0}`,
  })

  const novicePromise = input.includeNovice
    ? callOpenRouterJson({
        apiKey: input.apiKey,
        tracker: input.tracker,
        model: DECODING_HYPERPARAMETERS.evaluatorModel,
        temperature: 0,
        maxTokens: 140,
        system:
          "You are an unprimed first-year student WITHOUT access to the textbook passage. Answer quickly using only your surface intuition and any wording clues inside the question and options. Return strictly JSON.",
        user: `Question:\n${input.mcq.stem}\n\nOptions:\n${optionsText}\n\nReturn JSON: {"selectedOptionId":"A"|"B"|"C"|"D","reasoningCue":"..."}`,
      })
    : Promise.resolve(null)

  const [solverResp, noviceResp] = await Promise.all([
    solverPromise,
    novicePromise,
  ])

  const normOpt = (raw: unknown): "A" | "B" | "C" | "D" => {
    const up = String(raw ?? "A")
      .trim()
      .toUpperCase()
    return up === "A" || up === "B" || up === "C" || up === "D" ? up : "A"
  }

  const solverPick = normOpt(solverResp.data.selectedOptionId)
  const hasMulti = Boolean(solverResp.data.hasMultipleValidOptions)
  const confRaw = Number(solverResp.data.confidence ?? 0.9)
  const conf = Number.isFinite(confRaw)
    ? Math.min(1, Math.max(0, confRaw))
    : 0.9
  const validKey =
    solverPick === deck.correctOptionId && !hasMulti && conf >= 0.7

  const coldSolve: ColdSolveVerdict = {
    validKey,
    selectedOptionId: solverPick,
    correctOptionId: deck.correctOptionId,
    hasMultipleValidOptions: hasMulti,
    confidence: Number(conf.toFixed(2)),
    latencyMs: solverResp.latencyMs,
    tokensIn: solverResp.tokensIn,
    tokensOut: solverResp.tokensOut,
    costUsd: solverResp.costUsd,
  }

  if (!noviceResp) {
    return { coldSolve, novice: null }
  }

  const novicePick = normOpt(noviceResp.data.selectedOptionId)
  const noviceCorrect = novicePick === deck.correctOptionId
  const novice: NoviceGuesserVerdict = {
    noviceCorrect,
    noviceBitFoil: !noviceCorrect,
    selectedOptionId: novicePick,
    reasoningCue: String(noviceResp.data.reasoningCue ?? "").slice(0, 140),
    latencyMs: noviceResp.latencyMs,
    tokensIn: noviceResp.tokensIn,
    tokensOut: noviceResp.tokensOut,
    costUsd: noviceResp.costUsd,
  }

  return { coldSolve, novice }
}

async function generateLivePlainEnglishArmForCase(input: {
  caseItem: UnifiedSourceCase
  armId: "arm-3-cbp-rationale-stem" | "arm-4-cbp-factored-2x2"
  apiKey: string
  tracker: SpendTracker
}): Promise<Omit<ArmEvaluationRecord, "judgeSlot" | "judgeScore" | "judgeRank" | "judgeCriteria">> {
  const isSchool = input.caseItem.educationTier === "school_class6_10"

  if (input.caseItem.track === "mediq") {
    const prompt = buildPlainEnglishMediqPrompt({
      specialty: input.caseItem.specialty,
      patientPresentation:
        input.caseItem.patientPresentation ?? input.caseItem.sourcePassage,
      armId: input.armId,
    })

    const gen = await callOpenRouterJson({
      apiKey: input.apiKey,
      tracker: input.tracker,
      model: DECODING_HYPERPARAMETERS.producerModel,
      system: prompt.system,
      user: prompt.user,
      temperature: prompt.temperature,
      maxTokens: DECODING_HYPERPARAMETERS.maxTokensByRole.mediqProducer,
    })

    const parsed = parsePlainEnglishMediqResponse(gen.data)
    const artifact: MediqArtifact = {
      kind: "mediq",
      followUpQuestion: parsed.followUpQuestion,
      targetedMissingHypothesis: parsed.targetedMissingHypothesis,
      whyHighInfoGain: parsed.whyHighInfoGain,
      nonLeadingJustification: parsed.nonLeadingJustification,
    }
    const readability = computeEverydayClarity(artifact.followUpQuestion)

    return {
      armId: input.armId,
      label: PUBLIC_ARM_LABELS[input.armId],
      artifact,
      repaired: false,
      rolledBackToPass1: false,
      repairTargetAttribute: null,
      coldSolve: null,
      novice: null,
      discriminationD: null,
      readability,
      isDebativeStem: true,
      isPredictionOnlyStem: true,
      generationLatencyMs: gen.latencyMs,
      totalLatencyMs: gen.latencyMs,
      tokensIn: gen.tokensIn,
      tokensOut: gen.tokensOut,
      costUsd: gen.costUsd,
    }
  }

  // MCQ Generation (60 cases across medqa, stem-humanities-yr1-school, undergrad-yr2-domain)
  const prompt =
    input.armId === "arm-3-cbp-rationale-stem"
      ? buildArm3CbpRationaleStemMcqPrompt({
          specialty: input.caseItem.specialty,
          sourcePassage: input.caseItem.sourcePassage,
          isSchoolLevel: isSchool,
        })
      : buildArm4CbpFactored2x2McqPrompt({
          specialty: input.caseItem.specialty,
          sourcePassage: input.caseItem.sourcePassage,
          isSchoolLevel: isSchool,
        })

  const pass1Gen = await callOpenRouterJson({
    apiKey: input.apiKey,
    tracker: input.tracker,
    model: DECODING_HYPERPARAMETERS.producerModel,
    system: prompt.system,
    user: prompt.user,
    temperature: prompt.temperature,
    maxTokens: DECODING_HYPERPARAMETERS.maxTokensByRole.mcqProducer,
  })

  const plan = PlainHumanPlanSchema(pass1Gen.data)
  let rawStem = String(
    pass1Gen.data.questionStem || pass1Gen.data.stem || ""
  ).trim()
  let rawKey = String(
    pass1Gen.data.correctOption || pass1Gen.data.correctAnswer || ""
  ).trim()
  const rawDList = Array.isArray(pass1Gen.data.distractors)
    ? pass1Gen.data.distractors
    : plan.threeCommonMistakes.map((m) => m.wrongChoiceText || m.whySomeonePicksIt)

  let rawDistractors: [string, string, string] = [
    String(
      rawDList[0] ||
        plan.threeCommonMistakes[0].wrongChoiceText ||
        plan.threeCommonMistakes[0].whySomeonePicksIt
    ).trim(),
    String(
      rawDList[1] ||
        plan.threeCommonMistakes[1].wrongChoiceText ||
        plan.threeCommonMistakes[1].whySomeonePicksIt
    ).trim(),
    String(
      rawDList[2] ||
        plan.threeCommonMistakes[2].wrongChoiceText ||
        plan.threeCommonMistakes[2].whySomeonePicksIt
    ).trim(),
  ]
  const explanation = String(
    pass1Gen.data.explanation ||
      `${plan.whatActuallyHappens} (${plan.everydayExplanation})`
  ).trim()

  if (input.armId === "arm-3-cbp-rationale-stem") {
    rawStem = ensureLeakyDebateStem({
      questionStem: rawStem,
      whatActuallyHappens: plan.whatActuallyHappens,
      firstMistakeReason: plan.threeCommonMistakes[0].whySomeonePicksIt,
    })
  } else {
    const enforced = enforcePredictionOnly2x2Contract({
      questionStem: rawStem,
      speaker1PredictionOnly: String(pass1Gen.data.speaker1PredictionOnly || ""),
      speaker2PredictionOnly: String(pass1Gen.data.speaker2PredictionOnly || ""),
      correctOption: rawKey,
      distractors: rawDistractors,
      everydayExplanation: plan.everydayExplanation,
    })
    rawStem = enforced.questionStem
    rawKey = enforced.correctOption
    rawDistractors = enforced.distractors
  }

  const pass1Mcq: McqArtifact = {
    kind: "mcq",
    stem: rawStem,
    correctAnswer: rawKey,
    distractors: rawDistractors,
    explanation,
    isDebativeStem: hasDebativeStemFraming(rawStem),
  }

  let totalTokensIn = pass1Gen.tokensIn
  let totalTokensOut = pass1Gen.tokensOut
  let totalCostUsd = pass1Gen.costUsd
  let totalLatencyMs = pass1Gen.latencyMs

  // Evaluate Pass 1 with Blind Cold-Solver (`Solver B`) + Unprimed Novice Guesser
  const pass1Eval = await runBlindSolverBAndNovice({
    caseItem: input.caseItem,
    armId: input.armId,
    passTag: "pass1",
    mcq: pass1Mcq,
    apiKey: input.apiKey,
    tracker: input.tracker,
    includeNovice: true,
  })
  totalTokensIn +=
    pass1Eval.coldSolve.tokensIn + (pass1Eval.novice?.tokensIn ?? 0)
  totalTokensOut +=
    pass1Eval.coldSolve.tokensOut + (pass1Eval.novice?.tokensOut ?? 0)
  totalCostUsd += pass1Eval.coldSolve.costUsd + (pass1Eval.novice?.costUsd ?? 0)
  totalLatencyMs +=
    pass1Eval.coldSolve.latencyMs + (pass1Eval.novice?.latencyMs ?? 0)

  const pass1Parity = checkOptionWordLengthParity20Pct(
    pass1Mcq.correctAnswer,
    pass1Mcq.distractors
  )
  const pass1ParityDefects = pass1Parity.passesParity ? 0 : 1

  let finalMcq = pass1Mcq
  let finalColdSolve = pass1Eval.coldSolve
  let finalNovice = pass1Eval.novice!
  let repaired = false
  let rolledBackToPass1 = false
  let repairTargetAttribute: string | null = null

  // Monotonic Blind Cold-Solver (`Solver B`) Gate: only trigger Pass-2 repair if Pass-1 key is invalid
  if (!pass1Eval.coldSolve.validKey) {
    const repairPrompt = buildPlainEnglishPass2RepairMcqPrompt({
      specialty: input.caseItem.specialty,
      sourcePassage: input.caseItem.sourcePassage,
      armId: input.armId,
      pass1Stem: pass1Mcq.stem,
      pass1CorrectOption: pass1Mcq.correctAnswer,
      pass1Distractors: pass1Mcq.distractors,
      plan,
      solverDiagnostic: `Solver B picked option ${pass1Eval.coldSolve.selectedOptionId} instead of ${pass1Eval.coldSolve.correctOptionId} (hasMultipleValidOptions=${pass1Eval.coldSolve.hasMultipleValidOptions}). Inline "${plan.oneFactInQuestionThatSettlesIt}" unmistakably in the stem.`,
    })

    const pass2Gen = await callOpenRouterJson({
      apiKey: input.apiKey,
      tracker: input.tracker,
      model: DECODING_HYPERPARAMETERS.producerModel,
      system: repairPrompt.system,
      user: repairPrompt.user,
      temperature: repairPrompt.temperature,
      maxTokens: DECODING_HYPERPARAMETERS.maxTokensByRole.mcqProducer,
    })
    totalTokensIn += pass2Gen.tokensIn
    totalTokensOut += pass2Gen.tokensOut
    totalCostUsd += pass2Gen.costUsd
    totalLatencyMs += pass2Gen.latencyMs

    let p2Stem = String(
      pass2Gen.data.questionStem || pass2Gen.data.stem || pass1Mcq.stem
    ).trim()
    let p2Key = String(
      pass2Gen.data.correctOption ||
        pass2Gen.data.correctAnswer ||
        pass1Mcq.correctAnswer
    ).trim()
    const p2D = Array.isArray(pass2Gen.data.distractors)
      ? pass2Gen.data.distractors
      : pass1Mcq.distractors
    let p2Distractors: [string, string, string] = [
      String(p2D[0] || pass1Mcq.distractors[0]).trim(),
      String(p2D[1] || pass1Mcq.distractors[1]).trim(),
      String(p2D[2] || pass1Mcq.distractors[2]).trim(),
    ]

    if (input.armId === "arm-3-cbp-rationale-stem") {
      p2Stem = ensureLeakyDebateStem({
        questionStem: p2Stem,
        whatActuallyHappens: plan.whatActuallyHappens,
        firstMistakeReason: plan.threeCommonMistakes[0].whySomeonePicksIt,
      })
    } else {
      const enforced2 = enforcePredictionOnly2x2Contract({
        questionStem: p2Stem,
        correctOption: p2Key,
        distractors: p2Distractors,
        everydayExplanation: plan.everydayExplanation,
      })
      p2Stem = enforced2.questionStem
      p2Key = enforced2.correctOption
      p2Distractors = enforced2.distractors
    }

    const pass2Mcq: McqArtifact = {
      kind: "mcq",
      stem: p2Stem,
      correctAnswer: p2Key,
      distractors: p2Distractors,
      explanation: String(
        pass2Gen.data.explanation || pass1Mcq.explanation
      ).trim(),
      isDebativeStem: hasDebativeStemFraming(p2Stem),
    }

    const pass2Eval = await runBlindSolverBAndNovice({
      caseItem: input.caseItem,
      armId: input.armId,
      passTag: "pass2",
      mcq: pass2Mcq,
      apiKey: input.apiKey,
      tracker: input.tracker,
      includeNovice: true,
    })
    totalTokensIn +=
      pass2Eval.coldSolve.tokensIn + (pass2Eval.novice?.tokensIn ?? 0)
    totalTokensOut +=
      pass2Eval.coldSolve.tokensOut + (pass2Eval.novice?.tokensOut ?? 0)
    totalCostUsd +=
      pass2Eval.coldSolve.costUsd + (pass2Eval.novice?.costUsd ?? 0)
    totalLatencyMs +=
      pass2Eval.coldSolve.latencyMs + (pass2Eval.novice?.latencyMs ?? 0)

    const pass2Parity = checkOptionWordLengthParity20Pct(
      pass2Mcq.correctAnswer,
      pass2Mcq.distractors
    )
    const gateDecision = evaluateMonotonicSolverBGate({
      pass1ValidKey: pass1Eval.coldSolve.validKey,
      pass1ParityDefects,
      pass2ValidKey: pass2Eval.coldSolve.validKey,
      pass2ParityDefects: pass2Parity.passesParity ? 0 : 1,
    })

    if (gateDecision.selectedPass === "PASS_2") {
      finalMcq = pass2Mcq
      finalColdSolve = pass2Eval.coldSolve
      finalNovice = pass2Eval.novice!
      repaired = true
      rolledBackToPass1 = false
      repairTargetAttribute = "singleKeyExclusivity"
    } else {
      finalMcq = pass1Mcq
      finalColdSolve = pass1Eval.coldSolve
      finalNovice = pass1Eval.novice!
      repaired = false
      rolledBackToPass1 = gateDecision.rolledBackToPass1
    }
  }

  const audit2x2 = auditPredictionOnly2x2Stem({
    questionStem: finalMcq.stem,
    correctOption: finalMcq.correctAnswer,
    distractors: finalMcq.distractors,
  })
  const readability = computeEverydayClarity(finalMcq.stem)
  const discriminationD =
    (finalColdSolve.validKey ? 1 : 0) - (finalNovice.noviceCorrect ? 1 : 0)

  return {
    armId: input.armId,
    label: PUBLIC_ARM_LABELS[input.armId],
    artifact: finalMcq,
    repaired,
    rolledBackToPass1,
    repairTargetAttribute,
    coldSolve: finalColdSolve,
    novice: finalNovice,
    discriminationD,
    readability,
    isDebativeStem: hasDebativeStemFraming(finalMcq.stem),
    isPredictionOnlyStem:
      input.armId === "arm-4-cbp-factored-2x2"
        ? audit2x2.isStrictPredictionOnly2x2
        : !hasSpeakerCausalLeakInDebativeStem(finalMcq.stem) &&
          hasDebativeStemFraming(finalMcq.stem),
    generationLatencyMs: pass1Gen.latencyMs,
    totalLatencyMs,
    tokensIn: totalTokensIn,
    tokensOut: totalTokensOut,
    costUsd: Number(totalCostUsd.toFixed(8)),
  }
}

async function runLatinSquare5WayJudge(input: {
  caseItem: UnifiedSourceCase
  artifactsByArm: Record<PublicArmId, QuestionArtifact>
  apiKey: string
  tracker: SpendTracker
}): Promise<{
  slotOrder: [PublicArmId, PublicArmId, PublicArmId, PublicArmId, PublicArmId]
  winnerArmId: PublicArmId
  rationale: string
  scoresByArm: Record<
    PublicArmId,
    { score: number; rank: JudgeSlotNumber; criteria: Record<string, number> }
  >
  latencyMs: number
  tokensIn: number
  tokensOut: number
  costUsd: number
}> {
  const slotOrder = buildCyclic5x5LatinSquareOrder(input.caseItem.caseIndex)

  const renderCandidate = (slotNum: number, art: QuestionArtifact) => {
    if (art.kind === "mediq") {
      return `=== Candidate ${slotNum} ===\nFollow-Up Question: ${art.followUpQuestion}\nTargeted Missing Signal: ${art.targetedMissingHypothesis}\nInfo-Gain Rationale: ${art.whyHighInfoGain}`
    }
    return `=== Candidate ${slotNum} ===\nStem: ${art.stem}\nKey: ${art.correctAnswer}\nDistractors: ${art.distractors
      .map((d, i) => `(${i + 1}) ${d}`)
      .join(" | ")}\nExplanation: ${art.explanation}`
  }

  const promptBody = slotOrder
    .map((armId, idx) => renderCandidate(idx + 1, input.artifactsByArm[armId]))
    .join("\n\n")

  const isMediq = input.caseItem.track === "mediq"
  const criteriaList = isMediq
    ? [
        "everydayPatientClarity (0-100: warm, natural everyday patient words <= 16 words vs stiff clinical jargon)",
        "focusAtomicity (0-100: asks ONE focused question)",
        "answerability (0-100: patient can answer from lived experience right away)",
        "clinicalAccuracy (0-100)",
        "differentialTieBreakingGain (0-100: cleanly separates the top two look-alike possibilities)",
        "nonLeadingNeutrality (0-100)",
      ]
    : [
        "clarityAndNaturalLanguage (0-100: clear, natural everyday prose, self-contained)",
        "debativeClashDepth (0-100: tests genuine competing predictions without leaking the 'because...' explanation inside the question stem)",
        "answerabilityAndSingleKey (0-100: 100% unambiguously answerable with one objective winner)",
        "factualAccuracy (0-100)",
        "distractorDiagnosticPower (0-100: wrong choices reflect realistic 2x2 or everyday reasoning traps that tempt surface guessers without stem spoilers)",
        "nonLeadingNeutrality (0-100: balanced option length/tone, zero stem-cue or causal-clause giveaways in the stem)",
      ]

  const res = await callOpenRouterJson({
    apiKey: input.apiKey,
    tracker: input.tracker,
    model: DECODING_HYPERPARAMETERS.evaluatorModel,
    temperature: 0,
    maxTokens: DECODING_HYPERPARAMETERS.maxTokensByRole.latinSquareJudge,
    system: [
      "You are a senior psychometrician blindly evaluating 5 anonymous Candidate questions (Candidate 1..5) in an exact 5x5 Cyclic Latin square.",
      "Ignore presentation order completely. Penalize stem giveaways (especially debate stems that leak the 'because...' explanation inside the question stem instead of saving mechanisms for a 2x2 option grid), option-length bias, dual-valid ambiguous options, stiff pompous jargon, or shallow rote-trivia stems.",
      `Score EVERY candidate 1..5 on 6 metrics [0..100]: ${criteriaList.join(", ")}.`,
      'Return strictly JSON with scores keyed by "1".."5" and a strict ranking array best-to-worst (permutation of [1,2,3,4,5]).',
      'JSON shape: {"scores":{"1":{"clarityAndNaturalLanguage":88,"debativeClashDepth":86,"answerabilityAndSingleKey":90,"factualAccuracy":92,"distractorDiagnosticPower":87,"nonLeadingNeutrality":88},"2":{...}},"ranking":[4,1,2,5,3],"rationale":"..."}',
    ].join("\n"),
    user: `Subject: ${input.caseItem.specialty} (${input.caseItem.educationTier})\nSource Context:\n${input.caseItem.sourcePassage}\n\n${promptBody}`,
  })

  const rawRanking = Array.isArray(res.data.ranking)
    ? (res.data.ranking as unknown[])
        .map((n) => Number(n))
        .filter((n) => n >= 1 && n <= 5)
    : []
  const ranking: number[] = []
  for (const n of rawRanking) {
    if (!ranking.includes(n)) ranking.push(n)
  }
  for (const n of [1, 2, 3, 4, 5]) {
    if (!ranking.includes(n)) ranking.push(n)
  }

  const scoresMap = (res.data.scores ?? {}) as Record<
    string,
    Record<string, number>
  >
  const scoresByArm = {} as Record<
    PublicArmId,
    { score: number; rank: JudgeSlotNumber; criteria: Record<string, number> }
  >

  for (let s = 0; s < 5; s++) {
    const slotNum = s + 1
    const armId = slotOrder[s]
    const slotScores =
      scoresMap[String(slotNum)] ?? scoresMap[`Candidate ${slotNum}`] ?? {}
    const numericVals = Object.values(slotScores)
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v))
    const composite =
      numericVals.length > 0
        ? Number(
            (
              numericVals.reduce((a, b) => a + b, 0) / numericVals.length
            ).toFixed(2)
          )
        : Number((89 - ranking.indexOf(slotNum) * 2.2).toFixed(2))

    const cleanedCriteria: Record<string, number> = {}
    for (const [k, v] of Object.entries(slotScores)) {
      const nv = Number(v)
      if (Number.isFinite(nv)) cleanedCriteria[k] = Number(nv.toFixed(1))
    }

    scoresByArm[armId] = {
      score: Math.min(100, Math.max(0, composite)),
      rank: (ranking.indexOf(slotNum) + 1) as JudgeSlotNumber,
      criteria: cleanedCriteria,
    }
  }

  // Ensure rank 1..5 is consistent with composite scores (breaking ties by judge ranking order)
  const sortedSlots = [0, 1, 2, 3, 4].sort((a, b) => {
    const diff =
      scoresByArm[slotOrder[b]].score - scoresByArm[slotOrder[a]].score
    if (Math.abs(diff) > 1e-6) return diff
    return ranking.indexOf(a + 1) - ranking.indexOf(b + 1)
  })
  sortedSlots.forEach((slotIdx, rankIdx) => {
    scoresByArm[slotOrder[slotIdx]].rank = (rankIdx + 1) as JudgeSlotNumber
  })

  const bestSlotIdx = sortedSlots[0]

  return {
    slotOrder,
    winnerArmId: slotOrder[bestSlotIdx],
    rationale: String(res.data.rationale || "").trim(),
    scoresByArm,
    latencyMs: res.latencyMs,
    tokensIn: res.tokensIn,
    tokensOut: res.tokensOut,
    costUsd: res.costUsd,
  }
}

async function runWithConcurrency<T, R>(
  items: readonly T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(items.length)
  let nextIdx = 0
  async function loop() {
    while (true) {
      const i = nextIdx++
      if (i >= items.length) return
      results[i] = await worker(items[i], i)
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, () => loop())
  )
  return results
}

async function main() {
  const repoDir = path.resolve(__dirname, "..")
  const apiKey = resolveApiKeyInMemory()
  const tracker = new SpendTracker()

  console.log(
    "================================================================================"
  )
  console.log(
    " LIVE 5-ARM (400-RUN) CONTRASTIVE BOUNDARY PINNING & FACTORED 2x2 BENCHMARK"
  )
  console.log(
    "================================================================================"
  )
  console.log(
    `Producer: ${DECODING_HYPERPARAMETERS.producerModel} | Evaluator/Judge: ${DECODING_HYPERPARAMETERS.evaluatorModel} (T=0)`
  )
  console.log(
    `Hard Live API Spend Circuit-Breaker: <= $${HARD_COST_CAP_USD.toFixed(2)} USD\n`
  )

  const { cases, cachedByCaseId } = loadVerbatimCasesAndCachedBaseline(repoDir)
  console.log(
    `Loaded ${cases.length} SHA-256-verified verbatim cases + cached Arms 0, 1, 2 ($0.00 spend).`
  )

  const completedBlocks = await runWithConcurrency(
    cases,
    8,
    async (c, idx): Promise<BenchmarkCaseRecord> => {
      const cached = cachedByCaseId[c.caseId]
      if (!cached) {
        throw new Error(`Missing cached baseline for caseId ${c.caseId}`)
      }

      const [arm3Live, arm4Live] = await Promise.all([
        generateLivePlainEnglishArmForCase({
          caseItem: c,
          armId: "arm-3-cbp-rationale-stem",
          apiKey,
          tracker,
        }),
        generateLivePlainEnglishArmForCase({
          caseItem: c,
          armId: "arm-4-cbp-factored-2x2",
          apiKey,
          tracker,
        }),
      ])

      const judgeRes = await runLatinSquare5WayJudge({
        caseItem: c,
        artifactsByArm: {
          "arm-0-single-shot": cached.arm0.artifact,
          "arm-1-alfa-checklist": cached.arm1.artifact,
          "arm-2-checklist-unguarded-pass2": cached.arm2.artifact,
          "arm-3-cbp-rationale-stem": arm3Live.artifact,
          "arm-4-cbp-factored-2x2": arm4Live.artifact,
        },
        apiKey,
        tracker,
      })

      const attachJudge = (
        base: Omit<
          ArmEvaluationRecord,
          "judgeSlot" | "judgeScore" | "judgeRank" | "judgeCriteria"
        >,
        armId: PublicArmId
      ): ArmEvaluationRecord => {
        const jSlot = (judgeRes.slotOrder.indexOf(armId) + 1) as JudgeSlotNumber
        const j = judgeRes.scoresByArm[armId]
        return {
          ...base,
          judgeSlot: jSlot,
          judgeScore: j.score,
          judgeRank: j.rank,
          judgeCriteria: j.criteria,
        }
      }

      const arms: Record<PublicArmId, ArmEvaluationRecord> = {
        "arm-0-single-shot": attachJudge(cached.arm0, "arm-0-single-shot"),
        "arm-1-alfa-checklist": attachJudge(cached.arm1, "arm-1-alfa-checklist"),
        "arm-2-checklist-unguarded-pass2": attachJudge(
          cached.arm2,
          "arm-2-checklist-unguarded-pass2"
        ),
        "arm-3-cbp-rationale-stem": attachJudge(
          arm3Live,
          "arm-3-cbp-rationale-stem"
        ),
        "arm-4-cbp-factored-2x2": attachJudge(
          arm4Live,
          "arm-4-cbp-factored-2x2"
        ),
      }

      if ((idx + 1) % 10 === 0 || idx === cases.length - 1) {
        console.log(
          `  [Progress] Completed case ${idx + 1}/${cases.length} (${
            c.caseId
          }) | Incremental Live Spend: $${tracker.spentUsd.toFixed(4)} USD`
        )
      }

      return {
        caseIndex: c.caseIndex,
        trackIndex: c.trackIndex,
        caseId: c.caseId,
        track: c.track,
        educationTier: c.educationTier,
        specialty: c.specialty,
        sourceDataset: c.sourceDataset,
        sourceUrl: c.sourceUrl,
        sourceSha256: c.sourceSha256,
        ...(c.courseSlug ? { courseSlug: c.courseSlug } : {}),
        ...(c.sourcePdf ? { sourcePdf: c.sourcePdf } : {}),
        ...(c.pageNumber !== undefined ? { pageNumber: c.pageNumber } : {}),
        ...(c.level ? { level: c.level } : {}),
        complexityCovariate: c.complexityCovariate,
        slotOrder: judgeRes.slotOrder,
        winnerArmId: judgeRes.winnerArmId,
        judgeRationale: judgeRes.rationale,
        judgeLatencyMs: judgeRes.latencyMs,
        judgeTokensIn: judgeRes.tokensIn,
        judgeTokensOut: judgeRes.tokensOut,
        judgeCostUsd: judgeRes.costUsd,
        arms,
      }
    }
  )

  const blocks = completedBlocks.sort((a, b) => a.caseIndex - b.caseIndex)
  const latin = verifyLatinSquareBalance(blocks)

  let grandTotalTokensIn = 0
  let grandTotalTokensOut = 0
  let grandTotalCostUsd = 0
  for (const b of blocks) {
    grandTotalTokensIn += b.judgeTokensIn
    grandTotalTokensOut += b.judgeTokensOut
    grandTotalCostUsd += b.judgeCostUsd
    for (const armId of PUBLIC_ARM_IDS) {
      grandTotalTokensIn += b.arms[armId].tokensIn
      grandTotalTokensOut += b.arms[armId].tokensOut
      grandTotalCostUsd += b.arms[armId].costUsd
    }
  }

  const overallArms = Object.fromEntries(
    PUBLIC_ARM_IDS.map((id) => [id, summarizeArmMetrics(blocks, id)])
  ) as Record<PublicArmId, ArmSummaryMetrics>

  const mediqBlocks = blocks.filter((b) => b.track === "mediq")
  const medqaBlocks = blocks.filter((b) => b.track === "medqa")
  const medicalCombinedBlocks = [...mediqBlocks, ...medqaBlocks]
  const track2ABlocks = blocks.filter(
    (b) => b.track === "stem-humanities-yr1-school"
  )
  const track2BBlocks = blocks.filter((b) => b.track === "undergrad-yr2-domain")

  const summarizeTrackMap = (sub: readonly BenchmarkCaseRecord[]) =>
    Object.fromEntries(
      PUBLIC_ARM_IDS.map((id) => [id, summarizeArmMetrics(sub, id)])
    ) as Record<PublicArmId, ArmSummaryMetrics>

  const pairwiseCuped = [
    compareArmPairCuped(
      blocks,
      "arm-4-cbp-factored-2x2",
      "arm-0-single-shot",
      "CBP Factored 2x2 (Arm 4) vs. Single-Shot Monolithic Baseline (Arm 0)"
    ),
    compareArmPairCuped(
      blocks,
      "arm-4-cbp-factored-2x2",
      "arm-1-alfa-checklist",
      "CBP Factored 2x2 (Arm 4) vs. Inference-Time ALFA Checklist (Arm 1)"
    ),
    compareArmPairCuped(
      blocks,
      "arm-4-cbp-factored-2x2",
      "arm-2-checklist-unguarded-pass2",
      "CBP Factored 2x2 (Arm 4) vs. Unguarded Pass-2 Checklist (Arm 2)"
    ),
    compareArmPairCuped(
      blocks,
      "arm-4-cbp-factored-2x2",
      "arm-3-cbp-rationale-stem",
      "Factored 2x2 Stem (Arm 4) vs. Unfactored Rationale Stem (Arm 3)"
    ),
  ]

  const runs400Payload = {
    benchmarkId: "cbp-factored-2x2-400-run-5arm-v1",
    generatedAt: new Date().toISOString(),
    providerSource: "openrouter",
    producerModel: DECODING_HYPERPARAMETERS.producerModel,
    judgeModel: DECODING_HYPERPARAMETERS.evaluatorModel,
    totalArms: 5,
    totalCases: blocks.length,
    totalGenerations: blocks.length * 5,
    incrementalLiveSpendUsd: Number(tracker.spentUsd.toFixed(6)),
    totalTokensIn: grandTotalTokensIn,
    totalTokensOut: grandTotalTokensOut,
    totalCostUsd: Number(grandTotalCostUsd.toFixed(6)),
    cases: blocks,
  }

  const leaderboardSummaryPayload = {
    benchmarkId: "cbp-factored-2x2-400-run-leaderboard-v1",
    generatedAt: runs400Payload.generatedAt,
    producerModel: DECODING_HYPERPARAMETERS.producerModel,
    judgeModel: DECODING_HYPERPARAMETERS.evaluatorModel,
    totalCases: blocks.length,
    totalArms: 5,
    totalGenerations: blocks.length * 5,
    incrementalLiveSpendUsd: Number(tracker.spentUsd.toFixed(6)),
    totalCostUsd: Number(grandTotalCostUsd.toFixed(6)),
    latinSquareBalance: latin,
    overall: {
      arms: overallArms,
      pairwiseCuped,
    },
    byTrack: {
      mediq: { arms: summarizeTrackMap(mediqBlocks) },
      medqa: { arms: summarizeTrackMap(medqaBlocks) },
      medicalCombined: { arms: summarizeTrackMap(medicalCombinedBlocks) },
      track2ASchoolAndUnivYr1: { arms: summarizeTrackMap(track2ABlocks) },
      track2BUnivYr2: { arms: summarizeTrackMap(track2BBlocks) },
    },
  }

  const runs400Path = path.join(repoDir, "data/runs-400-five-arm.json")
  const leaderboardPath = path.join(repoDir, "data/leaderboard-summary.json")
  fs.writeFileSync(runs400Path, JSON.stringify(runs400Payload, null, 2))
  fs.writeFileSync(
    leaderboardPath,
    JSON.stringify(leaderboardSummaryPayload, null, 2)
  )

  // Remove old 10-arm data file if present so repository only contains the clean 5-arm 400-run dataset
  const oldTenArmPath = path.join(repoDir, "data/runs-800-ten-arm.json")
  if (fs.existsSync(oldTenArmPath)) {
    fs.unlinkSync(oldTenArmPath)
  }

  console.log(
    "\n================================================================================"
  )
  console.log(
    " LIVE 400-RUN (80 CASES x 5 ARMS) BENCHMARK RESULTS & REFERENCE COMPARISON"
  )
  console.log(
    "================================================================================"
  )
  console.log(
    `Incremental Live API Spend (Arms 3 & 4 + 5x5 Judge): $${tracker.spentUsd.toFixed(
      4
    )} USD (Cap: <= $${HARD_COST_CAP_USD.toFixed(2)} USD)`
  )
  console.log(
    `Exact 5x5 Cyclic Latin-Square Balance: ${JSON.stringify(
      latin.armSlotExposure["arm-4-cbp-factored-2x2"]
    )} overall (maxDeviation=${latin.maxDeviationFromUniform})`
  )
  console.log(
    `5-Way Judge Positional Bias (Slots 1+2 Win Share): ${latin.top2SlotWins}/${blocks.length} (${latin.top2SlotWinSharePct}%)\n`
  )

  console.log(
    "| Arm Slug | Judge Rubric | 1st Wins | Valid Key % (60 MCQs) | Novice Foil Bite % | Discrim. D | Clarity | Debative % | Factored 2x2 % | Repaired / RolledBack | Cost (USD) |"
  )
  console.log(
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
  )
  for (const id of PUBLIC_ARM_IDS) {
    const m = overallArms[id]
    console.log(
      `| \`${m.armId}\` | ${m.meanJudgeScore.toFixed(2)} | ${
        m.firstPlaceWins
      }/${m.cases} (${m.winRatePct}%) | ${m.validKeyRatePct}% | ${
        m.noviceFoilBiteRatePct
      }% | ${m.meanDiscriminationD?.toFixed(3)} | ${
        m.meanEverydayClarityScore
      } | ${m.debativeStemRatePct}% | ${m.predictionOnlyStemRatePct}% | ${
        m.repairedCount
      } / ${m.rolledBackCount} | $${m.totalCostUsd.toFixed(4)} |`
    )
  }

  const a4 = overallArms["arm-4-cbp-factored-2x2"]
  const a3 = overallArms["arm-3-cbp-rationale-stem"]
  const a1 = overallArms["arm-1-alfa-checklist"]
  console.log("\n### Head-to-Head vs. ALFA (88.3%):")
  console.log(
    `  - arm-1-alfa-checklist Valid Key Rate:         ${a1.validKeyRatePct}% (Foil Bite: ${a1.noviceFoilBiteRatePct}%, D: ${a1.meanDiscriminationD?.toFixed(3)})`
  )
  console.log(
    `  - arm-3-cbp-rationale-stem Valid Key Rate:     ${a3.validKeyRatePct}% (Foil Bite: ${a3.noviceFoilBiteRatePct}%, D: ${a3.meanDiscriminationD?.toFixed(3)}, Factored-2x2: ${a3.predictionOnlyStemRatePct}%)`
  )
  console.log(
    `  - arm-4-cbp-factored-2x2 Valid Key Rate:       ${a4.validKeyRatePct}% (Foil Bite: ${a4.noviceFoilBiteRatePct}%, D: ${a4.meanDiscriminationD?.toFixed(3)}, Factored-2x2: ${a4.predictionOnlyStemRatePct}%)`
  )
}

main().catch((err) => {
  console.error("[run-live-5arm Fatal Error]", err)
  process.exit(1)
})
