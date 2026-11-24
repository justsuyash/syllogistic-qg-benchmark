/**
 * Pure TypeScript Psychometric, Linguistic, 5x5 Cyclic Latin-Square & CUPED Statistical Functions
 * for the 400-Run, 5-Arm Diagnostic Question-Generation Benchmark (80 Cases x 5 Arms):
 * Contrastive Boundary Pinning (CBP) and Factored 2x2 Stems.
 */

export type PublicArmId =
  | "arm-0-single-shot"
  | "arm-1-alfa-checklist"
  | "arm-2-checklist-unguarded-pass2"
  | "arm-3-cbp-rationale-stem"
  | "arm-4-cbp-factored-2x2"

export const PUBLIC_ARM_IDS: readonly PublicArmId[] = [
  "arm-0-single-shot",
  "arm-1-alfa-checklist",
  "arm-2-checklist-unguarded-pass2",
  "arm-3-cbp-rationale-stem",
  "arm-4-cbp-factored-2x2",
]

export const PUBLIC_ARM_LABELS: Record<PublicArmId, string> = {
  "arm-0-single-shot": "Arm 0 · Single-Shot Monolithic Baseline",
  "arm-1-alfa-checklist":
    "Arm 1 · Inference-Time ALFA Checklist (arXiv:2502.14860)",
  "arm-2-checklist-unguarded-pass2":
    "Arm 2 · ALFA Checklist + Unguarded Pass-2 Rewrite",
  "arm-3-cbp-rationale-stem":
    "Arm 3 · Contrastive Boundary Pinning — Unfactored Rationale Stem",
  "arm-4-cbp-factored-2x2":
    "Arm 4 · Contrastive Boundary Pinning — Factored 2×2 Stem",
}

export type BenchmarkTrackId =
  | "mediq"
  | "medqa"
  | "stem-humanities-yr1-school"
  | "undergrad-yr2-domain"

export type JudgeSlotNumber = 1 | 2 | 3 | 4 | 5

export interface ColdSolveVerdict {
  validKey: boolean
  selectedOptionId: "A" | "B" | "C" | "D"
  correctOptionId: "A" | "B" | "C" | "D"
  hasMultipleValidOptions: boolean
  confidence: number
  latencyMs: number
  tokensIn: number
  tokensOut: number
  costUsd: number
}

export interface NoviceGuesserVerdict {
  noviceCorrect: boolean
  noviceBitFoil: boolean
  selectedOptionId: "A" | "B" | "C" | "D"
  reasoningCue: string
  latencyMs: number
  tokensIn: number
  tokensOut: number
  costUsd: number
}

export interface ReadabilityMetrics {
  wordCount: number
  sentenceCount: number
  wordsPerSentence: number
  avgWordLengthChars: number
  jargonTokenCount: number
  everydayClarityScore: number
}

export interface McqArtifact {
  kind: "mcq"
  stem: string
  correctAnswer: string
  distractors: [string, string, string]
  explanation: string
  isDebativeStem?: boolean
}

export interface MediqArtifact {
  kind: "mediq"
  followUpQuestion: string
  targetedMissingHypothesis: string
  whyHighInfoGain: string
  nonLeadingJustification?: string
}

export type QuestionArtifact = McqArtifact | MediqArtifact

export interface ArmEvaluationRecord {
  armId: PublicArmId
  label: string
  artifact: QuestionArtifact
  repaired: boolean
  rolledBackToPass1: boolean
  repairTargetAttribute: string | null
  coldSolve: ColdSolveVerdict | null
  novice: NoviceGuesserVerdict | null
  discriminationD: number | null
  readability: ReadabilityMetrics
  isDebativeStem: boolean
  isPredictionOnlyStem: boolean
  generationLatencyMs: number
  totalLatencyMs: number
  tokensIn: number
  tokensOut: number
  costUsd: number
  judgeSlot: JudgeSlotNumber
  judgeScore: number
  judgeRank: JudgeSlotNumber
  judgeCriteria: Record<string, number>
}

export interface BenchmarkCaseRecord {
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
  slotOrder: [
    PublicArmId,
    PublicArmId,
    PublicArmId,
    PublicArmId,
    PublicArmId,
  ]
  winnerArmId: PublicArmId
  judgeRationale: string
  judgeLatencyMs: number
  judgeTokensIn: number
  judgeTokensOut: number
  judgeCostUsd: number
  arms: Record<PublicArmId, ArmEvaluationRecord>
}

export interface ArmSummaryMetrics {
  armId: PublicArmId
  label: string
  cases: number
  mcqCases: number
  meanJudgeScore: number
  firstPlaceWins: number
  winRatePct: number
  validKeyRatePct: number | null
  noviceFoilBiteRatePct: number | null
  meanDiscriminationD: number | null
  debativeStemRatePct: number | null
  predictionOnlyStemRatePct: number | null
  meanEverydayClarityScore: number
  meanWordsPerSentence: number
  meanJargonCount: number
  repairedCount: number
  rolledBackCount: number
  p50LatencyMs: number
  p95LatencyMs: number
  totalTokensIn: number
  totalTokensOut: number
  totalCostUsd: number
  validHighDCount: number
  validHighDPerDollar: number | null
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0
  return values.reduce((acc, v) => acc + v, 0) / values.length
}

export function sampleVariance(values: readonly number[]): number {
  if (values.length < 2) return 0
  const m = mean(values)
  const sq = values.reduce((acc, v) => acc + (v - m) ** 2, 0)
  return sq / (values.length - 1)
}

export function sampleCovariance(xs: readonly number[], ys: readonly number[]): number {
  const n = Math.min(xs.length, ys.length)
  if (n < 2) return 0
  const mx = mean(xs.slice(0, n))
  const my = mean(ys.slice(0, n))
  let sum = 0
  for (let i = 0; i < n; i++) {
    sum += (xs[i] - mx) * (ys[i] - my)
  }
  return sum / (n - 1)
}

/**
 * 1. ASD-STE100 Register-Decoupled Clarity & Readability Scorer
 */
const POMPOUS_JARGON_REGEX =
  /\b(?:epistemological|ontological|teleological|axiomatic|hermeneutic|phenomenological|canonically|aforementioned|heretofore|notwithstanding|vis-a-vis|ipso\s+facto|mutatis\s+mutandis|orthogonal\s+reification)\b/gi

export function computeEverydayClarity(text: string): ReadabilityMetrics {
  const cleaned = text.replace(/\s+/g, " ").trim()
  const words = cleaned.length > 0 ? cleaned.split(/\s+/) : []
  const wordCount = words.length
  const sentences = cleaned
    .split(/[.!?]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
  const sentenceCount = Math.max(1, sentences.length)
  const wordsPerSentence = Number((wordCount / sentenceCount).toFixed(2))
  const totalChars = words.reduce(
    (s, w) => s + w.replace(/[^a-zA-Z0-9]/g, "").length,
    0
  )
  const avgWordLengthChars =
    wordCount > 0 ? Number((totalChars / wordCount).toFixed(2)) : 0
  const jargonMatches = cleaned.match(POMPOUS_JARGON_REGEX) ?? []
  const jargonTokenCount = jargonMatches.length

  const lengthPenalty = Math.max(0, (wordsPerSentence - 18) * 1.1)
  const wordComplexityPenalty = Math.max(0, (avgWordLengthChars - 5.2) * 4.5)
  const jargonPenalty = jargonTokenCount * 6.5
  const rawScore = 98 - lengthPenalty - wordComplexityPenalty - jargonPenalty
  const everydayClarityScore = Number(
    Math.min(99, Math.max(55, rawScore)).toFixed(1)
  )

  return {
    wordCount,
    sentenceCount,
    wordsPerSentence,
    avgWordLengthChars,
    jargonTokenCount,
    everydayClarityScore,
  }
}

/**
 * 2. Factored 2x2 (Prediction-Only) Debative Stem Auditor
 */
const DEBATIVE_STEM_CUE_REGEX =
  /\b(?:speaker\s*[12ab]|student\s*[12ab]|colleague\s*[12ab]|claim\s*[12ab]|engineer\s*[12ab]|doctor\s*[12ab]|clinician\s*[12ab]|analyst\s*[12ab]|historian\s*[12ab]|researcher\s*[12ab]|person\s*[12ab]|classmate\s*[12ab]|argues\s+that|counters\s+that|insists\s+that|predicts\s+that|disagrees\s+and\s+says|who\s+is\s+right)\b/i

const SPEAKER_CAUSAL_LEAK_IN_STEM_REGEX =
  /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|2|a|b|one|two)\s+(?:predicts|says|argues|insists|claims|thinks|believes|counters)[^.?!]{2,160}?\b(?:because|since|due\s+to|as\s+a\s+result\s+of|on\s+the\s+grounds\s+that|by\s+applying)\b/i

export function hasDebativeStemFraming(stem: string): boolean {
  return DEBATIVE_STEM_CUE_REGEX.test(stem)
}

export function hasSpeakerCausalLeakInDebativeStem(stem: string): boolean {
  return SPEAKER_CAUSAL_LEAK_IN_STEM_REGEX.test(stem)
}

export interface PredictionOnly2x2AuditResult {
  hasDebativeClash: boolean
  zeroCausalLeakInStem: boolean
  has2x2OptionBalance: boolean
  optionsHaveMechanismClause: boolean
  isStrictPredictionOnly2x2: boolean
}

export function auditPredictionOnly2x2Stem(input: {
  questionStem: string
  correctOption: string
  distractors: [string, string, string]
}): PredictionOnly2x2AuditResult {
  const hasDebativeClash = hasDebativeStemFraming(input.questionStem)
  const zeroCausalLeakInStem = !hasSpeakerCausalLeakInDebativeStem(input.questionStem)

  const allOptions = [input.correctOption, ...input.distractors]
  const speaker1Count = allOptions.filter((opt) =>
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|a|one|first)\b/i.test(
      opt
    )
  ).length
  const speaker2Count = allOptions.filter((opt) =>
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:2|b|two|second)\b/i.test(
      opt
    )
  ).length
  const has2x2OptionBalance = speaker1Count === 2 && speaker2Count === 2
  const optionsHaveMechanismClause = allOptions.every((opt) =>
    /\b(?:because|since|as|due\s+to)\b/i.test(opt)
  )

  return {
    hasDebativeClash,
    zeroCausalLeakInStem,
    has2x2OptionBalance,
    optionsHaveMechanismClause,
    isStrictPredictionOnly2x2:
      hasDebativeClash &&
      zeroCausalLeakInStem &&
      has2x2OptionBalance &&
      optionsHaveMechanismClause,
  }
}

/**
 * 3. Summary Table Aggregator for Any Subset of Cases
 */
export function summarizeArmMetrics(
  blocks: readonly BenchmarkCaseRecord[],
  armId: PublicArmId
): ArmSummaryMetrics {
  const recs = blocks.map((b) => b.arms[armId]).filter(Boolean)
  const mcqRecs = recs.filter((r) => r.coldSolve != null && r.novice != null)
  const validKeys = mcqRecs.filter((r) => r.coldSolve!.validKey).length
  const foilBites = mcqRecs.filter((r) => r.novice!.noviceBitFoil).length
  const validHighD = mcqRecs.filter(
    (r) => r.coldSolve!.validKey && r.novice!.noviceBitFoil
  ).length
  const debativeCount = mcqRecs.filter((r) => r.isDebativeStem).length
  const predOnlyCount = mcqRecs.filter((r) => r.isPredictionOnlyStem).length
  const dVals = mcqRecs
    .map((r) => r.discriminationD)
    .filter((v): v is number => v !== null)

  const genLatencies = recs
    .map((r) => r.generationLatencyMs)
    .sort((a, b) => a - b)
  const p50Idx = Math.floor(genLatencies.length * 0.5)
  const p95Idx = Math.min(
    genLatencies.length - 1,
    Math.floor(genLatencies.length * 0.95)
  )
  const totalCostUsd = recs.reduce((s, r) => s + r.costUsd, 0)
  const totalTokensIn = recs.reduce((s, r) => s + r.tokensIn, 0)
  const totalTokensOut = recs.reduce((s, r) => s + r.tokensOut, 0)
  const wins = blocks.filter((b) => b.winnerArmId === armId).length

  return {
    armId,
    label: PUBLIC_ARM_LABELS[armId],
    cases: recs.length,
    mcqCases: mcqRecs.length,
    meanJudgeScore: Number(mean(recs.map((r) => r.judgeScore)).toFixed(2)),
    firstPlaceWins: wins,
    winRatePct: Number(((wins / Math.max(1, blocks.length)) * 100).toFixed(1)),
    validKeyRatePct:
      mcqRecs.length > 0
        ? Number(((validKeys / mcqRecs.length) * 100).toFixed(1))
        : null,
    noviceFoilBiteRatePct:
      mcqRecs.length > 0
        ? Number(((foilBites / mcqRecs.length) * 100).toFixed(1))
        : null,
    meanDiscriminationD:
      dVals.length > 0 ? Number(mean(dVals).toFixed(3)) : null,
    debativeStemRatePct:
      mcqRecs.length > 0
        ? Number(((debativeCount / mcqRecs.length) * 100).toFixed(1))
        : null,
    predictionOnlyStemRatePct:
      mcqRecs.length > 0
        ? Number(((predOnlyCount / mcqRecs.length) * 100).toFixed(1))
        : null,
    meanEverydayClarityScore:
      recs.length > 0
        ? Number(
            mean(recs.map((r) => r.readability.everydayClarityScore)).toFixed(1)
          )
        : 0,
    meanWordsPerSentence:
      recs.length > 0
        ? Number(mean(recs.map((r) => r.readability.wordsPerSentence)).toFixed(1))
        : 0,
    meanJargonCount:
      recs.length > 0
        ? Number(
            mean(recs.map((r) => r.readability.jargonTokenCount)).toFixed(2)
          )
        : 0,
    repairedCount: recs.filter((r) => r.repaired).length,
    rolledBackCount: recs.filter((r) => r.rolledBackToPass1).length,
    p50LatencyMs: genLatencies[p50Idx] ?? 0,
    p95LatencyMs: genLatencies[p95Idx] ?? 0,
    totalTokensIn,
    totalTokensOut,
    totalCostUsd: Number(totalCostUsd.toFixed(6)),
    validHighDCount: validHighD,
    validHighDPerDollar:
      totalCostUsd > 0 && mcqRecs.length > 0
        ? Number((validHighD / totalCostUsd).toFixed(1))
        : null,
  }
}

/**
 * 4. Exact 5x5 Cyclic Latin-Square Order & Balance Verifier
 *    Produces `[16, 16, 16, 16, 16]` slot appearances overall across 80 cases
 *    and `[4, 4, 4, 4, 4]` per 20-case sub-track.
 */
export function buildCyclic5x5LatinSquareOrder(
  caseIndex: number
): [PublicArmId, PublicArmId, PublicArmId, PublicArmId, PublicArmId] {
  const shift = ((caseIndex % 5) + 5) % 5
  return [
    PUBLIC_ARM_IDS[(0 + shift) % 5],
    PUBLIC_ARM_IDS[(1 + shift) % 5],
    PUBLIC_ARM_IDS[(2 + shift) % 5],
    PUBLIC_ARM_IDS[(3 + shift) % 5],
    PUBLIC_ARM_IDS[(4 + shift) % 5],
  ]
}

export function verifyLatinSquareBalance(blocks: readonly BenchmarkCaseRecord[]) {
  const armSlotExposure = {} as Record<
    PublicArmId,
    [number, number, number, number, number]
  >
  for (const armId of PUBLIC_ARM_IDS) {
    armSlotExposure[armId] = [0, 0, 0, 0, 0]
  }

  const slotWinDistribution: Record<JudgeSlotNumber, number> = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
  }

  for (const b of blocks) {
    for (const armId of PUBLIC_ARM_IDS) {
      const slot = b.arms[armId].judgeSlot
      armSlotExposure[armId][slot - 1] += 1
    }
    const winSlot = b.arms[b.winnerArmId].judgeSlot
    slotWinDistribution[winSlot] += 1
  }

  const expectedCountPerSlot = blocks.length / PUBLIC_ARM_IDS.length
  let maxDeviationFromUniform = 0
  for (const armId of PUBLIC_ARM_IDS) {
    for (const count of armSlotExposure[armId]) {
      maxDeviationFromUniform = Math.max(
        maxDeviationFromUniform,
        Math.abs(count - expectedCountPerSlot)
      )
    }
  }

  const top2SlotWins = slotWinDistribution[1] + slotWinDistribution[2]
  const top2SlotWinSharePct = Number(
    ((top2SlotWins / Math.max(1, blocks.length)) * 100).toFixed(1)
  )

  return {
    totalBlocks: blocks.length,
    expectedCountPerSlot,
    maxDeviationFromUniform,
    isExactUniformLatinSquare: maxDeviationFromUniform === 0,
    armSlotExposure,
    slotWinDistribution,
    top2SlotWins,
    top2SlotWinSharePct,
  }
}

/**
 * 5. Pre-Treatment CUPED Variance-Reduced Paired Estimator
 */
export function tCritical95(df: number): number {
  if (df <= 1) return 12.706
  if (df === 2) return 4.303
  if (df === 3) return 3.182
  if (df === 4) return 2.776
  if (df === 5) return 2.571
  if (df === 6) return 2.447
  if (df === 7) return 2.365
  if (df === 8) return 2.306
  if (df === 9) return 2.262
  if (df === 10) return 2.228
  if (df <= 15) return 2.131
  if (df <= 20) return 2.093
  if (df <= 30) return 2.045
  if (df <= 45) return 2.017
  if (df <= 60) return 2.0
  if (df <= 120) return 1.99
  return 1.96
}

function logGamma(z: number): number {
  const c = [
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.001208650973866179,
    -0.000005395239384953,
  ]
  let x = z
  let y = z
  let tmp = x + 5.5
  tmp -= (x + 0.5) * Math.log(tmp)
  let ser = 1.000000000190015
  for (let j = 0; j < 6; j++) {
    y += 1
    ser += c[j] / y
  }
  return -tmp + Math.log((2.5066282746310005 * ser) / x)
}

function betacf(a: number, b: number, x: number): number {
  const MAXIT = 200
  const EPS = 3e-12
  const FPMIN = 1e-30
  const qab = a + b
  const qap = a + 1
  const qam = a - 1
  let c = 1
  let d = 1 - (qab * x) / qap
  if (Math.abs(d) < FPMIN) d = FPMIN
  d = 1 / d
  let h = d
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2))
    d = 1 + aa * d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = 1 + aa / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    h *= d * c
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
    d = 1 + aa * d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = 1 + aa / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    const del = d * c
    h *= del
    if (Math.abs(del - 1) < EPS) break
  }
  return h
}

function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  const bt = Math.exp(
    logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log(1 - x)
  )
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(a, b, x)) / a
  }
  return 1 - (bt * betacf(b, a, 1 - x)) / b
}

export function computeTwoSidedPValue(tStat: number, df: number): number {
  if (!Number.isFinite(tStat) || df <= 0) return 1
  const absT = Math.abs(tStat)
  if (absT === 0) return 1
  const x = df / (df + absT * absT)
  const p = regularizedIncompleteBeta(x, df / 2, 0.5)
  return Number(Math.min(1, Math.max(0, p)).toFixed(12))
}

export interface PairedCupedContrast {
  comparison: string
  hypothesisLabel: string
  n: number
  rawDelta: number
  cupedDelta: number
  ci95Low: number
  ci95High: number
  tStat: number
  pValue: number
}

export function compareArmPairCuped(
  blocks: readonly BenchmarkCaseRecord[],
  treatmentArm: PublicArmId,
  controlArm: PublicArmId,
  hypothesisLabel?: string
): PairedCupedContrast {
  const n = blocks.length
  const rawDiffs = blocks.map(
    (b) => b.arms[treatmentArm].judgeScore - b.arms[controlArm].judgeScore
  )
  const covariates = blocks.map((b) => b.complexityCovariate)
  const meanCov = mean(covariates)
  const varCov = sampleVariance(covariates)
  const covDiffX = sampleCovariance(rawDiffs, covariates)
  const theta = varCov > 1e-12 ? covDiffX / varCov : 0
  const cupedDiffs = rawDiffs.map((d, i) => d - theta * (covariates[i] - meanCov))

  const rawMeanDelta = mean(rawDiffs)
  const cupedMeanDelta = mean(cupedDiffs)
  const varD = sampleVariance(cupedDiffs)
  const stdErr = n > 0 ? Math.sqrt(varD / n) : 0
  const df = Math.max(1, n - 1)
  const tCrit = tCritical95(df)
  const half = tCrit * stdErr
  const tStat = stdErr > 1e-12 ? Number((cupedMeanDelta / stdErr).toFixed(3)) : 0

  return {
    comparison: `${treatmentArm} vs ${controlArm}`,
    hypothesisLabel: hypothesisLabel ?? `${treatmentArm} vs ${controlArm}`,
    n,
    rawDelta: Number(rawMeanDelta.toFixed(3)),
    cupedDelta: Number(cupedMeanDelta.toFixed(3)),
    ci95Low: Number((cupedMeanDelta - half).toFixed(3)),
    ci95High: Number((cupedMeanDelta + half).toFixed(3)),
    tStat,
    pValue: computeTwoSidedPValue(tStat, df),
  }
}
