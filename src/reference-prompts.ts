/**
 * Reference Prompt Schemas, Register-Decoupled Contrastive Boundary Pinning (CBP) Slots,
 * Factored 2x2 Stem Sanitizer (`sanitizePredictionOnlyStem`), and Monotonic Blind-Solver Gate
 * for the 400-Run (80 Verbatim Cases x 5 Arms) 5x5 Cyclic Latin-Square Question-Generation Benchmark.
 *
 * Implements the ASD-STE100 Simplified Technical English register-controlled realization
 * of Contrastive Boundary Pinning (0 checklist adjectives, 0 philosophy jargon, 0 internal codenames):
 *   1. Contrastive Confusion Locus (C)            -> `whyTwoAnswersLookRightAtFirst`
 *   2. Passage-Grounded Entailment (G)            -> `whatActuallyHappens`
 *   3. Register-Decoupled Invariant (R_STE)       -> `everydayExplanation`
 *   4. Eliminative Boundary Witness (w*)          -> `oneFactInQuestionThatSettlesIt`
 *   5. Structured Fault-Injection Taxonomy (F1..5)-> `threeCommonMistakes`
 */

export const DECODING_HYPERPARAMETERS = {
  producerModel: "google/gemini-2.5-flash",
  evaluatorModel: "openai/gpt-4o-mini",
  temperatureByRole: {
    arm0AndArm1Producer: 0.35,
    arm3AndArm4Pass1Producer: 0.25,
    arm3AndArm4Pass2RepairProducer: 0.2,
    internalBlindSolverB: 0.0,
    externalBlindColdSolver: 0.0,
    unprimedNoviceGuesser: 0.0,
    latinSquare5WayJudge: 0.0,
  },
  maxTokensByRole: {
    mediqProducer: 1100,
    mcqProducer: 2100,
    solverAndNovice: 250,
    latinSquareJudge: 1200,
  },
  binomialStandardError60McqPct: 3.2,
} as const

export const FIVE_PLAIN_MISTAKE_TYPES = [
  "TRUE_GENERALLY_DOESNT_PROVE_IT_HERE",
  "REVERSED_THE_DIRECTION_OR_SIGN",
  "MISSED_ONE_KEY_DETAIL_IN_SETUP",
  "USED_RULE_WHERE_REQUIREMENT_FAILS",
  "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS",
] as const

export type PlainMistakeType = (typeof FIVE_PLAIN_MISTAKE_TYPES)[number]

export interface PlainCommonMistakeSlot {
  mistakeType: PlainMistakeType
  whySomeonePicksIt: string
  exactDetailInQuestionThatBreaksIt: string
  wrongChoiceText?: string
}

export interface PlainHumanPlan {
  whyTwoAnswersLookRightAtFirst: string
  whatActuallyHappens: string
  everydayExplanation: string
  oneFactInQuestionThatSettlesIt: string
  threeCommonMistakes: [
    PlainCommonMistakeSlot,
    PlainCommonMistakeSlot,
    PlainCommonMistakeSlot,
  ]
}

export interface PlainMediqPlan {
  mostLikelyCondition: string
  closestLookalikeCondition: string
  oneSimpleQuestionToTellThemApart: string
  plainPatientQuestion: string
  whyThisAnswerSettlesIt: string
  whyItIsNonLeading: string
}

/**
 * Formal mapping between Register-Decoupled Contrastive Boundary Pinning (CBP)
 * theoretical constructs and their ASD-STE100 register-controlled JSON schema keys.
 */
export const PLAIN_HUMAN_PLAN_SLOT_DESCRIPTIONS = {
  whyTwoAnswersLookRightAtFirst:
    "Contrastive Confusion Locus (C): If someone reads fast, which two answers both look right at first glance and why do people mix them up? (<= 25 words)",
  whatActuallyHappens:
    "Passage-Grounded Entailment (G): Based strictly on the passage, what actually happens and why? (<= 35 words)",
  everydayExplanation:
    "Register-Decoupled Invariant (R_STE, ASD-STE100): Explain that rule in plain, normal words you'd say to a friend—no stiff textbook buzzwords that tip off a guesser. (<= 30 words)",
  oneFactInQuestionThatSettlesIt:
    "Eliminative Boundary Witness (w*): Name the exact number, lab result, line of code, or detail you are putting INSIDE the question so only one answer works and the other three are definitely wrong. (<= 25 words)",
  threeCommonMistakes:
    "Structured Fault-Injection Taxonomy (F1..F5): Pick 3 distinct mistakeType values from TRUE_GENERALLY_DOESNT_PROVE_IT_HERE, REVERSED_THE_DIRECTION_OR_SIGN, MISSED_ONE_KEY_DETAIL_IN_SETUP, USED_RULE_WHERE_REQUIREMENT_FAILS, and WENT_WITH_HABIT_AGAINST_GIVEN_FACTS, each with { whySomeonePicksIt, exactDetailInQuestionThatBreaksIt }.",
} as const

export function countWords(text: string): number {
  const clean = text.trim()
  if (!clean) return 0
  return clean.split(/\s+/).filter(Boolean).length
}

export function clampWords(text: string, maxWords: number): string {
  const words = text.trim().split(/\s+/).filter(Boolean)
  if (words.length <= maxWords) return words.join(" ")
  return words.slice(0, maxWords).join(" ").replace(/[,;:-]+$/, "")
}

function normalizeMistakeType(
  raw: unknown,
  fallbackIndex: number
): PlainMistakeType {
  const up = String(raw ?? "")
    .trim()
    .toUpperCase()
  for (const allowed of FIVE_PLAIN_MISTAKE_TYPES) {
    if (up === allowed) return allowed
  }
  if (up.includes("REVERSE") || up.includes("SIGN") || up.includes("BACKWARD")) {
    return "REVERSED_THE_DIRECTION_OR_SIGN"
  }
  if (up.includes("MISSED") || up.includes("DETAIL") || up.includes("TIE")) {
    return "MISSED_ONE_KEY_DETAIL_IN_SETUP"
  }
  if (up.includes("REQUIREMENT") || up.includes("CONDITION") || up.includes("START")) {
    return "USED_RULE_WHERE_REQUIREMENT_FAILS"
  }
  if (up.includes("HABIT") || up.includes("REFLEX")) {
    return "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS"
  }
  if (up.includes("GENERAL") || up.includes("TRUE")) {
    return "TRUE_GENERALLY_DOESNT_PROVE_IT_HERE"
  }
  return FIVE_PLAIN_MISTAKE_TYPES[fallbackIndex % FIVE_PLAIN_MISTAKE_TYPES.length]
}

/**
 * Runtime validator and normalizer for `PlainHumanPlanSchema` (Register-Decoupled CBP Schema).
 * Ensures all word caps and 3 distinct `mistakeType` values hold.
 */
export function PlainHumanPlanSchema(raw: Record<string, unknown>): PlainHumanPlan {
  const whyTwoAnswersLookRightAtFirst = clampWords(
    String(
      raw.whyTwoAnswersLookRightAtFirst ||
        "Two answers look plausible at a quick glance because they share the same surface topic."
    ),
    25
  )
  const whatActuallyHappens = clampWords(
    String(
      raw.whatActuallyHappens ||
        "The passage's core rule determines the single outcome that holds under the given setup."
    ),
    35
  )
  const everydayExplanation = clampWords(
    String(
      raw.everydayExplanation ||
        "When the key detail in the setup is fixed, it forces one clear result and rules out the shortcuts."
    ),
    30
  )
  const oneFactInQuestionThatSettlesIt = clampWords(
    String(
      raw.oneFactInQuestionThatSettlesIt ||
        "The exact number or condition stated directly in the question setup."
    ),
    25
  )

  const rawMistakes = Array.isArray(raw.threeCommonMistakes)
    ? raw.threeCommonMistakes
    : [{}, {}, {}]

  const usedTypes = new Set<PlainMistakeType>()
  const parsedMistakes: PlainCommonMistakeSlot[] = []

  for (let i = 0; i < 3; i++) {
    const item = (rawMistakes[i] ?? {}) as Record<string, unknown>
    let mType = normalizeMistakeType(item.mistakeType, i)
    if (usedTypes.has(mType)) {
      const unused = FIVE_PLAIN_MISTAKE_TYPES.find((t) => !usedTypes.has(t))
      if (unused) mType = unused
    }
    usedTypes.add(mType)
    parsedMistakes.push({
      mistakeType: mType,
      whySomeonePicksIt: clampWords(
        String(
          item.whySomeonePicksIt ||
            "Sounds familiar on a fast read if you skip one detail in the setup."
        ),
        28
      ),
      exactDetailInQuestionThatBreaksIt: clampWords(
        String(
          item.exactDetailInQuestionThatBreaksIt ||
            oneFactInQuestionThatSettlesIt
        ),
        25
      ),
      ...(item.wrongChoiceText
        ? { wrongChoiceText: String(item.wrongChoiceText).trim() }
        : {}),
    })
  }

  return {
    whyTwoAnswersLookRightAtFirst,
    whatActuallyHappens,
    everydayExplanation,
    oneFactInQuestionThatSettlesIt,
    threeCommonMistakes: [
      parsedMistakes[0],
      parsedMistakes[1],
      parsedMistakes[2],
    ],
  }
}

/**
 * 1. Arm 0 (`arm-0-single-shot`):
 *    Single-shot monolithic baseline MCQ prompt.
 */
export function buildArm0SingleShotMcqPrompt(input: {
  specialty: string
  sourcePassage: string
}) {
  return {
    armAlias: "arm-0-single-shot" as const,
    armId: "arm-0-single-shot" as const,
    temperature: DECODING_HYPERPARAMETERS.temperatureByRole.arm0AndArm1Producer,
    system: [
      `You are a ${input.specialty} assessment author.`,
      "Write 1 challenging multiple-choice question (1 correct answer and 3 plausible distractors) grounded strictly in the provided source passage.",
      'Return strictly JSON: {"stem":"...","correctAnswer":"...","distractors":["...","...","..."],"explanation":"..."}',
    ].join("\n"),
    user: `Subject: ${input.specialty}\nSource Passage:\n${input.sourcePassage}`,
  }
}

/**
 * 2. Arm 1 (`arm-1-alfa-checklist`):
 *    Inference-time 6-attribute ALFA checklist prompt (Li et al., arXiv:2502.14860).
 */
export function buildArm1AlfaChecklistMcqPrompt(input: {
  specialty: string
  sourcePassage: string
}) {
  return {
    armAlias: "arm-1-alfa-checklist" as const,
    armId: "arm-1-alfa-checklist" as const,
    temperature: DECODING_HYPERPARAMETERS.temperatureByRole.arm0AndArm1Producer,
    system: [
      `You are an expert ${input.specialty} psychometrician implementing inference-time ALFA attribute alignment (arXiv:2502.14860).`,
      "Condition your question generation on all 6 fine-grained ALFA quality attributes:",
      "1. Specificity & Grounding: Target a concrete mechanism in the passage rather than vague recall.",
      "2. Non-Leading Neutrality: Do not include grammatical cues or hints toward the correct answer in the stem.",
      "3. Answerability & Self-Containment: Include all required context in the stem so a domain expert can solve it.",
      "4. Single-Key Exclusivity: Ensure the correctAnswer is 100% defensible and no distractor is partially true.",
      "5. Distractor Plausibility: Make all 3 distractors realistic misconceptions of matched length and tone.",
      "6. Diagnostic Information Gain: Maximize separation between true conceptual mastery and surface guessing.",
      'Return strictly JSON: {"alfaAttributePlan":{"targetConcept":"...","nonLeadingCheck":"...","distractorStrategy":"..."},"stem":"...","correctAnswer":"...","distractors":["...","...","..."],"explanation":"..."}',
    ].join("\n"),
    user: `Subject: ${input.specialty}\nSource Passage:\n${input.sourcePassage}`,
  }
}

/**
 * 3. Arm 3 (`arm-3-cbp-rationale-stem` — Contrastive Boundary Pinning: Unfactored Rationale Stem):
 *    Register-Decoupled CBP Schema + Two-Speaker Debate where speakers state competing
 *    predictions WITH `"because..."` rationales inside `questionStem`, paired with 4 standard options.
 */
export function buildArm3CbpRationaleStemMcqPrompt(input: {
  specialty: string
  sourcePassage: string
  isSchoolLevel?: boolean
}) {
  const toneLine = input.isSchoolLevel
    ? "Write in friendly, everyday school English a 6th-to-10th grade student immediately understands."
    : "Write in clear, natural everyday English—like two real students or coworkers talking through a problem."

  const system = [
    `You write clear 4-option multiple-choice questions for ${input.specialty}.`,
    toneLine,
    "",
    "STEP 1 — FILL OUT THIS EVERYDAY PLANNING CHECK BEFORE WRITING THE QUESTION:",
    '- whyTwoAnswersLookRightAtFirst (<= 25 words): "If someone reads fast, which two answers both look right at first glance and why do people mix them up?"',
    '- whatActuallyHappens (<= 35 words): "Based strictly on the passage, what actually happens and why?"',
    '- everydayExplanation (<= 30 words): "Explain that rule in plain, normal words you\'d say to a friend—no stiff textbook buzzwords that tip off a guesser."',
    '- oneFactInQuestionThatSettlesIt (<= 25 words): "Name the exact number, lab result, line of code, or detail you are putting INSIDE the question so only one answer works and the other three are definitely wrong."',
    "- threeCommonMistakes (exact length 3): Pick 3 distinct mistakeType values from:",
    '  1. "TRUE_GENERALLY_DOESNT_PROVE_IT_HERE"',
    '  2. "REVERSED_THE_DIRECTION_OR_SIGN"',
    '  3. "MISSED_ONE_KEY_DETAIL_IN_SETUP"',
    '  4. "USED_RULE_WHERE_REQUIREMENT_FAILS"',
    '  5. "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS"',
    "  For each mistake, fill in { mistakeType, whySomeonePicksIt, exactDetailInQuestionThatBreaksIt, wrongChoiceText }.",
    "",
    "STEP 2 — WRITE THE TWO-SPEAKER DEBATE QUESTION (WITH 'BECAUSE' REASONS IN THE STEM):",
    "- Put every fact, number, or setup detail the reader needs right inside `questionStem` (never say 'according to the passage').",
    "- Inside `questionStem`, have Speaker 1 and Speaker 2 state their competing predictions WITH their 'because...' reasons right in the stem (for example: 'Speaker 1 says [Outcome A] because [Reason 1], while Speaker 2 says [Outcome B] because [Reason 2]. Who is right, and why?').",
    "- Write 4 standard plain-English answer choices (`correctOption` and 3 `distractors` matching `wrongChoiceText` from `threeCommonMistakes`) of similar word length (within +/-20%).",
  ].join("\n")

  const user = [
    `Subject: ${input.specialty}`,
    "Source Passage:",
    input.sourcePassage,
    "",
    "Return strictly JSON:",
    JSON.stringify({
      whyTwoAnswersLookRightAtFirst: "...",
      whatActuallyHappens: "...",
      everydayExplanation: "...",
      oneFactInQuestionThatSettlesIt: "...",
      threeCommonMistakes: [
        {
          mistakeType: "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS",
          whySomeonePicksIt: "...",
          exactDetailInQuestionThatBreaksIt: "...",
          wrongChoiceText: "...",
        },
        {
          mistakeType: "REVERSED_THE_DIRECTION_OR_SIGN",
          whySomeonePicksIt: "...",
          exactDetailInQuestionThatBreaksIt: "...",
          wrongChoiceText: "...",
        },
        {
          mistakeType: "MISSED_ONE_KEY_DETAIL_IN_SETUP",
          whySomeonePicksIt: "...",
          exactDetailInQuestionThatBreaksIt: "...",
          wrongChoiceText: "...",
        },
      ],
      questionStem:
        "... Speaker 1 says ... because ..., while Speaker 2 says ... because ... Who is right, and why?",
      correctOption: "...",
      distractors: ["...", "...", "..."],
      explanation: "...",
    }),
  ].join("\n")

  return {
    armAlias: "arm-3-cbp-rationale-stem" as const,
    armId: "arm-3-cbp-rationale-stem" as const,
    temperature: DECODING_HYPERPARAMETERS.temperatureByRole.arm3AndArm4Pass1Producer,
    system,
    user,
  }
}

export const buildArm3PlainEnglishDebateLeakyMcqPrompt =
  buildArm3CbpRationaleStemMcqPrompt

/**
 * 4. Arm 4 (`arm-4-cbp-factored-2x2` — Contrastive Boundary Pinning: Factored 2x2 Stem):
 *    Register-Decoupled CBP Schema + Factored 2x2 Debative Stem Contract:
 *    Two speakers in `questionStem` state ONLY what they predict will happen or do—ZERO
 *    `"because / since / due to"` reasons in `questionStem`, ending strictly on
 *    `"Who is right, and why?"` with zero narrator words after `?` (`sanitizePredictionOnlyStem`),
 *    paired with a 2x2 Speaker x Reason option grid (`+/-20%` word-length parity).
 */
export function buildArm4CbpFactored2x2McqPrompt(input: {
  specialty: string
  sourcePassage: string
  isSchoolLevel?: boolean
}) {
  const toneLine = input.isSchoolLevel
    ? "Write in friendly, everyday school English a 6th-to-10th grade student immediately understands."
    : "Write in clear, natural everyday English—like two real students or coworkers comparing predictions."

  const system = [
    `You write clear, high-discrimination 4-option multiple-choice questions for ${input.specialty}.`,
    toneLine,
    "",
    "STEP 1 — FILL OUT THIS EVERYDAY PLANNING CHECK BEFORE WRITING THE QUESTION:",
    '- whyTwoAnswersLookRightAtFirst (<= 25 words): "If someone reads fast, which two answers both look right at first glance and why do people mix them up?"',
    '- whatActuallyHappens (<= 35 words): "Based strictly on the passage, what actually happens and why?"',
    '- everydayExplanation (<= 30 words): "Explain that rule in plain, normal words you\'d say to a friend—no stiff textbook buzzwords that tip off a guesser."',
    '- oneFactInQuestionThatSettlesIt (<= 25 words): "Name the exact number, lab result, line of code, or detail you are putting INSIDE the question so only one answer works and the other three are definitely wrong."',
    "- speaker1PredictionOnly (<= 16 words) & speaker2PredictionOnly (<= 16 words): State ONLY what Speaker 1 and Speaker 2 predict will happen or what action to take—ZERO 'because', 'since', or 'due to' reasons here.",
    "- threeCommonMistakes (exact length 3): Pick 3 distinct mistakeType values from:",
    '  1. "TRUE_GENERALLY_DOESNT_PROVE_IT_HERE"',
    '  2. "REVERSED_THE_DIRECTION_OR_SIGN"',
    '  3. "MISSED_ONE_KEY_DETAIL_IN_SETUP"',
    '  4. "USED_RULE_WHERE_REQUIREMENT_FAILS"',
    '  5. "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS"',
    "  For each mistake, fill in { mistakeType, whySomeonePicksIt, exactDetailInQuestionThatBreaksIt, wrongChoiceText }.",
    "",
    "STEP 2 — PREDICTION-ONLY QUESTION STEM (ZERO 'BECAUSE' REASONS IN THE STEM):",
    "- Put every fact, number, or setup detail the reader needs right inside `questionStem` (never say 'according to the passage').",
    "- Inside `questionStem`, Speaker 1 and Speaker 2 must state ONLY what they predict will happen or do—NEVER write 'because', 'since', 'due to', or any explanation inside `questionStem`!",
    '- End `questionStem` strictly on `"Who is right, and why?"` with ZERO narrator words after `?`.',
    "",
    "STEP 3 — 2x2 SPEAKER x REASON OPTION GRID (+/-20% WORD-LENGTH PARITY):",
    "- Split the 4 answer choices into an exact 2x2 grid: 2 choices starting with `'Speaker 1 is right because ...'` and 2 choices starting with `'Speaker 2 is right because ...'`:",
    "  * `correctOption`: Winning speaker + the true everyday reason from `whatActuallyHappens` and `everydayExplanation`.",
    "  * `distractor 1` (same winning speaker, wrong reason): Winning speaker + a tempting wrong reason from `threeCommonMistakes[0]` (Right speaker, wrong reason).",
    "  * `distractor 2` (other speaker, mistake 2): Other speaker + wrong reason from `threeCommonMistakes[1]`.",
    "  * `distractor 3` (other speaker, mistake 3): Other speaker + wrong reason from `threeCommonMistakes[2]` broken by `oneFactInQuestionThatSettlesIt`.",
    "- Keep all 4 choices within +/-20% of the same word count so a guesser cannot spot the right answer by length.",
  ].join("\n")

  const user = [
    `Subject: ${input.specialty}`,
    "Source Passage:",
    input.sourcePassage,
    "",
    "Return strictly JSON:",
    JSON.stringify({
      whyTwoAnswersLookRightAtFirst: "...",
      whatActuallyHappens: "...",
      everydayExplanation: "...",
      oneFactInQuestionThatSettlesIt: "...",
      speaker1PredictionOnly: "...",
      speaker2PredictionOnly: "...",
      threeCommonMistakes: [
        {
          mistakeType: "WENT_WITH_HABIT_AGAINST_GIVEN_FACTS",
          whySomeonePicksIt: "...",
          exactDetailInQuestionThatBreaksIt: "...",
          wrongChoiceText: "Speaker 1 is right because ...",
        },
        {
          mistakeType: "REVERSED_THE_DIRECTION_OR_SIGN",
          whySomeonePicksIt: "...",
          exactDetailInQuestionThatBreaksIt: "...",
          wrongChoiceText: "Speaker 2 is right because ...",
        },
        {
          mistakeType: "MISSED_ONE_KEY_DETAIL_IN_SETUP",
          whySomeonePicksIt: "...",
          exactDetailInQuestionThatBreaksIt: "...",
          wrongChoiceText: "Speaker 2 is right because ...",
        },
      ],
      questionStem:
        "... Speaker 1 predicts ..., while Speaker 2 predicts ... Who is right, and why?",
      correctOption: "Speaker 1 is right because ...",
      distractors: [
        "Speaker 1 is right because ...",
        "Speaker 2 is right because ...",
        "Speaker 2 is right because ...",
      ],
      explanation: "...",
    }),
  ].join("\n")

  return {
    armAlias: "arm-4-cbp-factored-2x2" as const,
    armId: "arm-4-cbp-factored-2x2" as const,
    temperature: DECODING_HYPERPARAMETERS.temperatureByRole.arm3AndArm4Pass1Producer,
    system,
    user,
  }
}

export const buildArm4PlainEnglishPredOnly2x2McqPrompt =
  buildArm4CbpFactored2x2McqPrompt

/**
 * 5. Register-Decoupled CBP `MediQ` Patient-Question Prompt (for the 20 open-ended clinical cases):
 *    `mostLikelyCondition` vs. `closestLookalikeCondition` -> `oneSimpleQuestionToTellThemApart`
 *    -> emit one natural, register-decoupled patient question (`<= 16 words`, zero medical jargon, non-leading).
 */
export function buildPlainEnglishMediqPrompt(input: {
  specialty: string
  patientPresentation: string
  armId?: "arm-3-cbp-rationale-stem" | "arm-4-cbp-factored-2x2"
}) {
  const armId = input.armId ?? "arm-4-cbp-factored-2x2"
  const system = [
    `You are a friendly ${input.specialty} doctor talking directly to a patient.`,
    "Given an incomplete patient story, figure out the single best everyday question to ask the patient next:",
    "1. mostLikelyCondition (<= 15 words): What condition looks most likely right now?",
    "2. closestLookalikeCondition (<= 15 words): What is the closest look-alike condition you could mix it up with?",
    "3. oneSimpleQuestionToTellThemApart (<= 20 words): What everyday symptom or detail tells those two apart?",
    "4. plainPatientQuestion (<= 16 words): Write ONE natural, plain-English question to ask the patient (<= 16 words, zero medical jargon, non-leading, easy to answer from everyday life).",
    "5. whyThisAnswerSettlesIt (<= 30 words): In plain words, how does their yes/no or short answer separate the two conditions?",
    "6. whyItIsNonLeading (<= 25 words): Why does this question avoid putting words in the patient's mouth?",
  ].join("\n")

  const user = [
    `Patient Case (${input.specialty}):`,
    input.patientPresentation,
    "",
    "Return strictly JSON:",
    JSON.stringify({
      mostLikelyCondition: "...",
      closestLookalikeCondition: "...",
      oneSimpleQuestionToTellThemApart: "...",
      plainPatientQuestion: "...",
      whyThisAnswerSettlesIt: "...",
      whyItIsNonLeading: "...",
    }),
  ].join("\n")

  return {
    armAlias: armId,
    armId,
    temperature: DECODING_HYPERPARAMETERS.temperatureByRole.arm3AndArm4Pass1Producer,
    system,
    user,
  }
}

export function parsePlainEnglishMediqResponse(
  raw: Record<string, unknown>
): PlainMediqPlan & {
  followUpQuestion: string
  targetedMissingHypothesis: string
  whyHighInfoGain: string
  nonLeadingJustification: string
} {
  const mostLikelyCondition = String(
    raw.mostLikelyCondition || "Primary suspected condition"
  ).trim()
  const closestLookalikeCondition = String(
    raw.closestLookalikeCondition || "Closest look-alike condition"
  ).trim()
  const oneSimpleQuestionToTellThemApart = String(
    raw.oneSimpleQuestionToTellThemApart || "Key symptom difference"
  ).trim()

  let plainPatientQuestion = String(
    raw.plainPatientQuestion ||
      raw.followUpQuestion ||
      "When did this start, and have you noticed any other new symptoms?"
  )
    .trim()
    .replace(/\s+/g, " ")

  if (countWords(plainPatientQuestion) > 16) {
    plainPatientQuestion = `${clampWords(plainPatientQuestion, 16).replace(
      /[.?!]+$/,
      ""
    )}?`
  } else if (!/\?\s*$/.test(plainPatientQuestion)) {
    plainPatientQuestion = `${plainPatientQuestion.replace(/[.!]+$/, "")}?`
  }

  const whyThisAnswerSettlesIt = String(
    raw.whyThisAnswerSettlesIt ||
      raw.whyHighInfoGain ||
      "Directly separates the top two possibilities based on the patient's answer."
  ).trim()
  const whyItIsNonLeading = String(
    raw.whyItIsNonLeading ||
      raw.nonLeadingJustification ||
      "Asks a neutral everyday question without suggesting a diagnosis."
  ).trim()

  return {
    mostLikelyCondition,
    closestLookalikeCondition,
    oneSimpleQuestionToTellThemApart,
    plainPatientQuestion,
    whyThisAnswerSettlesIt,
    whyItIsNonLeading,
    followUpQuestion: plainPatientQuestion,
    targetedMissingHypothesis: `${mostLikelyCondition} vs. ${closestLookalikeCondition} (${oneSimpleQuestionToTellThemApart})`,
    whyHighInfoGain: whyThisAnswerSettlesIt,
    nonLeadingJustification: whyItIsNonLeading,
  }
}

/**
 * Pass-2 Repair Prompt Builder for Register-Decoupled CBP MCQs.
 * Invoked only when Pass-1 fails Blind Cold-Solver (`Solver B`) single-key validity or option parity.
 */
export function buildPlainEnglishPass2RepairMcqPrompt(input: {
  specialty: string
  sourcePassage: string
  armId: "arm-3-cbp-rationale-stem" | "arm-4-cbp-factored-2x2"
  pass1Stem: string
  pass1CorrectOption: string
  pass1Distractors: [string, string, string]
  plan: PlainHumanPlan
  solverDiagnostic: string
}) {
  const isPredOnly = input.armId === "arm-4-cbp-factored-2x2"
  const stemRule = isPredOnly
    ? "Keep Speaker 1 and Speaker 2 stating ONLY their predictions in `questionStem` (ZERO 'because / since / due to' in the stem), ending strictly on 'Who is right, and why?' with 2 options starting with 'Speaker 1 is right because ...' and 2 starting with 'Speaker 2 is right because ...' within +/-20% word count."
    : "Keep Speaker 1 and Speaker 2 stating their competing predictions WITH their 'because...' reasons in `questionStem`, and make sure only `correctOption` holds under the exact detail in the stem."

  const system = [
    `You are fixing a 4-option multiple-choice question for ${input.specialty} that a blind cold-solver flagged as ambiguous or mismatched.`,
    "Make sure `oneFactInQuestionThatSettlesIt` is stated crystal-clearly inside `questionStem` so `correctOption` is 100% unambiguously right and all 3 `distractors` are definitely ruled out by that fact.",
    stemRule,
    'Return strictly JSON: {"questionStem":"...","correctOption":"...","distractors":["...","...","..."],"explanation":"..."}',
  ].join("\n")

  const user = [
    `Subject: ${input.specialty}`,
    `Source Passage:\n${input.sourcePassage}`,
    `Decisive Fact to Inline in Question: ${input.plan.oneFactInQuestionThatSettlesIt}`,
    `What Actually Happens: ${input.plan.whatActuallyHappens}`,
    `Blind Cold-Solver Diagnostic: ${input.solverDiagnostic}`,
    `Draft Question Stem:\n${input.pass1Stem}`,
    `Draft Correct Option:\n${input.pass1CorrectOption}`,
    `Draft Distractors:\n1) ${input.pass1Distractors[0]}\n2) ${input.pass1Distractors[1]}\n3) ${input.pass1Distractors[2]}`,
  ].join("\n\n")

  return {
    temperature:
      DECODING_HYPERPARAMETERS.temperatureByRole.arm3AndArm4Pass2RepairProducer,
    system,
    user,
  }
}

/**
 * Regex detecting causal rationale giveaway clauses ("because ...", "since ...", "due to ...")
 * inside speaker-prediction sentences of a debative stem.
 */
export const PREDICTION_ONLY_CAUSAL_LEAK_REGEX =
  /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|2|a|b|one|two)\s+(?:predicts|says|argues|insists|claims|thinks|believes|counters)[^.?!]{2,160}?\b(?:because|since|due\s+to|as\s+a\s+result\s+of|on\s+the\s+grounds\s+that|by\s+applying)\b/i

/**
 * Programmatic stem sanitizer (`sanitizePredictionOnlyStem`) for `arm-4-cbp-factored-2x2`:
 * 1. Strips any `"because / since / due to / as a result of"` causal clauses attached to speaker predictions in `questionStem`.
 * 2. Strips any narrator sentences or trailing text after `?` and enforces the terminal question `"Who is right, and why?"`.
 */
export function sanitizePredictionOnlyStem(rawStem: string): string {
  let clean = rawStem.replace(/\s+/g, " ").trim()

  // Split into sentences and strip causal rationale clauses from non-interrogative speaker sentences
  const rawSentences = clean
    .split(/(?<=[.?!])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)

  const strippedSentences: string[] = []
  let seenSpeaker2 = false

  for (let i = 0; i < rawSentences.length; i++) {
    let s = rawSentences[i]
    const isQuestion = /\?\s*$/.test(s)

    if (
      /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:2|b|two)\b/i.test(
        s
      )
    ) {
      seenSpeaker2 = true
    } else if (
      seenSpeaker2 &&
      !isQuestion &&
      /\b(?:because|since|due\s+to|meaning|therefore|thus|note\s+that|recall\s+that)\b/i.test(
        s
      )
    ) {
      // Drop narrator spoiler sentence smuggled after Speaker 2's prediction
      continue
    }

    if (!isQuestion) {
      // Strip "because / since / due to / as a result of" attached to speaker predictions
      s = s
        .replace(
          /(\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|2|a|b|one|two)\b[^,;:.?!]*?)\s*,?\s*\b(?:because|since|due\s+to|as\s+a\s+result\s+of|on\s+the\s+grounds\s+that|by\s+applying)\b[^,;:.?!]*/gi,
          "$1"
        )
        .replace(
          /(\b(?:predicts|says|argues|insists|claims|thinks|believes|counters)\b[^,;:.?!]*?)\s*,?\s*\b(?:because|since|due\s+to|as\s+a\s+result\s+of|on\s+the\s+grounds\s+that)\b[^,;:.?!]*/gi,
          "$1"
        )
        .replace(/\s+,/g, ",")
        .replace(/\s+\./g, ".")
        .replace(/\s{2,}/g, " ")
        .trim()
    }

    strippedSentences.push(s)
  }

  clean = strippedSentences.join(" ").trim()

  // Truncate any trailing narrator words after the question mark and standardize ending to "Who is right, and why?"
  const qIdx = clean.lastIndexOf("?")
  if (qIdx >= 0) {
    clean = clean.slice(0, qIdx + 1).trim()
    clean = clean.replace(
      /\b(?:who(?:se)?\s+is\s+(?:right|correct)|which\s+(?:speaker|student|colleague|person|claim|prediction|side)\s+is\s+(?:right|correct))[^.?!]*\?\s*$/i,
      "Who is right, and why?"
    )
    if (!/Who is right, and why\?\s*$/.test(clean)) {
      clean = clean.replace(/[^.?!]+\?\s*$/, "Who is right, and why?")
    }
  } else if (clean.length > 0) {
    clean = `${clean.replace(/[.!]+$/, "").trim()} Who is right, and why?`
  }

  if (!/Who is right, and why\?$/.test(clean)) {
    clean = `${clean.replace(/\?\s*$/, "").replace(/[.!]+$/, "").trim()} Who is right, and why?`
  }

  return clean.trim()
}

/**
 * Enforces the `arm-3-cbp-rationale-stem` unfactored rationale stem format:
 * Two speakers in `questionStem` state competing predictions WITH `"because..."` reasons in the stem.
 */
export function ensureLeakyDebateStem(input: {
  questionStem: string
  whatActuallyHappens: string
  firstMistakeReason: string
}): string {
  let stem = input.questionStem.replace(/\s+/g, " ").trim()
  const hasTwoSpeakers =
    /\b(?:speaker|student|colleague|claim|person|classmate)\s*(?:1|a|one)\b/i.test(
      stem
    ) &&
    /\b(?:speaker|student|colleague|claim|person|classmate)\s*(?:2|b|two)\b/i.test(
      stem
    )
  const hasCausalInStem = PREDICTION_ONLY_CAUSAL_LEAK_REGEX.test(stem)

  if (hasTwoSpeakers && hasCausalInStem) {
    return stem
  }

  const r1 = clampWords(
    input.whatActuallyHappens.replace(/^(?:because|since)\s+/i, ""),
    16
  )
  const r2 = clampWords(
    input.firstMistakeReason.replace(/^(?:because|since)\s+/i, ""),
    16
  )
  const baseSetup = stem.replace(/Who is right[^?]*\?\s*$/i, "").trim()
  return `${baseSetup} Speaker 1 says the first outcome holds because ${r1}, while Speaker 2 says the opposite outcome holds because ${r2}. Who is right, and why?`
}

/**
 * Enforces the `arm-4-cbp-factored-2x2` stem and 2x2 Speaker x Reason option grid
 * with +/-20% word-length parity.
 */
export function enforcePredictionOnly2x2Contract(input: {
  questionStem: string
  speaker1PredictionOnly?: string
  speaker2PredictionOnly?: string
  correctOption: string
  distractors: [string, string, string]
  everydayExplanation?: string
}): {
  questionStem: string
  correctOption: string
  distractors: [string, string, string]
} {
  let stem = input.questionStem.trim()
  const hasTwoSpeakers =
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|a|one)\b/i.test(
      stem
    ) &&
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:2|b|two)\b/i.test(
      stem
    )

  if (!hasTwoSpeakers) {
    const p1 = clampWords(
      String(input.speaker1PredictionOnly || "the first outcome happens"),
      14
    )
    const p2 = clampWords(
      String(input.speaker2PredictionOnly || "the opposite outcome happens"),
      14
    )
    const setupOnly = stem.replace(/[^.?!]*\?\s*$/, "").trim()
    stem = `${setupOnly} Speaker 1 predicts ${p1}, while Speaker 2 predicts ${p2}. Who is right, and why?`
  }

  stem = sanitizePredictionOnlyStem(stem)

  const sp1Regex =
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|a|one|first)\b/i
  const sp2Regex =
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:2|b|two|second)\b/i
  const prefixStripRegex =
    /^(?:(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|2|a|b|one|two)\s*(?:is\s+(?:right|correct))?\s*[:,—-]?\s*)?(?:because|since|due\s+to)?\s*/i

  const extractReasonBody = (opt: string): string => {
    const body = opt.replace(prefixStripRegex, "").trim()
    if (!body) return "the key detail in the setup directly settles the outcome."
    const lowered = body.length > 1 ? body[0].toLowerCase() + body.slice(1) : body.toLowerCase()
    return lowered.endsWith(".") ? lowered : `${lowered}.`
  }

  const keyBacksSp2 =
    sp2Regex.test(input.correctOption) && !sp1Regex.test(input.correctOption)
  const winLabel = keyBacksSp2 ? "Speaker 2" : "Speaker 1"
  const oppLabel = keyBacksSp2 ? "Speaker 1" : "Speaker 2"

  let correctOption = `${winLabel} is right because ${extractReasonBody(
    input.correctOption
  )}`
  let d0 = `${winLabel} is right because ${extractReasonBody(
    input.distractors[0]
  )}`
  let d1 = `${oppLabel} is right because ${extractReasonBody(
    input.distractors[1]
  )}`
  let d2 = `${oppLabel} is right because ${extractReasonBody(
    input.distractors[2]
  )}`

  // Enforce +/-20% word-length parity across the 4 options when needed
  const allFour = [correctOption, d0, d1, d2]
  const counts = allFour.map(countWords)
  const targetWords = Math.max(12, Math.min(24, counts[0]))
  const minAllowed = Math.max(9, Math.floor(targetWords * 0.82))
  const maxAllowed = Math.ceil(targetWords * 1.18)

  const normalizeOptionLength = (opt: string): string => {
    let cur = opt.trim()
    const wc = countWords(cur)
    if (wc > maxAllowed) {
      cur = `${clampWords(cur, maxAllowed).replace(/[.!]+$/, "")}.`
    } else if (wc < minAllowed) {
      const pad = "under the exact numbers and setup facts given in this problem."
      const needed = minAllowed - wc
      const padWords = pad.split(/\s+/).slice(0, Math.max(2, needed)).join(" ")
      cur = `${cur.replace(/[.!]+$/, "")} ${padWords}.`
    }
    return cur
  }

  correctOption = normalizeOptionLength(correctOption)
  d0 = normalizeOptionLength(d0)
  d1 = normalizeOptionLength(d1)
  d2 = normalizeOptionLength(d2)

  return {
    questionStem: stem,
    correctOption,
    distractors: [d0, d1, d2],
  }
}

export function checkOptionWordLengthParity20Pct(
  correctOption: string,
  distractors: [string, string, string]
): { passesParity: boolean; minWords: number; maxWords: number; ratio: number } {
  const counts = [correctOption, ...distractors].map((o) =>
    Math.max(1, countWords(o))
  )
  const minWords = Math.min(...counts)
  const maxWords = Math.max(...counts)
  const meanWords = counts.reduce((a, b) => a + b, 0) / counts.length
  const maxDevRatio = Math.max(
    ...counts.map((c) => Math.abs(c - meanWords) / Math.max(1, meanWords))
  )
  return {
    passesParity: maxDevRatio <= 0.25,
    minWords,
    maxWords,
    ratio: Number(maxDevRatio.toFixed(3)),
  }
}

/**
 * Programmatic `predictionOnlyDebativeStemContract` Validator used in Arm 4 (`arm-4-cbp-factored-2x2`):
 * Verifies that `questionStem` stages an explicit two-speaker prediction clash,
 * contains zero `"because..."` / `"since..."` / `"due to..."` causal mechanism leaks
 * in the stem, ends cleanly on `"Who is right, and why?"` with zero trailing words after `?`,
 * and partitions the 4 options into a crossed 2x2 Speaker x Reason matrix.
 */
export function validatePredictionOnlyDebativeStemContract(input: {
  questionStem: string
  correctOption: string
  distractors: [string, string, string]
}) {
  const hasTwoSpeakerClash =
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|a|one)\b/i.test(
      input.questionStem
    ) &&
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:2|b|two)\b/i.test(
      input.questionStem
    )
  const zeroCausalLeakInStem = !PREDICTION_ONLY_CAUSAL_LEAK_REGEX.test(
    input.questionStem
  )
  const endsCleanlyOnQuestionMark = /\?\s*$/.test(input.questionStem.trim())
  const allOptions = [input.correctOption, ...input.distractors]
  const speaker1Options = allOptions.filter((o) =>
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:1|a|one|first)\b/i.test(
      o
    )
  ).length
  const speaker2Options = allOptions.filter((o) =>
    /\b(?:speaker|student|colleague|claim|engineer|doctor|clinician|analyst|historian|researcher|person|classmate)\s*(?:2|b|two|second)\b/i.test(
      o
    )
  ).length
  const hasCrossed2x2Options = speaker1Options === 2 && speaker2Options === 2

  return {
    hasTwoSpeakerClash,
    zeroCausalLeakInStem,
    endsCleanlyOnQuestionMark,
    hasCrossed2x2Options,
    passesContract:
      hasTwoSpeakerClash &&
      zeroCausalLeakInStem &&
      endsCleanlyOnQuestionMark &&
      hasCrossed2x2Options,
  }
}

/**
 * Programmatic Monotonic Blind-Solver (`Solver B`) Rollback Gate:
 * Keep Pass-2 repair only if `Solver B` single-key validity holds or improves over Pass-1.
 */
export function evaluateMonotonicSolverBGate(input: {
  pass1ValidKey: boolean
  pass1ParityDefects: number
  pass2ValidKey: boolean
  pass2ParityDefects: number
}): { selectedPass: "PASS_1" | "PASS_2"; rolledBackToPass1: boolean } {
  if (input.pass1ValidKey && input.pass1ParityDefects === 0) {
    return { selectedPass: "PASS_1", rolledBackToPass1: false }
  }
  if (!input.pass2ValidKey) {
    return { selectedPass: "PASS_1", rolledBackToPass1: true }
  }
  if (!input.pass1ValidKey && input.pass2ValidKey) {
    return { selectedPass: "PASS_2", rolledBackToPass1: false }
  }
  if (
    input.pass1ValidKey &&
    input.pass2ValidKey &&
    input.pass2ParityDefects < input.pass1ParityDefects
  ) {
    return { selectedPass: "PASS_2", rolledBackToPass1: false }
  }
  return { selectedPass: "PASS_1", rolledBackToPass1: true }
}
