/**
 * Standalone OpenAI / OpenRouter-Compatible Blind Cold-Solver & Unprimed Novice Foil-Bite Evaluator.
 * Allows external researchers to evaluate their own candidate MCQs using the exact deterministic
 * FNV-1a option shuffling, blind cold-solver validity check, and unprimed novice foil-bite protocol.
 */

export interface CandidateMcqForPsychometrics {
  id: string
  sourcePassage: string
  questionStem: string
  correctOption: string
  distractors: [string, string, string]
}

export interface ShuffledOptionsDeck {
  options: Record<"A" | "B" | "C" | "D", string>
  correctOptionId: "A" | "B" | "C" | "D"
}

/**
 * Deterministic 32-bit FNV-1a permutation so option order is reproducible and unbiased.
 */
export function shuffleFourOptionsDeterministic(
  correctOption: string,
  distractors: [string, string, string],
  seedKey: string
): ShuffledOptionsDeck {
  const entries = [
    { text: correctOption, isKey: true },
    { text: distractors[0], isKey: false },
    { text: distractors[1], isKey: false },
    { text: distractors[2], isKey: false },
  ]

  const fnv1a = (str: string): number => {
    let h = 0x811c9dc5
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i)
      h = Math.imul(h, 0x01000193)
    }
    return h >>> 0
  }

  const decorated = entries
    .map((entry, idx) => ({
      ...entry,
      hash: fnv1a(`${seedKey}::slot-${idx}::${entry.text.slice(0, 48)}`),
    }))
    .sort((a, b) => a.hash - b.hash)

  const letters: readonly ("A" | "B" | "C" | "D")[] = ["A", "B", "C", "D"]
  const options = {} as Record<"A" | "B" | "C" | "D", string>
  let correctOptionId: "A" | "B" | "C" | "D" = "A"

  decorated.forEach((item, idx) => {
    const letter = letters[idx]
    options[letter] = item.text
    if (item.isKey) correctOptionId = letter
  })

  return { options, correctOptionId }
}

export interface PsychometricEvalConfig {
  apiKey: string
  baseUrl?: string
  model?: string
}

/**
 * Evaluates a candidate MCQ with:
 *   1. Blind Cold-Solver (`validKey`: selects single winning option and verifies `hasMultipleValidOptions === false`)
 *   2. Unprimed Novice Guesser (`noviceBitFoil`: attempts question without source passage using surface cues)
 *   3. Item Discrimination `D = I(validKey) - I(noviceCorrect)`
 */
export async function evaluateCandidateMcqPsychometrics(
  candidate: CandidateMcqForPsychometrics,
  config: PsychometricEvalConfig
) {
  const baseUrl = (config.baseUrl ?? "https://openrouter.ai/api/v1").replace(
    /\/$/,
    ""
  )
  const model = config.model ?? "openai/gpt-4o-mini"
  const deck = shuffleFourOptionsDeterministic(
    candidate.correctOption,
    candidate.distractors,
    candidate.id
  )
  const formattedOptions = (["A", "B", "C", "D"] as const)
    .map((k) => `${k}) ${deck.options[k]}`)
    .join("\n")

  const callJson = async (system: string, user: string) => {
    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model,
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
      }),
    })
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`)
    }
    const payload = (await res.json()) as any
    const rawContent = String(payload?.choices?.[0]?.message?.content ?? "{}")
    return JSON.parse(rawContent)
  }

  const [solverData, noviceData] = await Promise.all([
    callJson(
      "You are an expert domain solver verifying single-key validity of a multiple-choice question against its source passage. Select the best option A..D and report whether multiple options are defensible. Return strictly JSON.",
      `Source Passage:\n${candidate.sourcePassage}\n\nQuestion:\n${candidate.questionStem}\n\nOptions:\n${formattedOptions}\n\nReturn JSON: {"selectedOptionId":"A"|"B"|"C"|"D","hasMultipleValidOptions":boolean,"confidence":0.0..1.0}`
    ),
    callJson(
      "You are a hurried, unprimed novice test-taker who has NOT read the textbook passage. You rely on first-glance surface heuristics, familiar buzzwords, and option length. Pick the option that looks most tempting at a glance. Return strictly JSON.",
      `Question:\n${candidate.questionStem}\n\nOptions:\n${formattedOptions}\n\nReturn JSON: {"selectedOptionId":"A"|"B"|"C"|"D","reasoningCue":"1 sentence"}`
    ),
  ])

  const normLetter = (v: unknown): "A" | "B" | "C" | "D" => {
    const up = String(v ?? "A").trim().toUpperCase()
    return up === "A" || up === "B" || up === "C" || up === "D" ? up : "A"
  }

  const solverPick = normLetter(solverData.selectedOptionId)
  const hasMultipleValidOptions = Boolean(solverData.hasMultipleValidOptions)
  const validKey =
    solverPick === deck.correctOptionId && !hasMultipleValidOptions

  const novicePick = normLetter(noviceData.selectedOptionId)
  const noviceCorrect = novicePick === deck.correctOptionId
  const noviceBitFoil = !noviceCorrect
  const discriminationD = (validKey ? 1 : 0) - (noviceCorrect ? 1 : 0)

  return {
    shuffledDeck: deck,
    coldSolve: {
      validKey,
      selectedOptionId: solverPick,
      correctOptionId: deck.correctOptionId,
      hasMultipleValidOptions,
      confidence: Number(solverData.confidence ?? 0.9),
    },
    novice: {
      noviceCorrect,
      noviceBitFoil,
      selectedOptionId: novicePick,
      reasoningCue: String(noviceData.reasoningCue ?? ""),
    },
    discriminationD,
  }
}
