import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const FORBIDDEN_WORD_TOKENS = [
  ["sans", "krit"].join(""),
  ["nya", "ya"].join(""),
  ["shastra", "rtha"].join(""),
  ["panca", "vayava"].join(""),
  ["hetva", "bhasa"].join(""),
  ["purva", "paksha"].join(""),
  ["sidd", "hanta"].join(""),
  ["sams", "haya"].join(""),
  ["nir", "naya"].join(""),
  ["udaha", "rana"].join(""),
  ["upa", "naya"].join(""),
  ["niga", "mana"].join(""),
  ["savyabhi", "cara"].join(""),
  ["viru", "ddha"].join(""),
  ["satprati", "paksha"].join(""),
  ["asi", "ddha"].join(""),
  ["bad", "hita"].join(""),
  ["got", "ama"].join(""),
  ["mati", "lal"].join(""),
  ["gan", "eri"].join(""),
  ["quiz", "beef"].join(""),
  ["acha", "rya"].join(""),
  ["ni", "ti"].join(""),
  ["dris", "hti"].join(""),
  ["j", "ev"].join(""),
  ["des", "hik"].join(""),
  ["q", "lo"].join(""),
  ["sharedSurface", "Confusion"].join(""),
  ["decisiveDisqualifying", "StemFact"].join(""),
  ["ruleIn", "PlainLife"].join(""),
  ["OVERGENERALIZED_NECESSARY", "_NOT_SUFFICIENT"].join(""),
]

const FORBIDDEN_INTERNAL_PATTERNS: ReadonlyArray<{ label: string; regex: RegExp }> = [
  {
    label: "Forbidden internal codename or classical terminology token",
    regex: new RegExp(`\\b(?:${FORBIDDEN_WORD_TOKENS.join("|")})\\b`, "i"),
  },
  {
    label: "Internal product substring",
    regex: new RegExp(["quiz", "beef"].join(""), "i"),
  },
  {
    label: "Internal ticket prefix",
    regex: new RegExp(`\\b(?:${["CR", "S"].join("")}|${["ST", "K"].join("")}|${["AU", "D"].join("")}|${["TC", "H"].join("")})-[A-Z0-9]+`, "i"),
  },
]

function walkFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  const files: string[] = []
  for (const e of entries) {
    if (e.name === "node_modules" || e.name === ".git" || e.name === "dist") continue
    const full = path.join(dir, e.name)
    if (e.isDirectory()) {
      files.push(...walkFiles(full))
    } else if (!full.endsWith(".pdf")) {
      files.push(full)
    }
  }
  return files
}

describe("Zero-Forbidden-Term & Zero-Internal-Codename Audit", () => {
  it("contains zero forbidden philosophical jargon, internal codenames, ticket prefixes, or legacy slot names across all repository files", () => {
    const repoRoot = path.resolve(__dirname, "..")
    const allFiles = walkFiles(repoRoot)
    const violations: string[] = []

    for (const filePath of allFiles) {
      const rel = path.relative(repoRoot, filePath)
      // Skip scanning this test file's own regex definitions
      if (rel === "tests/zero-leak-audit.test.ts") continue
      const content = fs.readFileSync(filePath, "utf-8")
      for (const pat of FORBIDDEN_INTERNAL_PATTERNS) {
        const match = content.match(pat.regex)
        if (match) {
          violations.push(`${rel}: matched ${pat.label} ("${match[0]}")`)
        }
      }
    }

    expect(violations).toEqual([])
  })
})
