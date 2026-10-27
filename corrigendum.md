# Corrigendum: Exp. 1 JSON assertion outcomes

Release corrigendum, version 0.2, 2026-10-27. The paper is unchanged. The
final release will link this document to the corrected `pipeline/exp1_assert.py`
and the affected prediction tables.

Change note (v0.2): the first review found that the fallback parser still
classified four malformed JSON responses by their first integer. Those four
rows are now `format_failure`; the explicit-field rule is unchanged. The
second review and the mechanical check described below were added.

This corrigendum accompanies the DiaLexPoL code and data release. The paper is
not revised; this document records a release-level correction and its effect
on interpretation.

All numerical tables in this document can be regenerated from the released
`llm_predictions/exp1_predictions.csv` by running
`analysis/corrigendum_exp1_json.ipynb`.

## What happened

The Exp. 1 JSON prompts returned structured answers containing fields such as
`confidence` and `sense_number`. The evidence is consistent with the assertion
receiving a parsed Python dictionary rather than the original JSON string. The assertion
converted that dictionary to a Python representation with single-quoted keys.
Its double-quoted `"sense_number"` regular expression then failed and the
fallback extracted the first integer in the representation. Depending on the
response, this was usually the confidence value or an integer in the
justification.

The released tables therefore preserve both `outcome_stored` (the original
promptfoo result) and `outcome_corrected` (the result obtained from the actual
`sense_number` field). `response_status` separately records `responded` or
`no_response`.

## Scope and results

The following claims in §5.1 (variable matrix), §5.3 (implementation details),
§6.2 (results), and §6.3 (error analysis) are retracted by this corrigendum:
the Exp. 1 JSON-penalty interpretation, the JSON rows in the reported accuracy
tables, and the claim that the JSON penalty justified restricting Phase B and
Exp. 2 to plain prompts. The corrected comparison is JSON 37.14% versus plain 35.31% on
the same restricted grid; the stored −16.6 percentage-point penalty does not
hold. The plain-only design remains valid, but its justification no longer rests
on a JSON penalty.

The affected run is `phase_a_json_restricted` (33,760 rows):

| Measure | Stored | Corrected |
|---|---:|---:|
| Correct | 6,070 (17.98%) | 12,540 (37.14%) |
| Format failure | 3,688 | 3,692 |
| No response | 5 | 5 |

Among rows whose stored and corrected outcomes differ:

| Movement | Rows |
|---|---:|
| stored correct → corrected wrong | 1,346 |
| stored wrong → corrected correct | 7,816 |
| net increase in correct rows | 6,470 |

The 9,162 row-level disagreements were classified by the first integer that
the faulty fallback could see: 9,114 first matched the confidence value, 48
matched an integer in the justification, and none remained unexplained. The
corrected parser treats a response with no parseable `sense_number` as a
`format_failure`.

There were 3,796 invalid or incomplete JSON outputs. The corrected rule now
scores 104 malformed outputs with an intact `"sense_number": N` field and
classifies four malformed outputs without that field as `format_failure`.
The latter four were the rows exposed by the first human review: three
extracted the confidence value `5`, and one extracted a number from prose.

The 108 rows consist of 104 rows with an explicit `"sense_number": N` field
despite malformed surrounding JSON and four rows where the fallback found a
number without a complete field. The corrected grouping is therefore 104
explicit-field rows plus 4 format failures; none of these 108 rows is an empty
output. A first private review covered all four fallback rows plus 16 seeded
stratified explicit-field rows (seed `20261027`): 17 passed and the three
confidence-value cases failed. A second private review covered 40 additional
seeded stratified explicit-field rows (seed `20261028`): all 40 passed. The
two explicit-field reviews therefore passed 56/56 rows. Both sheets remain
outside the release because they contain raw response excerpts.

The review criterion was: the extracted number must equal the integer in the
actual response's intact `sense_number` field, rather than a confidence value
or an unrelated integer. The four no-field fallback rows are now format
failures and are not treated as successful predictions. For the 104
explicit-field rows, the mechanical check found 104/104 extracted numbers
equal to the integer following the `sense_number` key; the human reviews
covered 56 of those rows, all passing.

The echo audit searched the raw `response.output` for every row in all five
final runs (237,040 rows) using each rendered prompt's first non-empty system
line and the fixed marker `Candidate senses:`. It found zero prompt echoes,
including zero among stored format failures. The exporter repeats this check
as a hard assertion.

The loose first-integer fallback was used for 598 responded Phase A plain rows
(1.91%), 702 Phase B plain rows (2.13%), 30 Exp. 2 CORE rows (0.09%), and 109
Exp. 2 EXTENDED rows (0.11%). These small shares are retained as part of the
original plain/prose parsing rule; the JSON correction does not alter them.

The following runs have identical stored and corrected outcomes on every row:

- `phase_a_plain_restricted` — 33,760 rows;
- `phase_b_plain` — 35,600 rows;
- `exp2_core` — 33,480 rows;
- `exp2_extended` — 100,440 rows.

No-response rows are retained as stored `wrong` outcomes in these unaffected
runs. They are also explicitly marked by `response_status` rather than being
reclassified by this corrigendum.

## No-response concentration

The plain Exp. 1 no-response rows are concentrated in
`llama-3.2-3b-instruct`:

| Run | Provider | Template | No response |
|---|---|---|---:|
| Phase A | llama-3.2-3b-instruct | fewshot | 837 |
| Phase A | llama-3.2-3b-instruct | zeroshot | 842 |
| Phase A | llama-3.2-3b-instruct | zeroshot_decomposed | 844 |
| Phase B | llama-3.2-3b-instruct | fewshot | 889 |
| Phase B | llama-3.2-3b-instruct | zeroshot | 890 |
| Phase B | llama-3.2-3b-instruct | zeroshot_decomposed | 890 |

The paper's stored plain-format-failure figures exclude these no-response
rows: they are recorded as `wrong`, not `format_failure`. The near-total
failure behaviour of `llama-3.2-3b-instruct` should be interpreted with this
distinction in mind.

In particular, §6.3 reports 891 of Phase B's 960 plain format failures for
`llama-3.2-3b-instruct` (25.0%). Phase B contains 3,560 rows for this provider,
of which 2,669 are null outputs counted as `wrong`. Thus all of its Phase B
plain rows are non-answers: 25% recorded as format failures and 75% as null
outputs. This is a response-availability issue, not part of the JSON assertion
bug.

Exp. 2 contains 29 no-response rows in CORE and 38 in EXTENDED, all from
`gemma-4-31b-it`: CORE has 5 few-shot, 6 justified, 9 zero-shot, and 9
decomposed rows; EXTENDED has 11 few-shot, 7 justified, 10 zero-shot, and 10
decomposed rows. These are not part of the JSON assertion correction and remain
classified as stored results.

## Interpretation

The paper’s Exp. 1 JSON accuracy and plain-versus-JSON comparison are affected:
stored JSON accuracy was 17.98%, whereas corrected JSON accuracy is 37.14%,
compared with 35.31% for the corresponding plain condition. Analyses based on
the stored `success` flag are affected. Analyses that used the independently
parsed `predicted_number` are not affected by this particular assertion bug,
but should be checked for consistency if they combine accuracy and prediction
fields.

The provider/template table shows that corrected JSON is competitive with plain
for most providers; `gemma-4-31b-it` reaches approximately 53–56% across its
JSON templates. The stored picture of a uniform JSON penalty is therefore no
longer supported.

The release does not silently rewrite the paper’s reported values. It preserves
the original stored result, provides the corrected result, and documents the
correction here.

## Provider/template before-and-after table

Each cell contains 844 rows.

| Provider | Template | Stored % | Corrected % |
|---|---|---:|---:|
| deepseek-chat-v3-0324 | decomposed | 17.65 | 43.13 |
| deepseek-chat-v3-0324 | fewshot | 13.74 | 47.99 |
| deepseek-chat-v3-0324 | justified | 23.46 | 42.54 |
| deepseek-chat-v3-0324 | zeroshot | 14.93 | 41.59 |
| gemma-3-4b-it | decomposed | 19.67 | 25.83 |
| gemma-3-4b-it | fewshot | 14.45 | 31.04 |
| gemma-3-4b-it | justified | 16.94 | 27.49 |
| gemma-3-4b-it | zeroshot | 17.77 | 26.18 |
| gemma-4-31b-it | decomposed | 12.20 | 53.91 |
| gemma-4-31b-it | fewshot | 12.09 | 56.16 |
| gemma-4-31b-it | justified | 11.97 | 53.91 |
| gemma-4-31b-it | zeroshot | 11.37 | 53.44 |
| gpt-4o-mini | decomposed | 24.64 | 34.60 |
| gpt-4o-mini | fewshot | 11.97 | 38.27 |
| gpt-4o-mini | justified | 21.68 | 34.36 |
| gpt-4o-mini | zeroshot | 26.42 | 35.55 |
| llama-3.1-8b-instruct | decomposed | 16.59 | 34.60 |
| llama-3.1-8b-instruct | fewshot | 12.09 | 36.26 |
| llama-3.1-8b-instruct | justified | 17.42 | 31.40 |
| llama-3.1-8b-instruct | zeroshot | 14.57 | 33.53 |
| llama-3.2-3b-instruct | all templates | 0.00 | 0.00 |
| llama-3.3-70b-instruct | decomposed | 40.40 | 40.40 |
| llama-3.3-70b-instruct | fewshot | 42.89 | 42.89 |
| llama-3.3-70b-instruct | justified | 37.80 | 37.80 |
| llama-3.3-70b-instruct | zeroshot | 40.64 | 40.64 |
| llama-4-maverick | decomposed | 18.36 | 47.99 |
| llama-4-maverick | fewshot | 13.27 | 53.32 |
| llama-4-maverick | justified | 16.94 | 49.64 |
| llama-4-maverick | zeroshot | 16.94 | 48.82 |
| phi-4 | decomposed | 29.15 | 35.31 |
| phi-4 | fewshot | 12.09 | 39.45 |
| phi-4 | justified | 32.46 | 37.20 |
| phi-4 | zeroshot | 26.30 | 35.55 |
| qwen3-235b-a22b-2507 | decomposed | 18.13 | 48.10 |
| qwen3-235b-a22b-2507 | fewshot | 11.26 | 50.47 |
| qwen3-235b-a22b-2507 | justified | 14.93 | 47.63 |
| qwen3-235b-a22b-2507 | zeroshot | 16.00 | 48.82 |
