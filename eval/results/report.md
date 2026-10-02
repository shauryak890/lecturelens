# LectureLens evaluation report

- Generated: 2026-10-02T09:12:37Z | prompt file version 1.2.0
- Models: llm: `gemini-3.8-flash`, judge: `gemini-3.1-flash-lite`, embeddings: `BAAI/bge-small-en-v1.5`, reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Dataset: `eval\qa_dataset.jsonl` (30 questions)
- Retrieval: mode `hybrid`, rerank `False`, chunk size 400 tokens, final_k 5

LLM-judge scores (faithfulness, relevancy) come from a single judge model and are best read comparatively across configurations rather than as absolute quality.

## Results

| Config | Hit@1 | Hit@5 | MRR | Faithful. | Relev. (1-5) | Abstain acc. | p50 latency |
|---|---|---|---|---|---|---|---|
| Default config (hybrid) | 0.800 | 1.000 | 0.873 | 1.000 | 5.00 | 100% | 3,150 ms |
| BM25 only | 0.720 | 1.000 | 0.815 | - | - | - | - |
| Dense only | 0.680 | 0.960 | 0.797 | - | - | - | - |
| Hybrid (RRF) | 0.800 | 1.000 | 0.873 | - | - | - | - |
| Hybrid + rerank | 0.680 | 0.960 | 0.790 | - | - | - | - |
| Default, minimal prompt (no rules) (10 q) | - | - | - | 1.000 | 5.00 | 100% | 3,255 ms |

## Retrieval (25 answerable questions)

| Config | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@5 | MRR | p50 retrieval |
|---|---|---|---|---|---|---|---|---|---|
| Default config (hybrid) | 0.800 | 0.920 | 1.000 | 0.680 | 0.767 | 0.873 | 0.819 | 0.873 | 17 ms |
| BM25 only | 0.720 | 0.880 | 1.000 | 0.633 | 0.760 | 0.873 | 0.788 | 0.815 | 2 ms |
| Dense only | 0.680 | 0.920 | 0.960 | 0.580 | 0.780 | 0.860 | 0.769 | 0.797 | 16 ms |
| Hybrid (RRF) | 0.800 | 0.920 | 1.000 | 0.680 | 0.767 | 0.873 | 0.819 | 0.873 | 22 ms |
| Hybrid + rerank | 0.680 | 0.920 | 0.960 | 0.580 | 0.787 | 0.873 | 0.779 | 0.790 | 1,114 ms |

Hit@5 by question type (default config):

| Type | Questions | Hit@5 |
|---|---|---|
| comparison | 4 | 1.000 |
| conceptual | 7 | 1.000 |
| factual | 7 | 1.000 |
| followup | 4 | 1.000 |
| keyword | 3 | 1.000 |

## Chunk-size sweep (retrieval only, default config)

| Chunk size (tokens) | Chunks | Avg tokens | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| 200 | 330 | 140 | 0.720 | 0.960 | 0.811 |
| 400 | 215 | 188 | 0.800 | 1.000 | 0.873 |
| 600 | 203 | 196 | 0.800 | 1.000 | 0.877 |

Sizes 400, 600 tie (same Hit@5, MRR within 0.01); the configured **400 tokens** is kept.

## Generation

| Variant | Questions | Faithfulness | Relevancy | Citation validity | Repairs | Citation hit | Abstain acc. | Abstain P / R | No-LLM abstentions | p50 / p95 | Prompt tokens/query | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Default config (hybrid) | 30 | 1.000 | 5.00 | 100% | 0% | 100% | 100% | 100% / 100% | 0 | 3,150 ms / 42,431 ms | 1813 | 0 |
| Default, minimal prompt (no rules) | 10 | 1.000 | 5.00 | 100% | 0% | 100% | 100% | 100% / 100% | 0 | 3,255 ms / 42,020 ms | 1753 | 0 |

Faithfulness = supported claims / claims (LLM judge, answered questions only). Citation validity = answers whose cited ids all existed before repair. Citation hit = answered answerable questions citing a gold page. Abstention P/R treats "not answerable" as the positive class. Latency is end-to-end per question and, during evaluation, includes waiting for the client-side rate limiter that the judge calls share, so p95 is mostly queueing; p50 is representative of interactive use.

## Examples

### Successes (3)

**q26 (unanswerable): How does RLHF work to align large language models?**

- Answer (low): I couldn't find this in your course material. The closest related topics covered in the excerpts are tokenization methods (such as BPE, WordPiece, and Unigram) [S3], n-gram language models, smoothing techniques, and perplexity evaluation [S1][S2][S5].
- Cited: 01_tokenization.md p.1, 02_ngram_language_models.md p.1 | faithfulness - | relevancy 5

**q01 (factual): What are the five components in the formal definition of a finite automaton?**

- Gold: Introduction to Theory of Computation.pdf p.5 (first retrieved at rank 1)
- Answer (high): A finite automaton is formally defined as a 5-tuple consisting of the following components [S1]: - **$Q$**: A finite set of states (or set of all states) [S1]. - **$\Sigma$**: A set of input symbols taken by the machine as input [S1]. - **$q$**: The initial state (or starting state) of the machine [S1]. - **$F$**: The set of final states [S1]. - **$\delta$**: The transition function, defined for a DFA as $\delta : Q \times \Sigma \to Q$ [S1].
- Cited: Introduction to Theory of Computation.pdf p.5 | faithfulness 1.00 | relevancy 5

**q02 (factual): What is a trap state (dead state) in a DFA?**

- Gold: Introduction to Theory of Computation.pdf p.14 (first retrieved at rank 1)
- Answer (high): A trap state (also called a dead state) in a DFA is a state where a transition enters it and from which the machine can never escape [S1]. In other words, it is a state that cannot lead to or reach a final state [S1][S2]. During DFA simplification or minimization, dead states can be removed from the DFA [S2].
- Cited: Introduction to Theory of Computation.pdf p.14, Introduction to Theory of Computation.pdf p.19 | faithfulness 1.00 | relevancy 5

### Failure analysis (0 of 0)

No failures in the default configuration.

## Per-question results (default config)

| Id | Type | Gold rank | Abstention correct | Gold cited | Faithful. | Relev. | Latency ms |
|---|---|---|---|---|---|---|---|
| q01 | factual | 1 | yes | yes | 1.00 | 5 | 3,596 |
| q02 | factual | 1 | yes | yes | 1.00 | 5 | 1,833 |
| q03 | factual | 1 | yes | yes | 1.00 | 5 | 3,976 |
| q04 | factual | 1 | yes | yes | 1.00 | 5 | 3,957 |
| q05 | factual | 1 | yes | yes | 1.00 | 5 | 2,525 |
| q06 | factual | 1 | yes | yes | 1.00 | 5 | 4,238 |
| q07 | factual | 1 | yes | yes | 1.00 | 5 | 2,707 |
| q08 | conceptual | 1 | yes | yes | 1.00 | 5 | 2,541 |
| q09 | conceptual | 1 | yes | yes | 1.00 | 5 | 3,150 |
| q10 | conceptual | 1 | yes | yes | 1.00 | 5 | 2,680 |
| q11 | conceptual | 4 | yes | yes | 1.00 | 5 | 3,006 |
| q12 | conceptual | 1 | yes | yes | 1.00 | 5 | 43,094 |
| q13 | conceptual | 2 | yes | yes | 1.00 | 5 | 3,226 |
| q14 | conceptual | 1 | yes | yes | 1.00 | 5 | 3,138 |
| q15 | comparison | 1 | yes | yes | 1.00 | 5 | 4,129 |
| q16 | comparison | 3 | yes | yes | 1.00 | 5 | 3,029 |
| q17 | comparison | 1 | yes | yes | 1.00 | 5 | 2,550 |
| q18 | comparison | 1 | yes | yes | 1.00 | 5 | 4,625 |
| q19 | keyword | 4 | yes | yes | 1.00 | 5 | 2,651 |
| q20 | keyword | 1 | yes | yes | 1.00 | 5 | 42,431 |
| q21 | keyword | 1 | yes | yes | 1.00 | 5 | 3,375 |
| q22 | followup | 1 | yes | yes | 1.00 | 5 | 2,738 |
| q23 | followup | 1 | yes | yes | 1.00 | 5 | 4,523 |
| q24 | followup | 1 | yes | yes | 1.00 | 5 | 4,222 |
| q25 | followup | 2 | yes | yes | 1.00 | 5 | 1,800 |
| q26 | unanswerable | - | yes | - | - | 5 | 3,585 |
| q27 | unanswerable | - | yes | - | - | 5 | 2,622 |
| q28 | unanswerable | - | yes | - | - | 5 | 2,639 |
| q29 | unanswerable | - | yes | - | - | 5 | 40,836 |
| q30 | unanswerable | - | yes | - | - | 5 | 4,155 |
