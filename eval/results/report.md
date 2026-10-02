# LectureLens evaluation report

- Generated: 2026-10-02T07:42:11Z | prompt file version 1.2.0
- Models: llm: `gemini-3.8-flash`, judge: `gemini-3.1-flash-lite`, embeddings: `BAAI/bge-small-en-v1.5`, reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Dataset: `eval\qa_dataset.jsonl` (30 questions)
- Retrieval: mode `hybrid`, rerank `True`, chunk size 400 tokens, final_k 5

LLM-judge scores (faithfulness, relevancy) come from a single judge model and are best read comparatively across configurations rather than as absolute quality.

## Results

| Config | Hit@1 | Hit@5 | MRR | Faithful. | Relev. (1-5) | Abstain acc. | p50 latency |
|---|---|---|---|---|---|---|---|
| Default config (hybrid + rerank) | 0.680 | 0.960 | 0.790 | 0.990 | 4.87 | 97% | 3,625 ms |
| BM25 only | 0.720 | 1.000 | 0.815 | - | - | - | - |
| Dense only | 0.680 | 0.960 | 0.797 | - | - | - | - |
| Hybrid (RRF) | 0.800 | 1.000 | 0.873 | - | - | - | - |
| Hybrid + rerank | 0.680 | 0.960 | 0.790 | - | - | - | - |
| Default, minimal prompt (no rules) (10 q) | - | - | - | 1.000 | 5.00 | 100% | 887 ms |

## Retrieval (25 answerable questions)

| Config | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@5 | MRR | p50 retrieval |
|---|---|---|---|---|---|---|---|---|---|
| Default config (hybrid + rerank) | 0.680 | 0.920 | 0.960 | 0.580 | 0.787 | 0.873 | 0.779 | 0.790 | 1,149 ms |
| BM25 only | 0.720 | 0.880 | 1.000 | 0.633 | 0.760 | 0.873 | 0.788 | 0.815 | 1 ms |
| Dense only | 0.680 | 0.920 | 0.960 | 0.580 | 0.780 | 0.860 | 0.769 | 0.797 | 19 ms |
| Hybrid (RRF) | 0.800 | 0.920 | 1.000 | 0.680 | 0.767 | 0.873 | 0.819 | 0.873 | 21 ms |
| Hybrid + rerank | 0.680 | 0.920 | 0.960 | 0.580 | 0.787 | 0.873 | 0.779 | 0.790 | 1,146 ms |

Hit@5 by question type (default config):

| Type | Questions | Hit@5 |
|---|---|---|
| comparison | 4 | 1.000 |
| conceptual | 7 | 1.000 |
| factual | 7 | 1.000 |
| followup | 4 | 0.750 |
| keyword | 3 | 1.000 |

## Chunk-size sweep (retrieval only, default config)

| Chunk size (tokens) | Chunks | Avg tokens | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| 200 | 330 | 140 | 0.680 | 0.920 | 0.777 |
| 400 | 215 | 188 | 0.680 | 0.960 | 0.790 |
| 600 | 203 | 196 | 0.680 | 0.960 | 0.793 |

Sizes 400, 600 tie (same Hit@5, MRR within 0.01); the configured **400 tokens** is kept.

## Generation

| Variant | Questions | Faithfulness | Relevancy | Citation validity | Repairs | Citation hit | Abstain acc. | Abstain P / R | No-LLM abstentions | p50 / p95 | Prompt tokens/query | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Default config (hybrid + rerank) | 30 | 0.990 | 4.87 | 100% | 0% | 100% | 97% | 83% / 100% | 3 | 3,625 ms / 31,453 ms | 1598 | 0 |
| Default, minimal prompt (no rules) | 10 | 1.000 | 5.00 | 100% | 0% | 100% | 100% | 100% / 100% | 3 | 887 ms / 6,480 ms | 1219 | 0 |

Faithfulness = supported claims / claims (LLM judge, answered questions only). Citation validity = answers whose cited ids all existed before repair. Citation hit = answered answerable questions citing a gold page. Abstention P/R treats "not answerable" as the positive class. Latency is end-to-end per question and, during evaluation, includes waiting for the client-side rate limiter that the judge calls share, so p95 is mostly queueing; p50 is representative of interactive use.

## Examples

### Successes (3)

**q26 (unanswerable): How does RLHF work to align large language models?**

- Answer (low): I couldn't find this in your course material. The closest related topics covered in the excerpts are tokenization methods (such as BPE, WordPiece, and Unigram language models) [S2], the role of tokenizers in language models [S4][S5], and n-gram language models, including maximum likelihood estimation and smoothing techniques [S1][S3].
- Cited: 01_tokenization.md p.1, 02_ngram_language_models.md p.1, PPT -3 tokenization-multilinguality.pdf p.8, PPT -3 tokenization-multilinguality.pdf p.16 | faithfulness - | relevancy 5

**q01 (factual): What are the five components in the formal definition of a finite automaton?**

- Gold: Introduction to Theory of Computation.pdf p.5 (first retrieved at rank 1)
- Answer (high): In the formal definition, a finite automaton is defined as a 5-tuple $\{ Q, \Sigma, q, F, \delta \}$ containing the following five components [S1]: - **$Q$**: A finite set of states (or set of all states) [S1]. - **$\Sigma$**: A set of input symbols that the machine takes as input [S1]. - **$q$**: The initial state (starting state of the machine) [S1]. - **$F$**: A set of final states [S1]. - **$\delta$**: The transition function, defined for a DFA as $\delta : Q \times \Sigma \to Q$ [S1].
- Cited: Introduction to Theory of Computation.pdf p.5 | faithfulness 1.00 | relevancy 5

**q02 (factual): What is a trap state (dead state) in a DFA?**

- Gold: Introduction to Theory of Computation.pdf p.14 (first retrieved at rank 1)
- Answer (high): A **trap state** (also called a **dead state**) in a DFA is a state entered by a transition from which the machine can never escape [S1]. In other words, it is a state that cannot reach or lead to a final state [S1][S2]. Because dead states cannot lead to a final state, they can be removed when simplifying or minimizing the DFA [S2].
- Cited: Introduction to Theory of Computation.pdf p.14, Introduction to Theory of Computation.pdf p.19 | faithfulness 1.00 | relevancy 5

### Failure analysis (2 of 2)

**q06 (factual): Which classes of languages are defined by finite-state automata, pushdown automata and Turing machines?**

- Gold: Introduction to Theory of Computation.pdf p.3, PPT-2 morphology and FSM.pdf p.16 (first retrieved at rank 1)
- Answer (high): Based on the course excerpts, these automata define the following language classes: - **Finite-state automata** define **regular languages** [S1][S4]. - **Pushdown automata** define **context-free languages** [S1][S4]. - **Turing machines** define **recursively enumerable languages** (as well as recursive languages) [S1][S4].
- Cited: Introduction to Theory of Computation.pdf p.3, PPT-2 morphology and FSM.pdf p.16 | faithfulness 0.75 | relevancy 5
- **What went wrong:** Partly unsupported (faithfulness 0.75): Turing machines define recursive languages.

**q25 (followup): And what does inverting one do?**

- Condensed to: *And what does inverting a finite-state transducer do?*
- Gold: PPT-2 morphology and FSM.pdf p.33 (first retrieved at rank none)
- Answer (low): I couldn't find this in your course material. The provided excerpts describe the formal definition of a finite-state transducer [S1], how it defines a relation between two regular languages [S2], and how it maps between surface and underlying forms [S3][S4], but they do not mention inverting a finite-state transducer.
- Cited: PPT-2 morphology and FSM.pdf p.31, PPT-2 morphology and FSM.pdf p.32, PPT1 morphology and FSM.pdf p.13, PPT1 morphology and FSM.pdf p.18 | faithfulness - | relevancy 1
- **What went wrong:** Wrongly abstained (no gold page in the top 5).

## Per-question results (default config)

| Id | Type | Gold rank | Abstention correct | Gold cited | Faithful. | Relev. | Latency ms |
|---|---|---|---|---|---|---|---|
| q01 | factual | 1 | yes | yes | 1.00 | 5 | 985 |
| q02 | factual | 1 | yes | yes | 1.00 | 5 | 975 |
| q03 | factual | 1 | yes | yes | 1.00 | 5 | 3,625 |
| q04 | factual | 1 | yes | yes | 1.00 | 5 | 3,569 |
| q05 | factual | 1 | yes | yes | 1.00 | 5 | 3,680 |
| q06 | factual | 1 | yes | yes | 0.75 | 5 | 7,299 |
| q07 | factual | 1 | yes | yes | 1.00 | 5 | 5,395 |
| q08 | conceptual | 2 | yes | yes | 1.00 | 5 | 10,761 |
| q09 | conceptual | 1 | yes | yes | 1.00 | 5 | 3,292 |
| q10 | conceptual | 1 | yes | yes | 1.00 | 5 | 3,609 |
| q11 | conceptual | 3 | yes | yes | 1.00 | 5 | 3,271 |
| q12 | conceptual | 1 | yes | yes | 1.00 | 5 | 3,403 |
| q13 | conceptual | 2 | yes | yes | 1.00 | 5 | 31,453 |
| q14 | conceptual | 1 | yes | yes | 1.00 | 5 | 6,446 |
| q15 | comparison | 3 | yes | yes | 1.00 | 5 | 5,534 |
| q16 | comparison | 3 | yes | yes | 1.00 | 5 | 10,483 |
| q17 | comparison | 1 | yes | yes | 1.00 | 5 | 3,421 |
| q18 | comparison | 1 | yes | yes | 1.00 | 5 | 4,245 |
| q19 | keyword | 4 | yes | yes | 1.00 | 5 | 4,344 |
| q20 | keyword | 2 | yes | yes | 1.00 | 5 | 3,832 |
| q21 | keyword | 1 | yes | yes | 1.00 | 5 | 35,131 |
| q22 | followup | 1 | yes | yes | 1.00 | 5 | 3,625 |
| q23 | followup | 1 | yes | yes | 1.00 | 5 | 3,189 |
| q24 | followup | 1 | yes | yes | 1.00 | 5 | 10,566 |
| q25 | followup | - | **no** | no | - | 1 | 3,183 |
| q26 | unanswerable | - | yes | - | - | 5 | 3,791 |
| q27 | unanswerable | - | yes | - | - | 5 | 28,141 |
| q28 | unanswerable | - | yes | - | - | 5 | 871 |
| q29 | unanswerable | - | yes | - | - | 5 | 796 |
| q30 | unanswerable | - | yes | - | - | 5 | 826 |
