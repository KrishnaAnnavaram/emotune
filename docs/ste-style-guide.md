# The writing standard: ASD-STE100 Simplified Technical English

The README of emotune and this file obey these rules. Section 3 gives the project vocabulary: the technical names and the technical verbs of emotune.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the emotune documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **label** | One of the six emotion names: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`. | class, category, tag |
| **label schema** | The ordered tuple `LABELS`. The position of a name is its id. | label map, label dict |
| **item** | One text with its true label. | sample, row, example (for test data) |
| **shot** | One train item that a few-shot prompt shows. | example (in prompts), demonstration |
| **split** | `train`, `validation` or `test`. | fold, partition |
| **leaked row** | A train item whose text is also in validation or test. | duplicate, overlap |
| **regime** | The way a model gets the task: `baseline`, `zero_shot`, `few_shot` or `fine_tuned`. | setting, mode |
| **classifier** | A component that returns one prediction per text. | model (for a pipeline), predictor |
| **model** | A language model, named in `configs/models.toml`. | LLM (in prose), network |
| **kind** | `base` or `instruct`: the type of a model checkpoint. | flavour, version |
| **revision** | The commit hash of a model or dataset on the Hugging Face Hub. | version, tag |
| **adapter** | The trained LoRA weights for one base model. | fine-tuned model, checkpoint |
| **answer** | The raw text that a chat model returns. | response, output, completion |
| **prediction** | A label or `invalid` for one item. | result, guess |
| **invalid** | An answer that is not exactly one label. It counts as wrong. | unknown, none, refusal (for all cases) |
| **label scoring** | Prediction by the highest log-likelihood of the six label strings. | constrained decoding, ranking |
| **greedy decoding** | Generation with `do_sample=False` (or temperature 0) and a strict parser. | deterministic sampling |
| **completion-only loss** | A training loss on the answer tokens only. Prompt and padding tokens get the label -100. | masked loss |
| **run** | One folder with `config.json`, `predictions.jsonl`, `metrics.json` and `report.md`. | experiment, job |
| **simulated chat model** | `SimulatedChatLLM`, a seeded stand-in for offline runs. | fake LLM, mock |
| **agreement** | The share of equal predictions of two runs on unlabelled texts, with Cohen's kappa. | consistency, overlap |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **parse** | Turn an answer into a label or `invalid` with the strict rules. |
| **score** | Compute the log-likelihood of each label string after the prompt. |
| **fine-tune** | Train a LoRA adapter with the completion-only loss. |
| **evaluate** | Compute accuracy, macro-F1, per-label scores and intervals on all items. |
| **compare** | Run McNemar and a paired bootstrap on two labelled runs of the same items. |
| **collect** | Get comment text from the YouTube Data API and drop all author data. |
| **hash** | Replace a comment id with a salted SHA-256 value. |
