Perfect moment for a capstone.
This is not an exam. It’s a *small research ritual* 🔬
If the earlier steps taught machinery, this teaches **judgment**.

Below is a **capstone task** that is:

* open-ended but bounded
* technically honest
* achievable in 2–4 hours
* deeply aligned with your minimal Transformer
* impossible to fake by copy-paste



# Final Capstone Task

## “Teach a Transformer to Speak a Small World”

### Narrative framing (give this to students)

> You will train a tiny Transformer on a *small, coherent language world*
> and demonstrate that it learns **local structure**, not memorization.
>
> Performance does not matter. Understanding does.



## Part A — Choose a Micro-Corpus (30 min)

Each student (or pair) must choose **one**:

✅ a poem (≥ 50 lines)
✅ a speech or manifesto
✅ a programming language file
✅ song lyrics from a single artist
✅ a chat log (self-written)

**Rules**

* Must be plain text
* ≤ 50 KB
* English allowed but not required

### Deliverable

A single `.txt` file:

```text
data.txt
```



## Part B — Train Your Own Tokenizer (30 min)

Students must:

1. Train a BPE tokenizer on *only* their corpus
2. Explain their tokenizer choices

### Required outputs

Print:

* vocab size
* 10 most frequent tokens
* encoding of one sentence

### Sanity check questions

* What is the longest token?
* Are punctuation tokens merged or split?
* Where does `[UNK]` appear?



## Part C — Build a Minimal GPT-like Model (45–60 min)

Students must modify your final Transformer to ensure:

✅ causal language modeling
✅ positional embeddings
✅ ≥ 2 layers
✅ ≥ 2 attention heads

**Explicit constraints**

* ≤ 100 lines of model code
* CPU-only training
* context length ≤ 64
* total parameters ≤ 1 million

They may:

* reuse your provided blocks
* remove features if they justify why



## Part D — Train and Log (30 min)

Students must train long enough to show:

📉 decreasing loss
📊 at least one logged failure (e.g. divergence, collapse)

### Required outputs

* training loss every N steps
* one screenshot or log snippet

### Reflection prompts

* When did loss plateau?
* What broke first when hyperparameters were wrong?



## Part E — Generate and Analyze (30 min)

Students must generate:

1. 3 samples from the trained model
2. 3 samples from an *untrained* model

### Required analysis (written, short)

* Which patterns are learned?
* Which are hallucinated?
* Where does repetition appear?

No “this looks good” allowed.
Force concrete observations.



## Part F — Interpret One Attention Head (Optional Bonus)

Students:

1. pick one layer and one head
2. visualize or print attention weights
3. interpret **one concrete behavior**

Examples:

* punctuation anchoring
* indentation tracking
* rhyme patterns
* syntax alignment

This is the “researcher move”.



## Final Deliverables

Students submit:

```
capstone/
├── data.txt
├── train.py
├── model.py
├── generate.py
├── samples.txt
└── reflection.md
```

### `reflection.md` must answer:

1. What *specific* structure did your model learn?
2. What did it fail to learn?
3. Which change gave the biggest improvement?
4. If you had 10× compute, what would you try next?



## Grading Rubric (Conceptual, Not Numerical)

| Dimension         | Excellent             | Acceptable  | Weak                |
| ----------------- | --------------------- | ----------- | ------------------- |
| Understanding     | Clear mental model    | Partial     | Vague               |
| Tokenizer insight | Informed choices      | Default     | Ignored             |
| Model design      | Justified             | Functional  | Copied              |
| Analysis          | Concrete              | Descriptive | Hand-wavy           |
| Honesty           | Acknowledges failures | Neutral     | Performance theater |

Nobody fails for bad outputs.
Only for shallow reasoning.


## Instructor’s Closing Moment

End the seminar with:

> This model works because ideas compose.
> Attention is just book-keeping.
> The hard part is deciding what to care about.



