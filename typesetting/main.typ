#set par(justify: true)
#set page(margin: 25.4mm, numbering: "1")
#set heading(numbering: "1.")
#set text(font: "New Computer Modern", size: 12pt)

#let todo(msg) = {
  [
    #text(fill: orange, weight: "bold", font: "JetBrains Mono")[TODO]
    #text(font: "JetBrains Mono")[: #msg]
  ]
}

#outline()

#pagebreak()

= Introduction

The go-to method for applying pretrained models to downstream tasks is
fine-tuning. Having to perform backpropagation on all model parameters sets
restrictive hardware, time and cost boundaries when working with large models.
It is also required to keep a full copy of the fine-tuned model for each
downstream task.

#todo("PEFT näited viidetega ja väga lühikesed seletused?")

#todo("Forward ja backward pass konkreetsem vahe")

There are various parameter efficient fine-tuning methods which enable learning
downstream tasks while only having to updated a small amount of additional
parameters or a subset of the original model's parameters. The addition of extra
parameters is usually not problematic because hardware limitations during a
model's forward pass are a lot smaller than during a backward pass. Using
additional parameters resolves the need to store a full copy of the updated
model for each downstream application.

Large language models can be used for different tasks without any fine-tuning,
@brown2020language showed that writing an appropriate input prompt was effective
for improving the behaviour of a frozen GPT-3 model. This enables the use of one
general model for many tasks, has no added training costs. Changing model
behaviour with input prompts is called in-context learning and the quality of
in-context learning is often worse and less stable than fine-tuning.

#todo("Mida selles töös teeme")

#pagebreak()

= Related Work

#todo("Quantization")

#todo("Parameter efficiency")

PEFT (Parameter Efficient Fine-Tuning) techniques broadly split into
context-based and weight-based.

In context-based methods, additional token embedding tensors, called the "soft
prompt" are initialized. During a forward pass the "soft prompt" is interleaved
with the input prompt. For fine-tuning the base model's weights are frozen, and
only the new "soft prompt" is updated.

Weight-based PEFT methods have various approaches that can be quite different
from each other, but they often involve merging learned smaller parameter
representations into the pretrained model. Merging updated weights into the
original model means no added parameters or changes in architecture, but each
downstream task needs a full copy of the updated model. We don't explore
weight-based PEFT techniques in depth, but their performance and parameter
efficiency is sometimes used as a comparison to context-based methods.

#todo("Diskreetsed meetodid, AutoPrompt ja muud")

== Context-Based Parameter Efficient Fine-Tuning

As a subtask of in-context learning @brown2020language discusses few-shot
learning, the act of including a few example tasks with solutions in the prompt.
Few-shot shot learning improves performance and provides motivation for
context-based methods, it's downsides are the requirement of human involvement
and it's effectiveness being limited by context length. Methods that work by
picking discrete tokens for the prompt are unstable, @liu2024gpt shows that
changing a single word in an input prompt may cause a substantial drop in
performance.

#todo("Mis on promptimine, näited few-shot ja template'id")

#todo("Teoreetiline püstitus, embedding tabelid")

#todo("Tokenization tehnikad SentencePiece BPE")

Vocabulary $V$, discrete tokens $T subset ZZ$, tokenizer $t: V^n arrow T^m$, $n$
and $m$ arbitrary natural numbers usually not equal, sequence length $s$, hidden
size $h$, embedding table $bold(e): T^(s) arrow RR^(s times h)$.

#todo("Domain shift ja kuidas lahendub, kas lahendub, mis see on?")

#todo("Võiks konkreetselt ära mõõta kui palju prompt-tuning säästab")

A downside of context-based methods is that the added learnable parameters are
in the very first layers of a large model. This means that updating the "soft
prompt" during backpropagation requires computing gradients for all the layers
after it. Even if the number learnable parameters is small the decrease in
hardware cost is also small.

=== Prompt Tuning

#todo("Prompt tuning on see mida mina teen")

#todo("SuperGLUE viide")

One of the simplest context-based fine-tuning methods is prompt tuning
@lester2021power. Prompt tuning freezes the entire pre-trained model and only
allows the learning of $k$ token embeddings prepended to the input prompt's
embeddings. In the paper they apply the technique on different sizes of T5
@raffel2020exploring, an encoder-decoder transformer, and show on SuperGLUE
benchmark, that prompt tuning becomes more competitive with full fine-tuning as
model size increases.

SuperGLUE is a classification benchmark, given a sequence of input tokens
$bold(x)$ the task is to predict a class label $y$. Instead of modelling the
task as predicting the probability distribution $p(y|bold(x))$, they cast all
tasks as conditional generation of $p(bold(y)|bold(x))$, where $bold(y)$ is a
sequence of tokens that represents the output class.

Consider a pretrained model parameterized by weights $theta$, with a hidden size
$h$ and a soft prompt $bold(p)$ with weights $theta_bold(p)$. Prompt tuning
takes input prompt embeddings $bold(x)^e in RR^(n times h)$ and soft prompt
embeddings $bold(p)^e in RR^(k times h)$. The concatenation
$[bold(p)^e, bold(x)^e] in RR^((k + n) times h)$ is now the input to the
transformer. Prompt tuning for conditional generation is learning the
distribution $p_(theta, theta_bold(p)) (bold(y) | [bold(p), bold(x)])$ by only
updating soft prompt weights $theta_bold(p)$.

#todo("Initialization, random, pretrained vocabulary, label based ")

#todo("Comparisons, Prompt ensembling, Interpretability")

=== P-Tuning

#todo("P-Tuning v2 on ka olemas, pole veel lugenud")

#todo("LSTM, LAMA, SuperGLUE viited")

P-Tuning @liu2024gpt mixes learnable continuous prompt embeddings with discrete
input tokens into a template
$T = {bold(p)^e_(0:i), bold(x), bold(p)^e_((i + 1):j), bold(y), bold(p)^e_((j + 1):k)}$.
Using an extra embedding function $f: bold(p)^e_i arrow p'_i$, the template is
then mapped into transformer input
$ T' = {p'_(0:i), bold(x)^e, p'_((i + 1):j), bold(y)^e, p'_((j + 1):k)}. $
The extra embedding function acts as a reparameterization of the learnable soft
prompt and is implemented by a small neural network, in their case a multi-layer
perceptron (MLP) or long short-term memory (LSTM). Similarly to prompt tuning
the learning goal is to update the soft prompt's weights $theta_bold(p)$, but
with an additional task of learning the extra embedding network $f$.

The authors evaluate their method with a frozen pretrained model and learned
soft prompt plus embedding network on LAMA knowledge probing benchmark. On
SuperGLUE they evaluate a fully fine-tuned model with the additional parameters.

#todo("HF PEFT viide")

Even if p-tuning was adopted by HuggingFace PEFT as a parameter efficient
fine-tuning method, the authors didn't sell it that way. Instead they compare
manual prompting, discrete prompt search methods and p-tuning, with a goal of
stabilizing training when there are minor changes in discrete prompts. They do
not demonstrate the difference in performance between full fine-tuning and
p-tuning with a frozen pretrained model.

#todo("Experiments osa on imelik")

=== Prefix Tuning

#todo("WARP lihtsam prefix tuning, pole veel lugenud")

#todo("Tulemised")

#todo("Initialization")

Prefix-tuning @li2021prefix works in a similar way to prompt tuning with some
extra steps. The learnable prefix is not just prepended to the input token
embeddings, but to activations of each transformer layer $h_i$.

$
  h_i = cases(
    bold(p)_i ", " & "if" i in bold(p)_"idx" ",",
    M(z_i, h_(<i)) ", " & "otherwise."
  )
$

Similar to @liu2024gpt they use an MLP to reparameterize a smaller prefix tensor
into the actual prefix tensor. This is done because directly updating the prefix
leads to unstable optimization and a small drop in performance.

In the paper they only apply the method on text-to-text tasks.

== Weight-Based Parameter Efficient Fine-Tuning

#todo("Viited BitFit, AdapterTuning, LoRA ja selle tuletised")

Some weight-based method work by simply freezing a subsets of the network,
others insert small learnable intermediary layers called adapters. More
complicated and widely used techniques such as LoRA @hu2022lora and its
derivates, learn smaller decomposed representations of weight matrices. The
learned small representations are merged into the base model's weights.

#todo("Mina ei kasuta, aga mõnikord on võrdluspunktina peft meetodites")

#todo("Lühidalt muid variante ka, AdapterTuning, vt Prefix-tuning lighweight")

== Language Models

#todo("Minul eksklusiivselt Transformer mudelid")

#todo("Encoder, Decoder, Encoder-Decoder architectures")

#todo("Pretraining objectives, BERT MLM, GPT2 causal LM, T5 span corruption")

#todo("Classification/QA/... tasks and head designs")

#todo("BART")

BERT @devlin2019bert, GPT @brown2020language, T5 @raffel2020exploring, LLaMA
@touvron2023llama.

#pagebreak()

= Experiments

#todo("Ilmselt peaks erinevatel arhitektuuridel erinevaid ülesandeid tegema?")

== Data

#todo("Mis on MultiNERD")

#todo("Kuidas MultiNERD-i kasutasin")

MultiNERD @tedeschi-navigli-2022-multinerd.

== Design

#todo("Head design, initialization meetodid")

#todo("System prompt vs gibberish")

#todo("Arhitektuuri disain")

#pagebreak()

= Results

#todo("Tabelid")

#todo("Graafikud")

== Interpretability

#todo("Õpitud promptimine cosine-similarity")

#todo("Klasterdamine algse embedding tabeli suhtes?")

#pagebreak()

= Conclusion

#pagebreak()

#bibliography("ref.bib", style: "annual-reviews-author-date")

#pagebreak()

#heading([Appendices], numbering: none)

#pagebreak()

#heading([License], numbering: none)
