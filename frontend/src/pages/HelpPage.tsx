/** HelpPage — plain-language guide to using LLM Evolution. */
import { BookOpen, Upload, BarChart2, Wand2, Database, HelpCircle } from "lucide-react";

interface SectionProps {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}

function Section({ icon, title, children }: SectionProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-blue-500">{icon}</span>
        <h2 className="font-semibold text-slate-800 text-lg">{title}</h2>
      </div>
      <div className="text-slate-600 text-sm leading-relaxed space-y-2">
        {children}
      </div>
    </div>
  );
}

function Step({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-700 text-xs font-bold flex items-center justify-center mt-0.5">
        {n}
      </span>
      <p>{children}</p>
    </div>
  );
}

function Callout({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-amber-50 border border-amber-200 rounded p-3 text-amber-800 text-sm">
      {children}
    </div>
  );
}

export function HelpPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-blue-500" />
          How to Use LLM Evolution
        </h1>
        <p className="text-slate-500 text-sm mt-1">
          A plain-language guide — no coding knowledge needed.
        </p>
      </div>

      {/* What is this app? */}
      <Section icon={<HelpCircle className="w-5 h-5" />} title="What is this app?">
        <p>
          LLM Evolution lets you watch <strong>13 different AI language models</strong> learn
          to write English — right in front of your eyes.
        </p>
        <p>
          Think of it like a science experiment. You give the app a book or any text file, and it
          feeds that text to 13 different AI "brains," each of which learns in a slightly different
          way. Then you can compare how well each one writes.
        </p>
        <p>
          The models range from very simple ones (that just memorise which letters tend to follow
          each other) all the way up to a mini version of the same Transformer technology that
          powers ChatGPT.
        </p>
      </Section>

      {/* Quick start */}
      <Section icon={<Upload className="w-5 h-5" />} title="Quick start — 4 steps">
        <Step n={1}>
          <span>
            <strong>Go to Ingest.</strong> Drag a <code>.txt</code> file onto the upload box, or
            click <em>Browse files</em> and pick one. Any plain English text works — a short story,
            a Wikipedia article, or even a book chapter. The longer the text, the better the models
            will learn.
          </span>
        </Step>
        <Step n={2}>
          <span>
            <strong>Watch the training happen.</strong> A progress bar will appear and the event log
            will start filling up. Behind the scenes the app is splitting your text into chunks and
            training all 13 models at the same time. You can also see each model generating sample
            sentences live in the <em>Monte Carlo Generation</em> panel — this is the model
            "practising" what it has learned so far.
          </span>
        </Step>
        <Step n={3}>
          <span>
            <strong>Go to Stats</strong> once training finishes. You'll see an accuracy graph for
            every model. The higher the line, the more real English words the model produces — a
            good rough measure of how well it learned.
          </span>
        </Step>
        <Step n={4}>
          <span>
            <strong>Go to Generate.</strong> Pick a word count and click <em>Generate</em>. All 13
            models will each write a short passage at the same time. Compare the results side by
            side — you'll immediately see which models sound more human and which ones are still
            a bit… random.
          </span>
        </Step>
        <Callout>
          <strong>Tip:</strong> The app works best with at least a few thousand words of text. A
          short paragraph won't give the models much to learn from. Try pasting a news article or
          a chapter from a book.
        </Callout>
      </Section>

      {/* The 13 models explained */}
      <Section icon={<BarChart2 className="w-5 h-5" />} title="The 13 models — what are they?">
        <p>
          All language models try to answer the same question:{" "}
          <em>"Given what I've already written, what word (or letter) should come next?"</em>
          The 13 models differ in <strong>how much context</strong> they look at and{" "}
          <strong>how clever</strong> their memory is.
        </p>

        <div className="space-y-2 pt-1">
          <div className="rounded border border-slate-100 bg-slate-50 p-3">
            <p className="font-medium text-slate-700 mb-1">Character n-gram models (5 models)</p>
            <p>
              These look at individual <em>letters</em>. A "bigram" model only looks at the previous
              1 letter to guess the next one. A "5-gram" looks at the previous 4 letters. They're
              fast and simple, but they often produce made-up words because they only think about
              letters, not meaning.
            </p>
          </div>

          <div className="rounded border border-slate-100 bg-slate-50 p-3">
            <p className="font-medium text-slate-700 mb-1">Word n-gram models (5 models)</p>
            <p>
              Same idea, but with whole <em>words</em> instead of letters. A word bigram picks the
              next word by looking at just the previous one. A word 5-gram looks at the previous 4
              words. These sound more natural, but they need a lot of training text to work well.
            </p>
          </div>

          <div className="rounded border border-slate-100 bg-slate-50 p-3">
            <p className="font-medium text-slate-700 mb-1">BPE n-gram models (5 models)</p>
            <p>
              BPE (Byte-Pair Encoding) splits text into <em>sub-word pieces</em> — common word
              parts like "ing", "un", or "tion". This is the same trick used by real large language
              models. It's a middle ground between characters and whole words.
            </p>
          </div>

          <div className="rounded border border-slate-100 bg-slate-50 p-3">
            <p className="font-medium text-slate-700 mb-1">Feedforward neural network</p>
            <p>
              A small neural network trained with PyTorch. It looks at a fixed window of recent
              tokens and learns patterns that the simple counting models can't capture.
            </p>
          </div>

          <div className="rounded border border-slate-100 bg-slate-50 p-3">
            <p className="font-medium text-slate-700 mb-1">Transformer neural network</p>
            <p>
              The most powerful model in the app. Transformers use a mechanism called{" "}
              <em>attention</em> to weigh which earlier words matter most — the same core idea
              behind GPT and ChatGPT. It trains slowly but usually produces the most coherent
              output.
            </p>
          </div>
        </div>
      </Section>

      {/* Reading the stats */}
      <Section icon={<BarChart2 className="w-5 h-5" />} title="Reading the Stats page">
        <p>
          The <strong>accuracy</strong> score shown on the Stats page is the percentage of words
          in a generated sample that are actual English words (found in a dictionary). It's a
          quick-and-dirty measure — a score of 80% means 8 out of every 10 generated words are real.
        </p>
        <p>
          The graph plots accuracy over time as more chunks of your text are processed. You should
          see the scores climb as each model learns more. If a model's line stays flat or low, it
          may need more training data.
        </p>
        <Callout>
          A high accuracy score doesn't mean the text <em>makes sense</em> — it just means the
          words exist. You'll see this clearly on the Generate page: some models hit 90%+ real
          words but still produce nonsense sentences, while the Transformer tends to produce more
          coherent phrases.
        </Callout>
      </Section>

      {/* Generate page */}
      <Section icon={<Wand2 className="w-5 h-5" />} title="The Generate page">
        <p>
          Click <strong>Generate</strong> and all 13 models write a passage at the same time.
          Each word is colour-coded:
        </p>
        <div className="flex flex-wrap gap-4 py-1">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-green-500 shrink-0" />
            <span>Green = real English word</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-400 shrink-0" />
            <span>Red = made-up / misspelled word</span>
          </span>
        </div>
        <p>
          The <em>Augment</em> checkbox runs an automatic spell-and-grammar correction on the
          output so you can see what the model was "trying" to say.
        </p>
      </Section>

      {/* DB page */}
      <Section icon={<Database className="w-5 h-5" />} title="The DB page">
        <p>
          The DB page lets you browse the raw data the app stores behind the scenes — things like
          the accuracy scores for every chunk, and the vocabulary the models built up.
        </p>
        <p>
          You can also click <strong>Reset DB</strong> here to wipe everything and start fresh with
          a new text file.
        </p>
        <Callout>
          Resetting is permanent — all training progress will be lost.
        </Callout>
      </Section>
    </div>
  );
}
