/** RealWordText — renders a sequence of words color-coded by realness.
 *
 * Green = real English word; red = fake/non-word.
 */

export interface WordResult {
  w: string;
  real: boolean;
}

export interface RealWordTextProps {
  words: WordResult[];
}

export function RealWordText({ words }: RealWordTextProps) {
  if (words.length === 0) {
    return <p className="font-mono text-sm text-slate-400 italic">No text yet.</p>;
  }
  return (
    <p className="font-mono text-sm leading-relaxed break-words">
      {words.map((x, i) => (
        <span key={i} className={x.real ? "text-green-600" : "text-red-500"}>
          {x.w}{" "}
        </span>
      ))}
    </p>
  );
}
