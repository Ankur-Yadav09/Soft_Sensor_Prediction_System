// Client-side equal-width histogram binning, used for residual distributions
// on the Predict page (the predict response ships raw rows, not pre-binned
// data — the Preprocess page's histograms are backend-computed since numpy
// is already loaded there; this is the lightweight browser-side equivalent).
export function computeHistogram(values: number[], bins = 20): { counts: number[]; binEdges: number[] } {
  if (values.length === 0) {
    return { counts: [], binEdges: [] }
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const width = range / bins
  const counts = new Array(bins).fill(0)
  for (const v of values) {
    let idx = Math.floor((v - min) / width)
    if (idx >= bins) idx = bins - 1
    if (idx < 0) idx = 0
    counts[idx] += 1
  }
  const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + i * width)
  return { counts, binEdges }
}
