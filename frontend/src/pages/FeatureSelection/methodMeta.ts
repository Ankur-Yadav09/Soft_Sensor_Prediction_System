// Mirrors src/feature_selection/auto_selector.py's METHOD_LABELS / METHOD_CATEGORIES —
// static domain constants, not runtime data, so mirroring them here (like
// Train's algorithmFields.ts) is the same pattern already used elsewhere.
export const METHOD_IDS = [
  'target_correlation',
  'mutual_information',
  'mrmr',
  'permutation_importance',
  'elasticnet',
] as const

export const METHOD_LABELS: Record<string, string> = {
  target_correlation: 'Target Correlation',
  mutual_information: 'Mutual Information',
  mrmr: 'mRMR',
  permutation_importance: 'Permutation Importance',
  elasticnet: 'Elastic Net',
}

export const METHOD_CATEGORIES: Record<string, string> = {
  target_correlation: 'Supervised',
  mutual_information: 'Supervised',
  mrmr: 'Advanced Filter',
  permutation_importance: 'Feature Importance',
  elasticnet: 'Intrinsic',
}

export const RECOMMENDATION_COLOR: Record<string, string> = {
  'Highly Recommended': 'var(--success-text)',
  Recommended: 'var(--primary)',
  Consider: '#b45309',
  'Weak Feature': 'var(--error-text)',
}
