// Mirror the sampler's category -> parent -> progress-bin -> state probabilities.
// Raw ties remain in the evidence; only states with ranking pairs enter sampling.
export function rankingSupport(shards, minimumStates = 16, minimumScales = 2) {
  const parents = new Map();
  for (const { index, contract, trajectory } of shards) {
    const id = index.asset;
    if (!parents.has(id))
      parents.set(id, {
        asset: id,
        category: index.category,
        bins: new Map(),
        modes: [false, true].map((preserve_uv) => ({
          preserve_uv,
          states: 0,
          informative: 0,
          pairs: 0,
          scales: new Set(),
        })),
      });
    const parent = parents.get(id);
    if (parent.category !== index.category) throw Error('Parent category changed');
    const mode = parent.modes[Number(contract.preserve_uv)];
    for (const state of trajectory) {
      let known = 0,
        preferred = 0;
      state.rows.forEach((row, i) => {
        if (row.geometry_rejected || (row.known_mask & 24) === 24) {
          known++;
          preferred += (state.preferred_mask >>> i) & 1;
        }
      });
      const pairs = preferred * (known - preferred);
      mode.states++;
      if (!pairs) continue;
      mode.informative++;
      mode.pairs += pairs;
      mode.scales.add(index.source_triangles);
      const bin = Math.min(3, Math.floor((state.triangles / index.source_triangles) * 4));
      if (!parent.bins.has(bin)) parent.bins.set(bin, []);
      parent.bins.get(bin).push(pairs);
    }
  }
  const categories = new Map(),
    rows = [];
  for (const parent of parents.values()) {
    const mean = parent.bins.size
      ? [...parent.bins.values()].reduce(
          (sum, pairs) => sum + pairs.reduce((a, b) => a + b, 0) / pairs.length,
          0,
        ) / parent.bins.size
      : 0;
    if (parent.bins.size) {
      if (!categories.has(parent.category)) categories.set(parent.category, []);
      categories.get(parent.category).push(mean);
    }
    for (const mode of parent.modes)
      rows.push({
        asset: parent.asset,
        category: parent.category,
        ...mode,
        scales: [...mode.scales].sort((a, b) => a - b),
        supported: mode.informative >= minimumStates && mode.scales.size >= minimumScales,
      });
  }
  const pairWeight = [...categories].map(([category, parents]) => ({
    category,
    expected_pairs_per_sampled_state: parents.reduce((a, b) => a + b, 0) / parents.length,
  }));
  const weights = pairWeight.map((c) => c.expected_pairs_per_sampled_state);
  return {
    minimum_informative_states: minimumStates,
    minimum_source_scales: minimumScales,
    admitted: rows.length > 0 && rows.every((r) => r.supported),
    rows,
    category_pair_weight: pairWeight,
    largest_pair_weight_ratio: weights.length ? Math.max(...weights) / Math.min(...weights) : null,
  };
}
