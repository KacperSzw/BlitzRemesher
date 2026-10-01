// Diagnostic metrics only: incomplete batches never acquire an aggregate SCORE.
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const audited = (measurement) => measurement?.complete === true && measurement.passed === true;
export const successful = (row) =>
  row?.code === 0 && !row.signal && !row.timed_out && !row.cancelled;

export function auditMetrics(audit, ranking = 'learned') {
  if (
    audit?.version !== 2 ||
    audit.complete !== true ||
    !audit.expected_assets?.length ||
    !audit.rows?.length
  )
    throw new Error('complete version-2 audit required');
  const rankings = audit.execution_settings?.audit_rankings;
  if (
    !Array.isArray(rankings) ||
    !rankings.includes(ranking) ||
    new Set(rankings).size !== rankings.length
  )
    throw new Error('invalid audit ranking contract');
  const expected = new Map(audit.expected_assets.map((asset) => [asset.id, asset.category]));
  if (
    expected.size !== audit.expected_assets.length ||
    [...expected].some(([id, category]) => !id || typeof category !== 'string' || !category)
  )
    throw new Error('invalid audit asset contract');
  const seen = new Set();
  let levels;
  for (const row of audit.rows) {
    const key = JSON.stringify([row.asset, row.ranking]);
    if (
      seen.has(key) ||
      !expected.has(row.asset) ||
      expected.get(row.asset) !== row.category ||
      !rankings.includes(row.ranking)
    )
      throw new Error('duplicate or unexpected audit row');
    seen.add(key);
    if (
      row.status !== 'complete' ||
      !Number.isFinite(row.seconds) ||
      row.seconds < 0 ||
      !Array.isArray(row.lods) ||
      row.lods.length < 2 ||
      row.neural?.resource_failures ||
      row.neural?.confirmation_resources ||
      row.neural?.confirmation_nonfinite ||
      row.neural?.confirmation_cancelled
    )
      throw new Error('failed audit row');
    levels ??= row.lods.length;
    if (levels !== row.lods.length) throw new Error('inconsistent LOD schedule');
    for (let i = 0; i < row.lods.length; ++i) {
      const lod = row.lods[i];
      if (
        !Number.isSafeInteger(lod.triangles) ||
        lod.triangles < 1 ||
        (i &&
          (lod.triangles > row.lods[i - 1].triangles ||
            !audited(lod.source) ||
            !audited(lod.adjacent)))
      )
        throw new Error('invalid or unaudited LOD');
    }
  }
  if (seen.size !== expected.size * rankings.length) throw new Error('missing audit rows');
  const assets = audit.rows
    .filter((row) => row.ranking === ranking)
    .map((row) => ({
      asset: row.asset,
      category: row.category,
      triangles: row.lods.map((lod) => lod.triangles),
      retained_ratio: mean(row.lods.slice(1).map((lod) => lod.triangles / row.lods[0].triangles)),
      seconds: row.seconds,
      gpu_workspace_peak_bytes: row.gpu_workspace_peak_bytes ?? null,
    }))
    .sort((a, b) => a.asset.localeCompare(b.asset));
  const categories = [...new Set(assets.map((asset) => asset.category))].sort().map((category) => ({
    category,
    retained_ratio: mean(
      assets.filter((asset) => asset.category === category).map((asset) => asset.retained_ratio),
    ),
  }));
  return {
    retained_ratio: mean(categories.map((category) => category.retained_ratio)),
    categories,
    assets,
    seconds: assets.reduce((sum, asset) => sum + asset.seconds, 0),
    gpu_workspace_peak_bytes: assets.every((asset) =>
      Number.isFinite(asset.gpu_workspace_peak_bytes),
    )
      ? Math.max(...assets.map((asset) => asset.gpu_workspace_peak_bytes))
      : null,
  };
}

export function compareAudits(before, after, ranking = 'learned') {
  for (const key of ['manifest_sha256', 'visual_settings_sha256', 'execution_settings_sha256'])
    if (typeof before?.[key] !== 'string' || !before[key] || before[key] !== after?.[key])
      throw new Error('comparison contract differs: ' + key);
  const base = auditMetrics(before, ranking),
    candidate = auditMetrics(after, ranking);
  if (
    JSON.stringify(
      base.assets.map((a) => [a.asset, a.category, a.triangles[0], a.triangles.length]),
    ) !==
    JSON.stringify(
      candidate.assets.map((a) => [a.asset, a.category, a.triangles[0], a.triangles.length]),
    )
  )
    throw new Error('comparison sources or schedules differ');
  const deltas = base.assets.map((asset, i) => ({
    asset: asset.asset,
    category: asset.category,
    triangle_deltas: asset.triangles
      .slice(1)
      .map((count, level) => candidate.assets[i].triangles[level + 1] - count),
  }));
  return {
    base,
    candidate,
    retained_ratio_improvement: base.retained_ratio - candidate.retained_ratio,
    generation_time_ratio: base.seconds > 0 ? candidate.seconds / base.seconds : null,
    per_lod_deltas: deltas,
    per_lod_regressions: deltas.flatMap((asset) =>
      asset.triangle_deltas.flatMap((delta, level) =>
        delta > 0 ? [{ asset: asset.asset, level: level + 1, triangles: delta }] : [],
      ),
    ),
  };
}

export function auditTermination(audit) {
  auditMetrics(audit, audit.execution_settings.audit_rankings[0]);
  const reasons = {},
    terminal = new Set(['target_reached', 'no_legal_actions', 'no_accepted_action']);
  for (const row of audit.rows) {
    if (
      row.neural?.action_diagnostics_version !== 1 ||
      !Array.isArray(row.neural.action_proposals) ||
      !row.neural.action_proposals.length
    )
      return { known: false, uncensored: false, reasons };
    for (const proposal of row.neural.action_proposals) {
      const reason = proposal.stop_reason;
      reasons[reason] = (reasons[reason] ?? 0) + 1;
      if (
        !terminal.has(reason) &&
        !['trial_budget', 'cancelled', 'resource', 'infeasible_seed'].includes(reason)
      )
        return { known: false, uncensored: false, reasons };
    }
  }
  return {
    known: true,
    uncensored: Object.keys(reasons).every((reason) => terminal.has(reason)),
    reasons,
  };
}
