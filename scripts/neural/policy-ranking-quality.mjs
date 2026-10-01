// Ranking experiments must keep the actor fixed and compare every scheduled LOD.
import { isDeepStrictEqual } from 'node:util';
import { auditMetrics, auditTermination, compareAudits } from './quality-metrics.mjs';

function validateCoverage(audit) {
  auditMetrics(audit);
  if (audit.visual_settings?.profile !== 'coverage') throw Error('coverage audit required');
  const areaLimit = audit.visual_settings.max_changed_area;
  if (!Number.isFinite(areaLimit) || areaLimit < 0 || areaLimit > 1)
    throw Error('invalid coverage area limit');
  for (const row of audit.rows)
    for (const lod of row.lods.slice(1))
      for (const [measurement, limit] of [
        [lod.source, lod.source_limit],
        [lod.adjacent, lod.transition_limit],
      ]) {
        if (
          !Number.isFinite(limit) ||
          limit <= 0 ||
          measurement.cancelled ||
          measurement.resource_limited ||
          measurement.nonfinite_error ||
          !Number.isFinite(measurement.error_px) ||
          measurement.error_px < 0 ||
          measurement.error_px > limit ||
          !Number.isFinite(measurement.coverage_upper_px) ||
          measurement.coverage_upper_px < 0 ||
          measurement.coverage_upper_px > limit ||
          !Number.isFinite(measurement.changed_area) ||
          measurement.changed_area < 0 ||
          measurement.changed_area > areaLimit
        )
          throw Error('invalid or over-limit coverage measurement');
      }
  const termination = auditTermination(audit);
  if (!termination.known || termination.reasons.cancelled || termination.reasons.resource)
    throw Error('unknown or interrupted action execution');
  return termination;
}

function withoutTimings(value) {
  if (Array.isArray(value)) return value.map(withoutTimings);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => key !== 'seconds' && !key.endsWith('_seconds'))
      .map(([key, child]) => [key, withoutTimings(child)]),
  );
}

export function comparePolicyRanking(before, after) {
  const termination = { before: validateCoverage(before), after: validateCoverage(after) };
  const comparison = compareAudits(before, after);
  compareAudits(before, after, 'constant');
  const controls = (audit) =>
    withoutTimings(
      audit.rows
        .filter((row) => row.ranking === 'constant')
        .sort((a, b) => a.asset.localeCompare(b.asset)),
    );
  if (!isDeepStrictEqual(controls(before), controls(after)))
    throw Error('constant-ranking control changed despite frozen placement policy');
  const perAsset = comparison.base.assets.map((base, i) => {
    const candidate = comparison.candidate.assets[i];
    return {
      asset: base.asset,
      category: base.category,
      relative_retained_improvement:
        (base.retained_ratio - candidate.retained_ratio) / base.retained_ratio,
      lods: base.triangles.slice(1).map((triangles, level) => ({
        level: level + 1,
        before: triangles,
        after: candidate.triangles[level + 1],
        relative_increase: candidate.triangles[level + 1] / triangles - 1,
      })),
    };
  });
  return {
    ...comparison,
    per_asset: perAsset,
    termination,
    constant_control_unchanged: true,
    relative_retained_improvement:
      comparison.retained_ratio_improvement / comparison.base.retained_ratio,
    worst_relative_lod_increase: Math.max(
      ...perAsset.flatMap((asset) => asset.lods.map((lod) => lod.relative_increase)),
    ),
    score: null,
  };
}
