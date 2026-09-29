// Freeze these training-only additions before the remote experiment. The rock
// adds a previously absent category; the others add intermediate pixel sizes.
export const freshConditions=Object.freeze([
  {asset:'ph_sweet_potato',pixels:32,states:24,previous:4,adjacent:2},
  {asset:'ph_painted_wooden_bench',pixels:128,states:16,previous:2,adjacent:3},
  {asset:'ph_namaqualand_boulder_04',pixels:32,states:16,previous:0,adjacent:3},
  {asset:'ph_namaqualand_boulder_04',pixels:128,states:8,previous:2,adjacent:3}
]);
