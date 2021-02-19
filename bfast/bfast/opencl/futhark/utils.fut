let nextodd (x: f32): i64 =
  let x = f32.round x |> t32 |> i64.i32
  in
  x + (1 - (x % 2))

-- gather
let (&<) 't [N] [n] (vs: [N]t) (indxs: [n]i64): [n]t =
  map (\i -> vs[i]) indxs

let w64 (x: i64) : f32 =
  i32.i64 x |> r32

let fl_op (x: i64) (op: f32 -> f32): i64 =
  x |> w64 |> op |> i64.f32

let pad_gather [n] 'a (vs: [n]a) (idxs: [n]i64) (zero: a): [n]a =
  map (\i -> if i >= 0 then vs[i] else zero) idxs
