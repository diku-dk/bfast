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

-- returns n-length padded arrays of values and indexes; and number of matching values
let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : ([n]t, [n]i64, i64) =
  let tfs = map (\a -> if p a then 1i64 else 0i64) arr
  let isT = scan (+) 0i64 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT - 1 else -1i64) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n 0i64) inds (iota n)
  in (rs, ks, i)

let filterPadNans = filterPadWithKeys (\i -> !(f32.isnan i)) 0f32

let argmin [n] 'a (xs: [n]a) (leq_op: a -> a -> bool) (max_val: a): i64 =
  let tpl_arr = zip xs (indices xs)
  let (_, min_idx) =
    reduce_comm (
      \(v_1, i_1) (v_2, i_2) -> if leq_op v_1 v_2 then (v_1, i_1) else (v_2, i_2)
    ) (max_val, 0i64) tpl_arr
  in
  min_idx

let nanargmin [n] (xs: [n]f32) : i64 =
  let op (a: f32) (b: f32) =
    let v_n = (a <= b)
    in
    if !(f32.isnan a) && !(f32.isnan b)
    then v_n
    else true

  let tpl_arr = zip xs (indices xs)
  let (_, min_idx) =
    reduce_comm (
      \(v_1, i_1) (v_2, i_2) -> if op v_1 v_2 then (v_1, i_1) else (v_2, i_2)
    ) (f32.highest, 0i64) tpl_arr
  in
  min_idx
