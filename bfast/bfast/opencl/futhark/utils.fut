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

-- gather for the padded (with -1) indexes
let pad_gather [n] 'a (vs: [n]a) (idxs: [n]i64) (zero: a): [n]a =
  map (\i -> if i >= 0 then vs[i] else zero) idxs

-- returns:
--- rs: array of matching values, padded with 0 of that type
--- ks: array of indexes of matching values, padded with -1
--- n : number of matching values
let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : ([n]t, [n]i64, i64) =
  let tfs = map (\a -> if p a then 1i64 else 0i64) arr
  let isT = scan (+) 0i64 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT - 1 else -1i64) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n (-1i64)) inds (iota n)
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
  in
  argmin xs op f32.highest

let gauss_jordan [nm] (n: i64) (m: i64) (A: *[nm]f32): [nm]f32 =
  loop A for i < n do
      let v1 = A[i]
      let A' = tabulate nm (\ind ->
                              let (k, j) = (ind / m, ind % m)
                              in
                              if v1 == 0.0
                              then A[k * m + j]
                              else
                                let x = A[j] / v1
                                in
                                if k < n - 1  -- Ap case
                                then A[(k + 1) * m + j] - A[(k + 1) * m + i] * x
                                else x        -- irow case
                           )
      in scatter A (iota nm) A'

let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m  = 2 * n
    let nm = n * m
    -- Pad the matrix with the identity matrix.
    let Ap = tabulate nm (\ind ->
                            let (i, j) = (ind / m, ind % m)
                            in
                            if j < n
                            then A[i,j]
                            else
                              if j == n + i
                              then 1
                              else 0
                         )
    let Ap'  = gauss_jordan n m Ap
    let Ap'' = unflatten n m Ap'
    -- Drop the identity matrix at the front
    in Ap''[:n, n:(2 * n)] :> [n][n]f32

let column_stack [n] (xs1: [n]f32) (xs2: [n]f32): [n][2]f32 =
  map2 (
    \x1 x2 -> [x1, x2]
  ) xs1 xs2

let row_stack [n] (xs1: [n]f32) (xs2: [n]f32): [2][n]f32 =
  xs1 ++ xs2 |> unflatten 2 n

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\v x y -> x * y * if (f32.isnan v) then 0.0 else 1.0) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss
