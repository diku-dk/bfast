let logplus (x: f64) : f64 =
  if x > (f64.exp 1)
  then f64.log x else 1

let adjustValInds [N] (n : i64) (ns : i64) (Ns : i64) (val_inds : [N]i64) (ind: i64) : i64 =
    if ind < Ns - ns then val_inds[ind + ns] - n else -1

let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : (i64, [n]t, [n]i64) =
  let tfs = map (\a -> if p a then 1i64 else 0i64) arr
  let isT = scan (+) 0i64 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT - 1 else -1i64) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n 0i64) inds (iota n)
  in (i, rs, ks)

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX_with_trend [N] (k2p2: i64) (f: f64) (mappingindices: [N]i64): [k2p2][N]f64 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f64
                else if i == 1 then f64.i64 ind
                else let (i', j') = (f64.i64 (i / 2), f64.i64 ind)
                     let angle = 2f64 * f64.pi * i' * j' / f
                     in  if i % 2 == 0 then f64.sin angle
                                       else f64.cos angle
            ) mappingindices
      ) (iota k2p2)

let mkX_no_trend [N] (k2p2m1: i64) (f: f64) (mappingindices: [N]i64): [k2p2m1][N]f64 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f64
                else let i = i + 1
                     let (i', j') = (f64.i64 (i / 2), f64.i64 ind)
                     let angle = 2f64 * f64.pi * i' * j' / f
                     in
                     if i % 2 == 0 then f64.sin angle
                                   else f64.cos angle
            ) mappingindices
      ) (iota k2p2m1)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-block parallelism                  --
---------------------------------------------------

let gauss_jordan [nm] (n:i64) (m:i64) (A: *[nm]f64): [nm]f64 =
  loop A for i < n do
      let v1 = A[i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then A[(k * m + j)] else
                            let x = A[j] / v1 in
                                if k < n - 1  -- Ap case
                                then A[((k + 1) * m + j)] - A[((k + 1) * m + i)] * x
                                else x        -- irow case
                   ) (iota nm)
      in  scatter A (iota nm) A'

let mat_inv [n0] (A: [n0][n0]f64): [n0][n0]f64 =
    let n  = n0
    let m  = 2 * n
    let nm = 2 * n0 * n0
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then A[i,j]
                                       else if j == n + i
                                            then 1
                                            else 0
                 ) (iota nm)
    let Ap'  = gauss_jordan n m Ap
    let Ap'' = unflatten n0 (m) Ap'
    -- Drop the identity matrix at the front
    in Ap''[0:n0, n0:(2 * n0)] :> [n0][n0]f64

--------------------------------------------------
--------------------------------------------------

let dotprod [n] (xs: [n]f64) (ys: [n]f64): f64 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f64) (ys: [m]f64) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f64) (xs: [n]f64) (ys: [n]f64) : f64 =
  f64.sum (map3 (\v x y -> x * y * if (f64.isnan v) then 0.0 else 1.0) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f64) (ys: [m]f64) =
    map (\xs -> map2 (\x y -> if (f64.isnan y) then 0 else x*y) xs ys |> f64.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f64) (yss: [p][m]f64) (vct: [p]f64) : [n][m]f64 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss
