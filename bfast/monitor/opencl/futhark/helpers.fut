let iota32 (x: i64) : []i32 =
  iota x |> map i32.i64

let iota3232 (x: i32) : []i32 =
  iota (i64.i32 x) |> map i32.i64

let logplus (x: f32) : f32 =
  if x > (f32.exp 1)
  then f32.log x else 1

let adjustValInds [N] (n : i32) (ns : i32) (Ns : i32) (val_inds : [N]i32) (ind: i32) : i32 =
    if ind < Ns - ns then (#[unsafe] val_inds[ind+ns]) - n else -1

let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : ([n](t,i32), i32) =
  let tfs = map (\a -> if p a then 1i64 else 0i64) arr
  let isT = scan (+) 0i64 tfs
  let i   = last isT |> i32.i64
  let inds= map2 (\a iT -> if p a then iT-1 else -1i64) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n 0i32) inds (iota32 n)
  in (zip rs ks, i)

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX_with_trend [N] (k2p2: i64) (f: f32) (mappingindices: [N]i32): [k2p2][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else if i == 1 then r32 ind
                else let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f
                     in  if i % 2 == 0 then f32.sin angle
                                       else f32.cos angle
            ) mappingindices
      ) (iota32 k2p2)

let mkX_no_trend [N] (k2p2m1: i64) (f: f32) (mappingindices: [N]i32): [k2p2m1][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else let i = i + 1
         let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f
                     in  if i % 2 == 0 then f32.sin angle
                                       else f32.cos angle
            ) mappingindices
      ) (iota32 k2p2m1)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-blockparallelism                   --
---------------------------------------------------

  let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
    loop A for i < n do
      let v1 = A[i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then #[unsafe] A[k*m+j] else
                            let x = #[unsafe] (A[j] / v1) in
                                if k < n-1  -- Ap case
                                then #[unsafe] ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
                                else x      -- irow case
                   ) (iota32 nm)
      in  scatter A (iota nm) A'

  let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m = 2*n
    let nm= n*m
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then #[unsafe] A[i,j]
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (iota nm)
    let Ap' = gauss_jordan (i32.i64 n) (i32.i64 m) Ap
    -- Drop the identity matrix at the front!
    let Ap'' = unflatten n m Ap'
    in
    tabulate_2d n n (\i j -> #[unsafe] Ap''[i, n + j])
--------------------------------------------------
--------------------------------------------------

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

-- | implementation is in this entry point
--   the outer map is distributed directly
entry remove_nans [m][n][p] (nan_value: i16) (images : [m][n][p]i16) =
  map (\block ->
      map (\row ->
          map (\el -> if el == nan_value then f32.nan else f32.i16 el
            ) row
        ) block
    ) images
