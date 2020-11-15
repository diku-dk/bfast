-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/D1.in.gz
-- compiled input @ data/D2.in.gz
-- compiled input @ data/D3.in.gz
-- compiled input @ data/D4.in.gz
-- compiled input @ data/D5.in.gz
-- compiled input @ data/D6.in.gz
-- compiled input @ data/peru.in.gz

-- output @ data/peru.out.gz
-- compiled input @ data/sahara.in.gz
-- output @ data/sahara.out.gz

import "helpers"

-- | implementation is in this entry point
--   the outer map is distributed directly
let mainFun [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let n64 = i64.i32 n
  let k2p2 = 2*k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X =
    (if trend > 0
          then mkX_with_trend (i64.i32 k2p2') freq mappingindices
    else mkX_no_trend (i64.i32 k2p2') freq mappingindices)
    |> intrinsics.opaque


  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| i32.i64 <| (N*N + 2*N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <| map (map (+zero)) (copy (transpose X))

  let Xh  =  (X[:,:n64])
  let Xth =  (Xt[:n64,:])
  let Yh  =  (images[:,:n64])

  ----------------------------------
  -- 2. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = intrinsics.opaque <| map (matmul_filt Xh Xth) Yh

  ----------------------------------
  -- 3. matrix inversion          --
  ----------------------------------
  let Xinv = intrinsics.opaque <| map mat_inv Xsqr

  ---------------------------------------------
  -- 4. several matrix-vector multiplication --
  ---------------------------------------------
  let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
               |> intrinsics.opaque

  let beta   = map2 matvecmul_row Xinv beta0    -- [2k+2]
               |> intrinsics.opaque -- ^ requires transposition of Xinv
                                    --   unless all parallelism is exploited

  let y_preds= map (matvecmul_row Xt) beta      -- [N]
               |> intrinsics.opaque -- ^ requires transposition of Xt (small)
                                    --   can be eliminated by passing
                                    --   (transpose X) instead of Xt

  ---------------------------------------------
  -- 5. filter etc.                          --
  ---------------------------------------------
  let (Nss, y_errors, val_indss) = ( intrinsics.opaque <| unzip3 <|
    map2 (\y y_pred ->
            let y_error_all = zip y y_pred |>
                map (\(ye,yep) -> if !(f32.isnan ye)
                                  then ye-yep else f32.nan )
            let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
            let (y_error, val_inds) = unzip tups
            in  (Ns, y_error, val_inds)
         ) images y_preds )

  ------------------------------------------------
  -- 6. ns and sigma (can be fused with above)  --
  ------------------------------------------------
  let (hs, nss, sigmas) = intrinsics.opaque <| unzip3 <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then #[unsafe] y_error[i] else 0.0) (iota3232 n)
                        |> map (\ a -> a*a ) |> reduce (+) 0.0
            let sigma = f32.sqrt ( sigma / (r32 (ns-k2p2)) )
            let h     = t32 ( (r32 ns) * hfrac )
            in  (h, ns, sigma)
         ) Yh y_errors

  ---------------------------------------------
  -- 7. moving sums first and bounds:        --
  ---------------------------------------------
  let hmax = reduce_comm (i32.max) 0 hs
  let MO_fsts = zip3 y_errors nss hs |>
    map (\(y_error, ns, h) -> #[unsafe]
            map (\i -> if i < h then #[unsafe] y_error[i + ns-h+1] else 0.0) (iota3232 hmax)
            |> reduce (+) 0.0
        ) |> intrinsics.opaque

  let BOUND = map (\q -> let t   = n+1+q
                         let time = #[unsafe] mappingindices[t-1]
                         let tmp = logplus ((r32 time) / (r32 mappingindices[N-1]))
                         in  lam * (f32.sqrt tmp)
                  ) (iota32 (N-n64))

  ---------------------------------------------
  -- 8. moving sums computation:             --
  ---------------------------------------------
  let (MOs, MOs_NN, breaks, means) = zip (zip4 Nss nss sigmas hs) (zip3 MO_fsts y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma, h), (MO_fst,y_error,val_inds) ) ->
            -- let Nmn = N-n
            let MO = map (\j -> if j >= Ns-ns then 0.0
                                else if j == 0 then MO_fst
                                else #[unsafe] (-y_error[ns-h+j] + y_error[ns+j])
                         ) (iota32 (N - n64)) |> scan (+) 0.0

            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
          let (is_break, fst_break) =
        map3 (\mo' b j ->  if j < Ns - ns && !(f32.isnan mo')
              then ( (f32.abs mo') > b, j )
              else ( false, j )
             ) MO' BOUND (iota32 (N - n64))
            |> reduce (\ (b1,i1) (b2,i2) ->
                                if b1 then (b1,i1)
                                else if b2 then (b2, i2)
                                else (b1,i1)
                             ) (false, -1)
          let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (iota32 (N - n64))
          |> reduce (+) 0.0

          let fst_break' = if !is_break then -1
                             else adjustValInds n ns Ns val_inds fst_break
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'
-- The computation of MO'' should be moved just after MO' to make bounds consistent
            let val_inds' = map (adjustValInds n ns Ns val_inds) (iota32 (N - n64))
            let MO'' = scatter (replicate (N - n64) f32.nan) (map i64.i32 val_inds') MO'
            in (MO'', MO', fst_break', mean)
        ) |> unzip4

  in (MO_fsts, Nss, nss, sigmas, MOs, MOs_NN, BOUND, breaks, means, y_errors, y_preds)


-- | Entry points
entry mainDetailed [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  mainFun trend k n freq hfrac lam mappingindices images

entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  let (_, _, _, _, _, _, _, breaks, means, _, _) =
    mainFun trend k n freq hfrac lam mappingindices images
  in (breaks, means)

-- | implementation is in this entry point
--   the outer map is distributed directly
entry remove_nans [m][n][p] (nan_value: i16) (images : [m][n][p]i16) =
  map (\block ->
      map (\row ->
          map (\el -> if el == nan_value then f32.nan else f32.i16 el
            ) row
        ) block
    ) images

entry reshapeTransp [m][n][p] (images : [m][n][p]f32) : [][m]f32 =
  let images' = map (flatten_to (n * p)) images
  in  transpose images'
