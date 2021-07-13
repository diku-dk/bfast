-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- the datasets can be downloaded from:
-- https://github.com/diku-dk/futhark-kdd19/tree/master/bfast-futhark/data
-- ==
-- compiled input @ data/D1.in.gz
-- compiled input @ data/D2.in.gz
-- compiled input @ data/D3.in.gz
-- compiled input @ data/D4.in.gz
-- compiled input @ data/D5.in.gz
-- compiled input @ data/D6.in.gz
-- compiled input @ data/peru.in.gz

import "lib/github.com/diku-dk/sorts/insertion_sort"
import "helpers"
import "mroc"

-- | implementation is in this entry point
--   the outer map is distributed directly
let mainFun [m][N] (trend: i64) (k: i64) (n: i64) (freq: f64)
                  (hfrac: f64) (level: f64) (lam: f64) (hist: i64) (conf: f64)
                  (mappingindices : [N]i64)
                  (images : [m][N]f64) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let k2p2 = 2 * k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = (if trend > 0
           then mkX_with_trend (k2p2') freq mappingindices
           else mkX_no_trend (k2p2') freq mappingindices)
          |> intrinsics.opaque

  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = f64.i64 <| (N * N + 2 * N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <| map (map (+zero)) (copy (transpose X))

  let Xh  = X[:,:n]
  let Xth = Xt[:n,:]
  let Yh  = images[:,:n]
  ----------------------------------
  -- 2. stable history            --
  ----------------------------------
  let hist_inds = if hist == -1
                  then mhistory_roc level conf Xth Yh
                  else replicate m hist
  -- Set stable history period.
  let images = map2 (\j ->
                       map2 (\i yi -> if i < j then f64.nan else yi) (iota N)
                    ) hist_inds images
  let Yh  = images[:,:n]

  ----------------------------------
  -- 3. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = intrinsics.opaque <| map (matmul_filt Xh Xth) Yh

  ----------------------------------
  -- 4. matrix inversion          --
  ----------------------------------
  let Xinv = intrinsics.opaque <| map mat_inv Xsqr

  ---------------------------------------------
  -- 5. several matrix-vector multiplication --
  ---------------------------------------------
  let beta0  = map (matvecmul_row_filt Xh) Yh   -- [m][2k+2]
               |> intrinsics.opaque

  let beta   = map2 matvecmul_row Xinv beta0    -- [m][2k+2]
               |> intrinsics.opaque -- ^ requires transposition of Xinv
                                    --   unless all parallelism is exploited

  let y_preds= map (matvecmul_row Xt) beta      -- [m][N]
               |> intrinsics.opaque -- ^ requires transposition of Xt (small)
                                    --   can be eliminated by passing
                                    --   (transpose X) instead of Xt

  ---------------------------------------------
  -- 6. filter etc.                          --
  ---------------------------------------------
  let (Nss, y_errors, val_indss) = intrinsics.opaque <| unzip3 <|
    map2 (\y y_pred ->
            let y_error_all = map2 (\ye yep -> if !(f64.isnan ye)
                                               then ye - yep
                                               else f64.nan
                                   ) y y_pred
            in filterPadWithKeys (\y -> !(f64.isnan y)) f64.nan y_error_all
         ) images y_preds

  ------------------------------------------------
  -- 7. ns and sigma (can be fused with above)  --
  ------------------------------------------------
  let (hs, nss, sigmas) = intrinsics.opaque <| unzip3 <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f64.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then y_error[i] else 0.0) (iota n)
                        |> map (\a -> a * a) |> reduce (+) 0.0
            let sigma = f64.sqrt (sigma / (f64.i64 (ns - k2p2)))
            let h     = i64.f64 ((f64.i64 ns) * hfrac)
            in  (h, ns, sigma)
         ) Yh y_errors

  ---------------------------------------------
  -- 8. moving sums first and bounds:        --
  ---------------------------------------------
  let hmax = reduce_comm (i64.max) 0 hs
  let MO_fsts = zip3 y_errors nss hs
                |> map (\(y_error, ns, h) ->
                          map (\i -> if i < h
                                     then y_error[i + ns - h + 1]
                                     else 0.0
                              ) (iota hmax)
                          |> reduce (+) 0.0
                       ) |> intrinsics.opaque

  let BOUND = map (\q -> let time = mappingindices[n + q]
                         let tmp  = logplus ((f64.i64 time) / (f64.i64 mappingindices[n - 1]))
                         in  lam * (f64.sqrt tmp)
                  ) (iota (N-n))

  ---------------------------------------------
  -- 9. error magnitude computation:         --
  ---------------------------------------------
  let magnitudes = zip3 Nss nss y_errors |>
    map (\(Ns, ns, y_error) ->
            map (\i -> if i < Ns - ns && !(f64.isnan y_error[ns + i])
                       then y_error[ns + i]
                       else f64.inf
                ) (iota (N - n))
                -- sort
                |> insertion_sort (f64.<=)
                -- extract median
                |> (\xs -> let i = (Ns - ns) / 2
                           let j = i - 1
                           in
                           if Ns == ns
                           then 0f64
                           else if (Ns - ns) % 2 == 0
                                then (xs[j] + xs[i]) / 2
                                else xs[i])
        ) |> intrinsics.opaque

  ---------------------------------------------
  -- 10. moving sums computation:            --
  ---------------------------------------------
  let (MOs, MOs_NN, breaks, means) = zip (zip4 Nss nss sigmas hs)
                                         (zip3 MO_fsts y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma, h), (MO_fst,y_error,val_inds) ) ->
            let MO = map (\j -> if j >= Ns - ns then 0.0
                                else if j == 0 then MO_fst
                                else y_error[ns + j] - y_error[ns - h + j]
                         ) (iota (N - n)) |> scan (+) 0.0

            let MO' = map (\mo -> mo / (sigma * (f64.sqrt (f64.i64 ns))) ) MO
            let (is_break, fst_break) =
                map3 (\mo' b j ->  if j < Ns - ns && !(f64.isnan mo')
                                   then ( (f64.abs mo') > b, j )
                                   else ( false, j )
                     ) MO' BOUND (iota (N - n))
                |> reduce (\(b1, i1) (b2, i2) -> if b1 then (b1, i1)
                                                  else if b2 then (b2, i2)
                                                  else (b1, i1)
                          ) (false, -1)
            let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (iota (N - n))
                       |> reduce (+) 0.0
                       |> (\x -> if (Ns - ns) == 0 then 0f64 else x / (f64.i64 (Ns - ns)))
            let fst_break' = if !is_break then -1
                             else adjustValInds n ns Ns val_inds fst_break
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'
            -- The computation of MO'' should be moved just after MO' to make bounds consistent
            let val_inds' = map (adjustValInds n ns Ns val_inds) (iota (N - n))
            let MO'' = scatter (replicate (N - n) f64.nan) val_inds' MO'
            in (MO'', MO', fst_break', mean)
        ) |> unzip4

  in (MO_fsts, Nss, nss, sigmas, MOs, MOs_NN, BOUND, breaks, means, magnitudes, y_errors, y_preds, hist_inds)


-- | Entry points
entry mainDetailed [m][N] (trend: i64) (k: i64) (n: i64) (freq: f64)
                  (hfrac: f64) (level: f64) (lam: f64) (hist: i64) (conf: f64)
                  (mappingindices : [N]i64)
                  (images : [m][N]f64) =
  mainFun trend k n freq hfrac level lam hist conf mappingindices images

entry mainMagnitude [m][N] (trend: i64) (k: i64) (n: i64) (freq: f64)
                           (hfrac: f64) (level: f64) (lam: f64) (hist: i64) (conf: f64)
                           (mappingindices : [N]i64)
                           (images : [m][N]f64) =
  let (_, Nss, _, _, _, _, _, breaks, means, magnitudes, _, _, hist_inds) =
    mainFun trend k n freq hfrac level lam hist conf mappingindices images
  in (Nss, breaks, means, magnitudes, hist_inds)

entry main [m][N] (trend: i64) (k: i64) (n: i64) (freq: f64)
                  (hfrac: f64) (level: f64) (lam: f64) (hist: i64) (conf: f64)
                  (mappingindices : [N]i64)
                  (images : [m][N]f64) =
  let (_, Nss, _, _, _, _, _, breaks, means, _, _, _, hist_inds) =
    mainFun trend k n freq hfrac level lam hist conf mappingindices images
  in (Nss, breaks, means, hist_inds)

entry convertToFloat [m][n][p] (nan_value: i16) (images : [m][n][p]i16) =
  map (\block ->
         map (\row ->
                map (\el -> if el == nan_value then f64.nan else f64.i16 el) row
             ) block
      ) images

entry reshapeTransp [m][n][p] (images : [m][n][p]f64) : [][m]f64 =
  let images' = map (flatten_to (n * p)) images
  in  transpose images'
