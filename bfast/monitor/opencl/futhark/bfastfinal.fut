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
  let k2p2 = 2 * k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = (if trend > 0
           then mkX_with_trend (i64.i32 k2p2') freq mappingindices
           else mkX_no_trend (i64.i32 k2p2') freq mappingindices)
          |> intrinsics.opaque

  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| i32.i64 <| (N * N + 2 * N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <| map (map (+zero)) (copy (transpose X))

  let Xh  = X[:,:n64]
  let Xth = Xt[:n64,:]
  let Yh  = images[:,:n64]

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
  -- 5. filter etc.                          --
  ---------------------------------------------
  let (Nss, y_errors, val_indss) = intrinsics.opaque <| unzip3 <|
    map2 (\y y_pred ->
            let y_error_all = map2 (\ye yep -> if !(f32.isnan ye)
                                               then ye - yep
                                               else f32.nan
                                   ) y y_pred
            in filterPadWithKeys (\y -> !(f32.isnan y)) f32.nan y_error_all
         ) images y_preds

  ------------------------------------------------
  -- 6. ns and sigma (can be fused with above)  --
  ------------------------------------------------
  let (hs, nss, sigmas) = intrinsics.opaque <| unzip3 <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then y_error[i] else 0.0) (iota3232 n)
                        |> map (\a -> a * a) |> reduce (+) 0.0
            let sigma = f32.sqrt (sigma / (r32 (ns - k2p2)))
            let h     = t32 ((r32 ns) * hfrac)
            in  (h, ns, sigma)
         ) Yh y_errors

  ---------------------------------------------
  -- 7. moving sums first and bounds:        --
  ---------------------------------------------
  let hmax = reduce_comm (i32.max) 0 hs
  let MO_fsts = zip3 y_errors nss hs
                |> map (\(y_error, ns, h) ->
                          map (\i -> if i < h
                                     then y_error[i + ns - h + 1]
                                     else 0.0
                              ) (iota3232 hmax)
                          |> reduce (+) 0.0
                       ) |> intrinsics.opaque

  let BOUND = map (\q -> let time = mappingindices[n + q]
                         let tmp  = logplus ((r32 time) / (r32 mappingindices[n - 1]))
                         in  lam * (f32.sqrt tmp)
                  ) (iota32 (N-n64))

  ---------------------------------------------
  -- 8. error magnitude computation:             --
  ---------------------------------------------
  let magnitudes = zip3 Nss nss y_errors |>
    map (\(Ns, ns, y_error) ->
            map (\i -> if i < Ns - ns && !(f32.isnan y_error[ns + i])
                       then y_error[ns + i]
                       else f32.inf
                ) (iota32 (N - n64))
                -- sort
                |> insertion_sort (f32.<=)
                -- extract median
                |> (\xs -> let i = (Ns - ns) / 2
                           let j = i - 1
                           in
                           if Ns == ns
                           then 0f32
                           else if (Ns - ns) % 2 == 0
                                then (xs[j] + xs[i]) / 2
                                else xs[i])
        ) |> intrinsics.opaque

  ---------------------------------------------
  -- 9. moving sums computation:             --
  ---------------------------------------------
  let (MOs, MOs_NN, breaks, means) = zip (zip4 Nss nss sigmas hs)
                                         (zip3 MO_fsts y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma, h), (MO_fst,y_error,val_inds) ) ->
            let MO = map (\j -> if j >= Ns - ns then 0.0
                                else if j == 0 then MO_fst
                                else y_error[ns + j] - y_error[ns - h + j]
                         ) (iota32 (N - n64)) |> scan (+) 0.0

            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
            let (is_break, fst_break) =
                map3 (\mo' b j ->  if j < Ns - ns && !(f32.isnan mo')
                                   then ( (f32.abs mo') > b, j )
                                   else ( false, j )
                     ) MO' BOUND (iota32 (N - n64))
                |> reduce (\(b1, i1) (b2, i2) -> if b1 then (b1, i1)
                                                  else if b2 then (b2, i2)
                                                  else (b1, i1)
                          ) (false, -1)
            let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (iota32 (N - n64))
                       |> reduce (+) 0.0
                       |> (\x -> if (Ns - ns) == 0 then 0f32 else x / (r32 (Ns - ns)))
            let fst_break' = if !is_break then -1
                             else adjustValInds n ns Ns val_inds fst_break
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'
            -- The computation of MO'' should be moved just after MO' to make bounds consistent
            let val_inds' = map (adjustValInds n ns Ns val_inds) (iota32 (N - n64))
            let MO'' = scatter (replicate (N - n64) f32.nan) (map i64.i32 val_inds') MO'
            in (MO'', MO', fst_break', mean)
        ) |> unzip4

  in (MO_fsts, Nss, nss, sigmas, MOs, MOs_NN, BOUND, breaks, means, magnitudes, y_errors, y_preds)


-- | Entry points
entry mainDetailed [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  mainFun trend k n freq hfrac lam mappingindices images

entry mainMagnitude [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                           (hfrac: f32) (lam: f32)
                           (mappingindices : [N]i32)
                           (images : [m][N]f32) =
  let (_, Nss, _, _, _, _, _, breaks, means, magnitudes, _, _) =
    mainFun trend k n freq hfrac lam mappingindices images
  in (Nss, breaks, means, magnitudes)

entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  let (_, Nss, _, _, _, _, _, breaks, means, _, _, _) =
    mainFun trend k n freq hfrac lam mappingindices images
  in (Nss, breaks, means)

entry convertToFloat [m][n][p] (nan_value: i16) (images : [m][n][p]i16) =
  map (\block ->
         map (\row ->
                map (\el -> if el == nan_value then f32.nan else f32.i16 el) row
             ) block
      ) images

entry reshapeTransp [m][n][p] (images : [m][n][p]f32) : [][m]f32 =
  let images' = map (flatten_to (n * p)) images
  in  transpose images'
