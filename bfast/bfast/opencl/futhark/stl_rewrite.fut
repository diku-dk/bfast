-- ==
-- compiled input @ co2nonan.in


import "loess"
import "interp"
import "utils"
import "ma"

let win_to_jump (x: i64): i64 =
  fl_op x ((/10) >-> f32.ceil)

let create_m (jump: i64) (n: i64): []i64 =
  let ev_p = 0..jump...(n-1) :> []i64
  in
  if (last ev_p) != n - 1
  then ev_p ++ [n - 1]
  else ev_p

let stl [m][n] (Y_l: [m][n]f32) (n_p: i64): ([m][n]f32, [m][n]f32, [m][n]f32) =
  -- set parameters for the seasonal LOESS smoothing
  let l_window = nextodd <| w64 <| n_p
  let l_jump = win_to_jump l_window
  let l_ev = create_m l_jump n

  -- set parameters for the trend LOESS smoothing
  let t_window = nextodd <| f32.ceil (1.5 * (w64 n_p)/(1 - 1.5 / (10 * (w64 n) + 1)))
  let t_jump = win_to_jump t_window
  let t_ev = create_m t_jump n

  let (_, nn_idx_l, n_nn_l) = map (\Y -> filterPadNans Y) Y_l |> unzip3
  let s = m * n

  -- cycle_sub_indices will keep track of what part of the
  -- seasonal each observation belongs to
  let n_sub = i64.f32 <| f32.ceil <| ((w64 n) / (w64 n_p))

  let cycle_sub_indices = (flatten <| replicate n_sub (iota n_p))[:n]

  let (seasonal_l, trend_l) =
    -- Inner loop
    -- initialize trend to 0s
    loop (_, trend_l) = ((replicate s 0f32 |> unflatten m n), (replicate s 0f32 |> unflatten m n)) for _i_inner < 2 do
      let cycle_sub_avgs_l =
        map2 (\Y trend ->
          -- Step 1: Detrending
          let Y_detrended = map2 (-) Y trend

          -- Step 2: Smoothing of cycle-subseries
          let cycle_sub_avgs =
            tabulate n_p (\i ->
              let (cs, cs_nns) = tabulate n (\j ->
                                               let yj = Y_detrended[j]
                                               let csij = cycle_sub_indices[j]
                                               let (v, n_v_nn) =
                                                 if csij == i && !(f32.isnan yj)
                                                 then (yj, 1f32)
                                                 else (0f32, 0f32)
                                               in (v, n_v_nn)
                                            ) |> unzip2
              in
              (f32.sum cs) / (f32.sum cs_nns)
            )
          in cycle_sub_avgs
        ) Y_l trend_l |> opaque

      let C_dim1 = n + 2 * n_p
      let C_dim2 = n_sub + 1
      let C_l =
        map (\cycle_sub_avgs ->
               let C_ = (flatten <| replicate C_dim2 cycle_sub_avgs)[:n + n_p]
               let C = C_ ++ C_[n:(n + n_p)] :> [C_dim1]f32
               in C
             ) cycle_sub_avgs_l |> opaque
      let ma3_l =
        map (\C ->
             -- Step 3: Low-pass filtering of collection of all the cycle-subseries
             -- apply 3 moving averages
             let ma3 = moving_averages C n_p :> [n]f32
             in ma3
            ) C_l |> opaque
      in
      map5 (\Y nn_idx n_nn C ma3 ->
          -- then apply LOESS
          let L = loess ma3 l_window l_ev nn_idx l_jump n_nn :> [n]f32

          -- Step 4: Detrend smoothed cycle-subseries
          -- start and end indices for after adding in extra n.p before and after
          let nd = n + n_p
          let c_slice = C[n_p:nd] :> [n]f32

          let seasonal = map2 (-) c_slice L

          -- Step 5: Deseasonalize
          let D = map2 (-) Y seasonal

          -- Step 6: Trend Smoothing
          let trend = loess D t_window t_ev nn_idx t_jump n_nn :> [n]f32
          in (seasonal, trend)
      ) Y_l nn_idx_l n_nn_l C_l ma3_l |> unzip2

  let remainder_l = map3 (\Y seasonal trend ->
                            map3 (\y s t -> y - s - t) Y seasonal trend
                         ) Y_l seasonal_l trend_l
  in (seasonal_l, trend_l, remainder_l)

entry main [m][n] (input: [m][n]f32) (period: i64): ([m][n]f32, [m][n]f32, [m][n]f32) =
  stl input period
