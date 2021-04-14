import "utils"

-- simple linear interpolation
let simple_interp [n] (xs: [n]f32) (ys: [n]f32) (x: f32) : f32 =
  -- first, find the segment
  let seg =
    let (idx, _) =
      -- tables are typically small, hence sequential
      loop (curr, cont) = (0, true) while (curr < n && cont) do
      let cont = x >= xs[curr]
      let curr = if cont then curr + 1 else curr
      in
      (curr, cont)
    in idx - 1
  in
  -- is it the last segment?
  if seg >= n - 1
  then
    -- use the last value of y
    ys[n - 1]
  else
    -- do linear interpolation between two points
    let x_0 = xs[seg]
    let x_1 = xs[seg + 1]
    let y_0 = ys[seg]
    let y_1 = ys[seg + 1]
    let frac = (x - x_0) / (x_1 - x_0)
    in
    y_0 * (1 - frac) + y_1 * frac

-- find p-value by performing a statistical test for a OLS-MOSUM process
let mosum_test [n] [k] [n_t] (X: [k][n]f32)
                        (y: [n]f32)
                        (nn_idx: [n]i64)
                        (n_nn: i64)
                        (h: f32)
                        (tableipl: [n_t]f32)
                        (tablep: [n_t]f32) =
  -- fit linear model
  let Xt   = transpose X
  let Xsqr = matmul_filt X Xt y
  let Xinv = mat_inv Xsqr
  let beta0 = matvecmul_row_filt X y
  let beta  = matvecmul_row Xinv beta0

  -- calculate errors of the fitted model
  let y_preds = matvecmul_row Xt beta
  let y_error_all = map2 (\ye yep -> if !(f32.isnan ye)
                                     then ye - yep
                                     else f32.nan
                         ) y y_preds
  let y_error = pad_gather y_error_all nn_idx 0f32

  -- parameters for the MOSUM process
  let sigma = f32.sqrt <| f32.sum <| map (\i -> i**2 / w64 (n_nn - k)) y_error
  let nh_nn = i64.f32 <| f32.floor ((w64 n_nn) * h)

  -- We don't add 0, since we only need the statistic
  let cusum = scan (+) 0f32 y_error

  -- length of the mosum process
  let n_mosum = n_nn - nh_nn
  -- calculate the MOSUM process from the cumulative sum of the errors
  let mosum1 = cusum[nh_nn:n_nn] :> [n_mosum]f32
  let mosum2 = cusum[:n_mosum] :> [n_mosum]f32
  let sum_diff = map2 (-) mosum1 mosum2
  let mosum = map (/(sigma * (f32.sqrt <| w64 n_nn))) sum_diff

  -- calculate the test statistic and p_value
  let stat = f32.maximum <| map f32.abs mosum
  let p_value = simple_interp tableipl tablep stat
  in p_value

let main =
  let get_x [n] (_: [n]f32) : [2][n]f32 =
    let ones = replicate n 1f32
    let x = iota n |> map (+1) |> map w64
    let X = row_stack ones x
    in X

  let y =
    -- [1120, 1160,  963, 1210, 1160, 1160,  813, 1230, 1370, 1140,  995,  935, 1110,  994,
    --  1020,  960, 1180,  799,  958, 1140, 1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100,
    --   774,  840,  874,  694,  940,  833,  701,  916,  692, 1020, 1050,  969,  831,  726,
    --   456,  824,  702, 1120, 1100,  832,  764,  821,  768,  845,  864,  862,  698,  845,
    --   744,  796, 1040,  759,  781,  865,  845,  944,  984,  897,  822, 1010,  771,  676,
    --   649,  846,  812,  742,  801, 1040,  860,  874,  848,  890,  744,  749,  838, 1050,
    --   918,  986,  797,  923,  975,  815, 1020,  906,  901, 1170,  912,  746,  919,  718,
    --   714,  740] :> []f32
    [1120, f32.nan,  963, 1210, 1160, 1160,  813, 1230, 1370, 1140,  995,  935, 1110,  994,
     1020,  960, 1180,  799,  958, 1140, 1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100,
      774,  840,  874,  694,  940,  833,  701,  916,  692, 1020, 1050,  969,  831,  726,
      456,  824,  702, 1120, 1100,  832,  764,  821,  768,  845,  864,  862,  698,  845,
      744,  796, 1040,  759,  781,  865,  845,  944,  984,  897,  822, 1010,  771,  676,
      649,  846,  812,  742,  801, 1040,  860,  874,  848,  890,  744,  749,  838, 1050,
      918,  986,  797,  923,  975,  815, 1020,  906,  901, 1170,  912,  746,  919,  718,
      714,  740] :> []f32
  let X = get_x y
  let (_, nn_idx, n_nn) = filterPadNans y
  let tableipl = [0f32, 1.1211f32, 1.2059f32, 1.2845f32, 1.3767f32]
  let tablep = [1f32, 0.1f32, 0.05f32, 0.025f32, 0.01f32]
  in
  mosum_test X y nn_idx n_nn 0.15f32 tableipl tablep
