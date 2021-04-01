import "helpers"

let efp_p_value [n] [k] [n_t] (X: [n][k]f32) (y: [n]f32) (h: f32) (tableipl: [n_t]f32) (tablep: [n_t]f32): f32 =
  -- fit linear model
  let Xt  = transpose X |> trace
  -- let Xsqr = matmul_filt X Xt
  -- let Xinv = mat_inv Xsqr
  -- let beta0  = matvecmul_row_filt X y
  -- let beta   = matvecmul_row Xinv beta0

  -- let y_preds= matvecmul_row Xt beta |> trace

  -- let (Nss, e, val_indss) = ( intrinsics.opaque <| unzip3 <|
  --   map2 (\y y_pred ->
  --           let y_error_all = zip y y_pred |>
  --               map (\(ye, yep) -> if !(f32.isnan ye)
  --                                 then ye - yep
  --                                 else f32.nan
  --                   )
  --           let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
  --           let (y_error, val_inds) = unzip tups
  --           in  (Ns, y_error, val_inds)
  --        ) images y_preds )

  -- let (hs, nss, sigmas) = intrinsics.opaque <| unzip3 <|
  --   map2 (\yh y_error ->
  --           let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
  --                       |> reduce (+) 0
  --           let sigma = map (\i -> if i < ns then y_error[i] else 0.0) (iota3232 n)
  --                       |> map (\a -> a * a) |> reduce (+) 0.0
  --           let sigma = f32.sqrt (sigma / (r32 (ns - k2p2)))
  --           let h     = t32 ((r32 ns) * hfrac)
  --           in  (h, ns, sigma)
  --        ) Yh y_errors


  -- let sigma = f32.sqrt <| f32.sum <| map (\i -> i**2 / (n - k)) e

  -- let nh = i64.f32 <| f32.floor ((w64 n) * h)

  -- let e_zero = [0f32] ++ e
  -- let process = scan (+) 0f32 e_zero
  -- let process = map2 (-) process[nh:] process[:(n - nh + 1)]
  -- let process = map (/ (sigma * (f32.sqrt n))) process

  -- let stat = f32.maximum <| map f32.abs process
  -- let p_value = interp stat tableipl tablep
  -- in
  -- p_value
  in 0.0


let main =
  let X = [ [1, 2, 3],
            [1, 1, 1]
          ]
  let y = [3, 5, 7]
  in
  efp_p_value X y 0 [] []

