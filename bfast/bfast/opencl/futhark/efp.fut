import "interp"

let efp_p_value [n] [k] [n_t] (X: [n][k]f32) (y: [n]f32) (h: f32) (tableipl: [n_t]f32) (tablep: [n_t]f32): f32 =
  -- fit linear model
  let fm = sm.OLS(y, X, missing='drop').fit()
  let e = y - fm.predict(exog=X)
  let sigma = f32.sqrt <| f32.sum <| map (\i -> i**2 / (n - k)) e

  let nh = i64.f32 <| f32.floor ((w64 n) * h)
  let e_zero = [0f32] ++ e
  let process = scan (+) 0f32 e_zero
  let process = map2 (-) process[nh:] process[:(n - nh + 1)]
  let process = map (/ (sigma * (f32.sqrt n))) process

  let stat = f32.maximum <| map f32.abs process
  let p_value = interp stat tableipl tablep
  in
  p_value
