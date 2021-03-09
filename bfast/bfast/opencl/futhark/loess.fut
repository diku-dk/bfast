import "interp"
import "utils"

let loess_proc [n] [n_m] (xx: [n]f32)          -- time values - should be 1:n unless there are nans
                         (yy: [n]f32)          -- the corresponding y values
                         (span: i64)           -- span of smoothing
                         (m: [n_m]i64)         -- points at which to evaluate the smooth
                         (l_idx: [n_m]i64)     -- index of left starting points
                         (max_dist: [n_m]f32)  -- distance between nn bounds for each point
                         : ([n_m]f32, [n_m]f32) =
  map3 (\i_l_idx i_m i_max_dist ->
         -- get weights, x, and a
         let (x, w) =
           tabulate span (\j ->
                            let x_j = xx[i_l_idx + j] - (r32 <| i32.i64 i_m)
                            -- tricube
                            let r = f32.abs x_j
                            let tmp1 = r / i_max_dist
                            let tmp2 = 1.0 - tmp1 ** 3
                            let tmp3 = tmp2 ** 3
                            in
                            (x_j, tmp3)
                         ) |> unzip
         let a = f32.sum w
         let xw = map2 (*) x w
         let x2w = map2 (*) x xw
         let b = f32.sum xw
         let c = f32.sum x2w
         let det = 1 / (a * c - b * b)
         let a1 = c * det
         let b1 = -b * det
         let c1 = a * det

         let fun (par1: f32) (par2: f32) : f32 =
           tabulate span (\j -> (w[j] * par1 + xw[j] * par2) * yy[i_l_idx + j]) |> f32.sum

         let result = fun a1 b1
         let slopes = fun b1 c1
         in (result, slopes)
       ) l_idx m max_dist |> unzip


let rl_indexes [n_m] (N: i64) (n: i64) (span: i64) (m: [n_m]i64) : ([n_m]i64, [n_m]i64) =
  let s2 = (span + 1) / 2
  in
  if N - 1 < span
  then
    replicate n_m (0, n - 1) |> unzip
  else
    map ( \i ->
            -- this is how stlplus does it
            let l = if i < s2 - 1 then 0
                    else if i >= s2 - 1 && i <= n - 1 - s2 then i - s2 + 1
                    else n - span
            let r = l + span - 1
            in (l, r)
        ) m |> unzip

let loess [N] [n_m] (y: [N]f32)        -- observations
                    (span: i64)        -- should be odd
                    (m: [n_m]i64)      -- indexes at which to apply the loess
                    (y_idx: [N]i64)    -- indexes of non-nan vals in y
                    (jump: i64)        -- how many values to skip
                    (n: i64)
                    : [N]f32 =
  let noNaNs = n == N
  let x = iota N
  let (l_idx, r_idx) = rl_indexes N n span m
  let (result, slopes) =
    if noNaNs
    then
      let aa = tabulate n_m (\i -> i64.abs (m[i] - x[l_idx[i]]))
      let bb = tabulate n_m (\i -> i64.abs (x[r_idx[i]] - m[i]))
      let md = map2 i64.max aa bb
      let max_dist =
        if span >= n
        then map (\x -> (w64 x) + ((w64 span) - (w64 n)) / 2f32) md
        else map w64 md
      in
      loess_proc (map w64 x)
                 y
                 span
                 m
                 l_idx
                 max_dist
    else
      let x2 = pad_gather x y_idx 0i64
      -- this is NOT how stlplus does it
      let nan_flags = map (\a -> if f32.isnan a then 1i64 else 0) y
      let nan_cusum = scan (+) 0 nan_flags
      let l_idx_nn = map (\i -> i64.max (i - nan_cusum[i]) 0) l_idx
      let r_idx_nn = map (+(span - 1)) l_idx_nn

      let max_dist = map3 (\l mv r ->
                             i64.max (i64.abs (x2[l] - mv)) (i64.abs (x2[r] - mv))
                          ) l_idx_nn m r_idx_nn |> map w64
      in
      loess_proc (map w64 x2)
                 (pad_gather y y_idx 0f32)
                 span
                 m
                 l_idx_nn
                 max_dist
  in
  if jump > 1
  then interp (map (i32.i64) m) result slopes N
  else result :> [N]f32


let main =
  -- let y = [f32.nan, 2.3, 3.2, f32.nan, 5.3, f32.nan, f32.nan, 8.3, 9.1, 10.3]
  let y = [1.2, 2.3, 3.2, 4.1, 5.3, 6.3, 7.1, 8.3, 9.1, 10.3]
  let span = 5
  let m = iota 10
  -- let m = [0, 2, 4, 6, 8]
  let y_idx = tabulate 10 (\i -> if (f32.isnan y[i]) then -1 else i) |> filter (>=0)
  let n_nn = length y_idx
  let y_idx = y_idx ++ (replicate (10 - n_nn) 0i64) :> [10]i64

  let jump = 1
  -- let jump = 2
  in
  loess y span m y_idx jump n_nn

-- let main =
--   let xx = [1, 2, 3, 4, 5]
--   let yy = [1.2, 2.3, 3.2, 4.1, 5.3]
--   let degree = 1
--   let span = 2
--   let ww = [1, 1, 1, 1, 1]
--   -- let m = [5, 7]
--   let m = [2, 3, 4]
--   -- let l_idx = [1, 2]
--   let l_idx = [0, 1, 2]
--   -- let max_dist = [3.43, 5.32]
--   let max_dist = [3.43, 5.32, 6.1]
--   in
--   loess_proc xx yy degree span ww m l_idx max_dist
