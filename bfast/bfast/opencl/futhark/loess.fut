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

let l_indexes [n_m] (N: i64) (n: i64) (span: i64) (m: [n_m]i64) : [n_m]i64 =
  let s2 = (span + 1) / 2
  in
  if N - 1 < span
  then
    replicate n_m 0
  else
    map ( \i ->
            if i < s2 - 1 then 0
            else if i >= s2 - 1 && i <= n - 1 - s2 then i - s2 + 1
            else n - span
        ) m

let loess [N] [n_m] (y: [N]f32)        -- observations
                    (span: i64)        -- should be odd
                    (m: [n_m]i64)      -- indexes at which to apply the loess
                    (y_idx: [N]i64)    -- indexes of non-nan vals in y
                    (n: i64)           -- n non-nan vals
                    : [N]f32 =
  let l_idx = l_indexes N n span m
  let nan_cusum = map (\mi -> if f32.isnan y[mi] then 1i64 else 0) m |> scan (+) 0
  let l_idx_nn = map2 (\i j -> i64.min (i64.max (i - j) 0) (n - span)) l_idx nan_cusum
  let r_idx_nn = map (+(span - 1)) l_idx_nn
  let md = map3 (\l mv r ->
                   i64.max (i64.abs (y_idx[l] - mv)) (i64.abs (y_idx[r] - mv))
                ) l_idx_nn m r_idx_nn |> map w64
  let max_dist = if span >= n
                 then
                   map (\x -> x + ((w64 span) - (w64 n)) / 2f32) md
                 else
                   md
  let (results, slopes) = loess_proc (map w64 y_idx)
                                     (pad_gather y y_idx 0f32)
                                     span
                                     m
                                     l_idx_nn
                                     max_dist
  in
  interp m results slopes N


-- let main =
--   let y = [1.2, 2.3, 3.2, 4.1, 5.3, 6.3, 7.1, 8.3, 9.1, 10.3, 11.1, 12, 13, 14, 15, 16, 17, 18, 19, 20]
--   -- let y = [1.2, f32.nan, 3.2, 4.1, 5.3, 6.3, f32.nan, 8.3, 9.1, 10.3, 11.1, 12, 13, 14, 15, 16, 17, 18, 19, 20]
--   -- let y = [1.2, f32.nan, 3.2, 4.1, 5.3, 6.3, f32.nan, 8.3, 9.1, 10.3, 11.1, 12, 13, 14, 15, 16, 17, 18, 19, 20]
--   -- let y = [f32.nan, f32.nan, 3.2, f32.nan, 5.3, 6.3, f32.nan, 8.3, 9.1, 10.3, 11.1, 12, 13, 14, 15, 16, 17, f32.nan, 19, 20]

--   let span = 7
--   -- let m = iota 20
--   let m = (0..2...19) |> trace
--   let (_, y_idx, n_nn) = filterPadNans y
--   -- let y_idx = tabulate 10 (\i -> if (f32.isnan y[i]) then -1 else i) |> filter (>=0)
--   -- let n_nn = length y_idx
--   -- let y_idx = y_idx ++ (replicate (10 - n_nn) 0i64) :> [10]i64
--   in
--   loess y span m y_idx n_nn |> trace
