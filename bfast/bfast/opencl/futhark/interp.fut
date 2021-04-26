import "utils"

-- 0d/1d Interpolation
let interp [N] (m: [N]i64) (fits: [N]f32) (slopes: [N]f32) (n: i64) : [n]f32 =
  -- calculate indexes for the m array
  let zeros = replicate n 0i64
  let ones = replicate N 1i64
  let ms = tabulate N (\j ->
                         if j == 0 || j > N - 2
                         -- scatter ignores invalid indexes, hence -1
                         then -1
                         else m[j] + 1
                      )
  let sc = scatter zeros ms ones
  let js = scan (+) 0i64 sc
  in
  map2 (\i j ->
          -- formula from stlplus/src/interp.cpp, Assume 'at' is always 1..n
          let h = w64 (m[j + 1] - m[j])
          let u = (w64 (i - m[j])) / h
          let u2 = u * u
          let u3 = u2 * u
          in
          (2 * u3 - 3 * u2 + 1) * fits[j] +
          (3 * u2 - 2 * u3)     * fits[j + 1] +
          (u3 - 2 * u2 + u)     * slopes[j] * h +
          (u3 - u2)             * slopes[j + 1] * h
       ) (iota n) js

-- let main =
--   let n = 10
--   let m = (0..2...19) :> [n]i64
--   let fits = [1.2294472f32, 3.2195842f32, 5.218707f32, 7.222566f32, 9.202555f32, 11.115634f32, 13.010007f32, 15.000002f32, 17.000002f32, 19.0f32] :> [n]f32
--   let slopes = [0.9934316f32, 0.9988706f32, 1.0238093f32, 0.9695239f32, 1.0000004f32, 0.9261906f32, 0.9847617f32, 1.0f32, 1.0000002f32, 1.0f32] :> [n]f32
--   in
--   interp m fits slopes 20 |> trace

-- let main =
--   let m = [1,3,5,7]
--   let fits = [1.1, 3.4, 5.3, 7.8]
--   let slopes = [0, 0, 0, 0]
--   in interp m fits slopes 7
