-- 0d/1d Interpolation
let interp [N] (m: [N]i32) (fits: [N]f32) (slopes: [N]f32) (n: i64) : [n]f32 =
  -- calculate indexes for the at array
  let is = iota n
  -- then, calculate indexes for the m array
  let zeros = replicate n 0i64
  let ones = replicate N 1i64
  let ms = map (\j ->
                  if j == 0 || j > N - 2
                  -- scatter ignores invalid indexes, hence -1
                  then -1
                  else m[j]
               ) (iota N) |> map (i64.i32)
  let sc = scatter zeros ms ones
  let js = scan (+) 0i64 sc
  in
  map2 (\i j ->
          -- formula from stlplus/src/interp.cpp, 'at' is always 1..n
          let h = r32 (m[j + 1] - m[j])
          let u = (r32 ((i32.i64 i) - m[j] + 1)) / h
          let u2 = u * u
          let u3 = u2 * u
          in
          (2 * u3 - 3 * u2 + 1) * fits[j] +
          (3 * u2 - 2 * u3)     * fits[j + 1] +
          (u3 - 2 * u2 + u)     * slopes[j] * h +
          (u3 - u2)             * slopes[j + 1] * h
       ) is js


-- let main =
--   let m = [1,3,5,7]
--   let fits = [1.1, 3.4, 5.3, 7.8]
--   let slopes = [0, 0, 0, 0]
--   in interp m fits slopes 7
