-- Moving averages
let single_ma (n_p: i64) (n: i64) (x: []f32) : [n]f32 =
  let ma_tmp = map (\i -> x[i]) (iota n_p) |> reduce (+) 0f32
  let n_p_f = i32.i64 n_p |> r32
  in
  map (\i ->
         if i == 0
         then
           ma_tmp / n_p_f
         else
           (x[i + n_p - 1] - x[i - 1]) / n_p_f
      ) (iota n) |> scan (+) 0


let moving_averages [n] (x: [n]f32) (n_p: i64): []f32 =
  let nn = n - n_p * 2
  in
  -- apply three moving averages
  single_ma n_p (nn + n_p + 1) x |> single_ma n_p (nn + 2) |> single_ma 3 nn


let main =
  let x = [1.2, 2.3, 3.2, 4.1, 5.3, 6.1, 7.3, 8.1, 9.3, 10.12]
  let n_p = 2
  in
  moving_averages x n_p
