-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/peru.in.gz

-- compiled input @ data/D1.in.gz
-- compiled input @ data/D2.in.gz
-- compiled input @ data/D3.in.gz
-- compiled input @ data/D4.in.gz
-- compiled input @ data/D5.in.gz
-- compiled input @ data/D6.in.gz

-- output @ data/peru.out.gz
-- compiled input @ data/sahara.in.gz
-- output @ data/sahara.out.gz

-- REMEMBER TO RUN:
-- futhark pkg add github.com/diku-dk/sorts
-- futhark pkg sync
-- import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/sorts/insertion_sort"
import "helpers"

-- | implementation is in this entry point
--   the outer map is distributed directly
-- let mainFun [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
--                   (hfrac: f32) (lam: f32)
--                   (mappingindices : [N]i32)
--                   (images : [m][N]f32) =

--   let n64 = i64.i32 n
--   let magnitudes = images |>
--     map (\y_error ->
--             let magnitude = map (\i -> if !(f32.isnan y_error[i])
--                                        then y_error[i]
--                                        else f32.inf
--                                 ) (iota32 (N - n64))
--                             -- sort
--                             |> radix_sort_float f32.num_bits f32.get_bit
--                             |> \x -> x[0]
--             in magnitude
--         )
--   in magnitudes

let mainFun [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =

  let n64 = i64.i32 n
  let magnitudes = images |>
    map (\y_error ->
            let magnitude = map (\i -> if !(f32.isnan y_error[i])
                                       then y_error[i]
                                       else f32.inf
                                ) (iota32 (N - n64))
                            -- sort
                            |> insertion_sort (f32.<=)
                            |> \x -> x[0]
            in magnitude
        )
  in magnitudes

entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  mainFun trend k n freq hfrac lam mappingindices images
