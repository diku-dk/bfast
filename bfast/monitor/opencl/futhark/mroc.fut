import "lib/github.com/diku-dk/statistics/statistics"
import "recresid"

module stats = mk_statistics f64

-- Compute the [sample standard
-- deviation](https://en.wikipedia.org/wiki/Standard_deviation),
-- `s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}`,
-- but ignoring nans.
let sample_sd_nan [m] (xs: [m]f64) (num_non_nan: i64) =
  let nf64 = f64.i64 num_non_nan
  -- add_nan is associative.
  -- TODO prove commutativity and use `reduce_comm` (probably no speedup)
  let add_nan a b = if f64.isnan a
                    then b
                    else if f64.isnan b
                         then a
                         else a + b
  let x_mean = (reduce add_nan 0 xs)/nf64
  let diffs = map (\x -> if f64.isnan x then 0 else (x - x_mean)**2) xs
  in (f64.sum diffs)/(nf64 - 1) |> f64.sqrt

-- Empircal fluctuation process containing recursive residuals.
-- Outputs recursive CUSUM and number of non nan values _excluding_
-- the prepended zero for each `y` in `ys`.
let rcusum [m][N][k] (X: [N][k]f64) (ys: [m][N]f64) =
  let (wTs, Nbar, ns) = mrecresid X ys
  let ws = transpose wTs
  -- Standardize and insert 0 in front.
  let Nmk = Nbar-k+1
  let (process, ns) = unzip <|
    map2 (\w npk ->
           let n = npk - k -- num non nan recursive residuals
           let s = sample_sd_nan w n
           let fr = s * f64.sqrt(f64.i64 n)
           let sdized = map (\j -> if j == 0 then 0 else w[j-1]/fr) (iota Nmk)
           in (scan(+) 0f64 sdized, n)
        ) ws ns
  in (process, Nbar, ns)

let std_normal_cdf =
  stats.cdf (stats.mk_normal {mu=0f64, sigma=1f64})

-- TODO this may be done cheaper for x < 0.3; see
--      R strucchange `pvalue.efp`.
--      Also not saving intermediate results.
let pval_brownian_motion_max (x: f64): f64 =
  -- Q is complementary CDF of N(0,1)
  let Q = \y -> 1 - std_normal_cdf y
  let exp = \y -> f64.e**y
  in 2 * (Q(3*x) + exp(-4*x**2) - exp(-4*x**2) * (Q x))

-- Structural change test for Brownian motion.
-- `num_non_nan` is without the initial zero in process.
let sctest [n] (process: [n]f64) (num_non_nan: i64) : f64 =
  let nf64 = f64.i64 num_non_nan
  let xs = process[1:]
  -- x = max(abs(xs * 1/(1 + 2*j))) where j = 1/n, 2/n, ..., n.
  let div i = 1 + (f64.i64 (2*i+2)) / nf64
  let x = map2 (\x i -> f64.abs (x/(div i))) xs (indices xs) |> f64.maximum
  in pval_brownian_motion_max x

-- `N` is padded length.
-- `nm1` is number of non-nan values excluding inital zero.
let boundary confidence N nm1: [N]f64 =
  let n = nm1 + 1
  -- conf*(1 + 2*t) with t in [0,1].
  let div = f64.i64 n - 1
  in map (\i -> if i < n
                then confidence + (2*confidence*(f64.i64 i))/div
                else f64.nan
         ) (iota N)

-- Map distributed stable history computation.
entry mhistory_roc [m][N][k] level confidence
                             (X: [N][k]f64) (ys: [m][N]f64) =
  let (rocs, Nbar, nns) = rcusum (reverse X) (map reverse ys)
  let pvals = map2 sctest rocs nns
  let n = Nbar - k + 1
  let bounds = map (boundary confidence n) nns
  -- index of first time roc crosses the boundary
  let inds =
    map2 (\roc bound ->
            let nm1 = n - 1
            let roc = roc[1:] :> [nm1]f64
            let bound = bound[1:] :> [nm1]f64
            in map3 (\i r b ->
                       if f64.abs r > b
                       then i
                       else i64.highest
                    ) (iota nm1) roc bound
               |> reduce_comm i64.min i64.highest
         ) rocs bounds
  in map3 (\ind nn pval ->
            let chk = !(f64.isnan pval) && pval < level && ind != i64.highest
            let y_start = if chk then nn - ind else 0
            in y_start
          ) inds nns pvals

-- Not faster, even though I reduce memory access.
-- Compiler seems to be better at fusing kernels than me.
entry mhistory_roc_inline [m][N][k] level confidence
                             (X: [N][k]f64) (ys: [m][N]f64) =
  -- RCUSUM
  -- Empircal fluctuation process containing recursive residuals.
  -- Outputs recursive CUSUM and number of non nan values _excluding_
  -- the prepended zero for each `y` in `ys`.
  let (wTs, Nbar, ns) = mrecresid (reverse X) (map reverse ys)
  let ws = transpose wTs
  let Nmkp1 = Nbar-k+1
  in map2 (\w npk -> -- INNER SIZE Nmkp1 (can be split into Nmkp1 and Nmkp1-1)
             -- RCUSUM CONT.
             -- Standardize and insert 0 in front.
             let n = npk - k -- num non nan recursive residuals
             let s = sample_sd_nan w n
             let fr = s * f64.sqrt(f64.i64 n)
             -- TODO never use prepended 0, but changing this results in
             -- significantly poorer performance. Some weird bug I think.
             let sdized = map (\j -> if j == 0 then 0 else w[j-1]/fr) (iota Nmkp1)
             let roc = scan (+) 0f64 sdized
             -- SCTEST
             -- Structural change test for Brownian motion.
             -- `num_non_nan` is without the initial zero in process.
             let nf64 = f64.i64 n
             let xs = roc[1:]
             let div i = 1 + (f64.i64 (2*i+2)) / nf64
             let x = f64.maximum <| map2 (\i x -> f64.abs (x/(div i))) (indices xs) xs
             let pval = pval_brownian_motion_max x
             -- BOUNDARY
             -- conf*(1 + 2*t) with t in [0,1].
             -- INDEX OF CROSSING
             let ind = map2 (\i r ->
                        if f64.abs r > (confidence + (2*confidence*(f64.i64 i+1))/nf64)
                        then i
                        else i64.highest
                     ) (indices xs) xs
                |> reduce_comm i64.min i64.highest
             -- INDEX OF HISTORY START
             let chk = !(f64.isnan pval) && pval < level && ind != i64.highest
             let y_start = if chk then n - ind else 0
             in y_start
          ) ws ns
