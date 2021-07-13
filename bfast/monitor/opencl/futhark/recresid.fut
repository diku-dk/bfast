import "lib/github.com/nhey/lm/lm"
import "helpers"

module lm = lm_f64

let mean_abs [n] (xs: [n]f64) =
  (reduce (+) 0 (map f64.abs xs)) / (f64.i64 n)

-- BFAST strucchange-cpp armadillo version of approx. equal, which is
-- an absolute difference |x - y| <= tol.
let approx_equal x y tol =
  (mean_abs (map2 (-) x y)) <= tol

let filter_nan_pad = filterPadWithKeys ((!) <-< f64.isnan) f64.nan

-- Map-distributed `recresid`.
entry mrecresid_nn [m][N][k] (Xs_nn: [m][N][k]f64) (ys_nn: [m][N]f64) =
  let tol = f64.sqrt(f64.epsilon) / (f64.i64 k)

  -- Initialise recursion by fitting on first `k` observations.
  let (X1s, betas, ranks) =
    map2 (\X_nn y_nn ->
            let model = lm.fit (transpose X_nn[:k, :]) y_nn[:k]
            in (model.cov_params, model.params, model.rank)
         ) Xs_nn ys_nn |> unzip3

  let num_recresids_padded = N - k
  let rets = replicate (m*num_recresids_padded) 0
             |> unflatten num_recresids_padded m

  let loop_body (r: i64) (X1: [k][k]f64) (beta: [k]f64)
                (X_nn: [N][k]f64) (y_nn: [N]f64) =
    -- Compute recursive residual
    let x = X_nn[r, :]
    let d = matvecmul_row X1 x
    let fr = 1 + (dotprod x d)
    let resid = y_nn[r] - dotprod x beta
    let recresid_r = resid / f64.sqrt(fr)
    -- Update formulas
    -- X1 = X1 - ddT/fr
    -- beta = beta + X1 x * resid
    let X1 = map2 (\d1 -> map2 (\d2 x -> x - (d1*d2)/fr) d) d X1
    let beta = map2 (+) beta (map (dotprod x >-> (*resid)) X1)
    in (X1, beta, recresid_r)

  -- Map is interchanged so that it is inside the sequential loop.
  let (_, r', X1s, betas, _, retsT) =
    loop (check, r, X1s, betas, ranks, rets_r) = (true,k,X1s,betas,ranks,rets)
      while check && r < N - 1 do
        let (checks, X1s, betas, ranks, recresids_r) = unzip5 <|
          map5 (\X1 beta X_nn y_nn rank ->
                  let (_, beta, recresid_r) = loop_body r X1 beta X_nn y_nn
                  -- Check numerical stability (rectify if unstable)
                  let (check, X1, beta, rank) =
                    -- We check update formula value against full OLS fit
                    let rp1 = r+1
                    -- NOTE We only need the transposed versions for the
                    -- first few iterations; I think it is more efficient
                    -- to transpose locally here because the matrix will
                    -- most definitely fit entirely in scratchpad memory.
                    -- Also we get to read from the array 1-strided.
                    let model = lm.fit (transpose X_nn[:rp1, :]) y_nn[:rp1]
                    -- Check that this and previous fit is full rank.
                    -- R checks nans in fitted parameters to same effect.
                    -- Also, yes it really is necessary to check all this.
                    let nona = !(f64.isnan recresid_r) && rank == k
                                                       && model.rank == k
                    let check = !(nona && approx_equal model.params beta tol)
                    -- Stop checking on all-nan ("empty") pixels.
                    let check = check && !(all f64.isnan y_nn)
                    in (check, model.cov_params, model.params, model.rank)
                  in (check, X1, beta, rank, recresid_r)
               ) X1s betas Xs_nn ys_nn ranks
        let rets_r[r-k, :] = recresids_r
        in (reduce_comm (||) false checks, r+1, X1s, betas, ranks, rets_r)

  let (_, _, retsT) =
    loop (X1s, betas, rets_r) = (X1s, betas, retsT) for r in (r'..<N) do
      let (X1s, betas, recresidrs) =
        unzip3 (map4 (loop_body r) X1s betas Xs_nn ys_nn)
      let rets_r[r-k, :] = recresidrs
      in (X1s, betas, rets_r)

  in retsT

-- Map-distributed `recresid`. There may be nan values in `ys`.
entry mrecresid [m][N][k] (X: [N][k]f64) (ys: [m][N]f64) =
  -- Rearrange `ys` so that valid values come before nans.
  let (ns, ys_nn, indss_nn) = unzip3 (map filter_nan_pad ys)
  -- Upper bound on number of non-nans
  let Nbar = i64.maximum ns
  -- Repeat this for `X`.
  let Xs_nn: *[m][Nbar][k]f64 =
    map (\j ->
           map (\i -> if i >= 0 then X[i, :] else replicate k f64.nan) indss_nn[j,:Nbar]
        ) (iota m)
  -- Subset ys
  let ys_nn = ys_nn[:,:Nbar]

  in (mrecresid_nn Xs_nn ys_nn, Nbar, ns)
