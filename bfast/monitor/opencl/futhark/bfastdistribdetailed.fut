-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/sahara.in.gz
-- output @ data/sahara.out.gz

let logplus (x: f32) : f32 =
  if x > (f32.exp 1)
  then f32.log x else 1

let partitionCos [n] 't                    
           (p : (t -> bool))
           (dummy : t)           
           (arr : [n]t) : ([n]t, i32) =
  let cs  = map p arr
  let tfs = map (\f -> if f then 1 else 0) cs
  let ffs = map (\f -> if f then 0 else 1) cs
  let isT = scan (+) 0 tfs
  let isF0= scan (+) 0 ffs

  let i   = last isT  
  let isF = map (+i) isF0
  let inds= map3 (\c iT iF ->            
                    if c then iT-1 
                         else iF-1
                 ) cs isT isF
  let r = scatter (replicate n dummy) inds arr            
  in  (r, i) 

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX_with_trend [N] (k2p2: i32) (f: f32) (mappingindices: [N]i32): [k2p2][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else if i == 1 then r32 ind
                else let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle 
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2)

let mkX_no_trend [N] (k2p2m1: i32) (f: f32) (mappingindices: [N]i32): [k2p2m1][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else let i = i + 1
		     let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle 
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2m1)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-blockparallelism                   --
---------------------------------------------------

  let gauss_jordan [nm] (n:i32) (A: *[nm]f32): [nm]f32 =
    let m = nm / n in
    loop A for i < n do
      let v1 = A[i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then unsafe A[k*m+j] else 
                            let x = unsafe (A[j] / v1) in
                                if k < n-1  -- Ap case
                                then unsafe ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
                                else x      -- irow case
                   ) (iota (n*m))
      in  scatter A (iota (n*m)) A'

  let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m = 2*n
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then unsafe ( A[i,j] )
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (iota (n*m))
    let Ap' = unflatten n m (gauss_jordan n Ap)

    -- Drop the identity matrix at the front.
    in Ap'[0:n,n:n * 2]
--------------------------------------------------
--------------------------------------------------

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\v x y -> if (f32.isnan v) then 0 else x*y) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
    map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss

----------------------------------------------------
----------------------------------------------------

-- | implementation is in this entry point
--   the outer map is distributed directly
entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let k2p2 = 2*k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = intrinsics.opaque <|
	  if trend > 0
          then mkX_with_trend k2p2' freq mappingindices
	  else mkX_no_trend   k2p2' freq mappingindices
  

  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| (N*N + 2*N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <|
            map (map (+zero)) (copy (transpose X))

  let Xh  = unsafe (X[:,:n])
  let Xth = unsafe (Xt[:n,:])
  let Yh  = unsafe (images[:,:n])
  
  ----------------------------------
  -- 2. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = intrinsics.opaque <|
             map (matmul_filt Xh Xth) Yh

  ----------------------------------
  -- 3. matrix inversion          --
  ----------------------------------
  let Xinv = intrinsics.opaque <|
             map mat_inv Xsqr

  ---------------------------------------------
  -- 4. several matrix-vector multiplication --
  ---------------------------------------------
  let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
               |> intrinsics.opaque
  let beta   = map2 matvecmul_row Xinv beta0    -- [2k+2]
               |> intrinsics.opaque
  let y_preds= map (matvecmul_row Xt) beta      -- [N]
               |> intrinsics.opaque

  ---------------------------------------------
  -- 5. filter etc.                          --
  ---------------------------------------------
  let (Nss, y_errors, val_indss) = unsafe ( intrinsics.opaque <| unzip3 <|
    map2 (\y y_pred ->
            let y_error_all = zip y y_pred |>
                map (\(ye,yep) -> if !(f32.isnan ye) 
                                  then ye-yep else f32.nan )
            let (tups, Ns) = zip2 y_error_all (iota N) |>
                partitionCos (\(ye,_) -> !(f32.isnan ye)) (0.0, 0)
            let (y_error, val_inds) = unzip tups
            in  (Ns, y_error, val_inds)
         ) images y_preds )

  ---------------------------------------------
  -- 6. ns and sigma                         --
  ---------------------------------------------
  let (nss, sigmas) = intrinsics.opaque <| unzip <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then unsafe y_error[i] else 0.0) (iota n)
                        |> map (\ a -> a*a ) |> reduce (+) 0.0
            let sigma = f32.sqrt ( sigma / (r32 (ns-k2p2)) )
            in  (ns, sigma)
         ) Yh y_errors

  ---------------------------------------------
  -- 7. moving sums first and bounds:        --
  ---------------------------------------------
  let hs = map (\ns -> t32 ( (r32 ns) * hfrac ) ) nss
  let MO_fsts = zip3 y_errors nss hs |>
    map (\(y_error, ns, h) -> unsafe
            map (\i -> unsafe y_error[i + ns-h+1]) (iota h) 
            |> reduce (+) 0.0 
        ) |> intrinsics.opaque

  let BOUND = map (\q -> let t   = n+1+q
                         let time = unsafe mappingindices[t-1]
                         let tmp = logplus ((r32 time) / (r32 mappingindices[N-1]))
                         in  lam * (f32.sqrt tmp)
                  ) (0 ... N-n-1)


  --let BOUND = map (\q -> let t   = n+q
  --                       let tmp = logplus ((r32 t) / (r32 n))
  --                       in  lam * (f32.sqrt tmp)
  --                ) (0 ... N-n-1)

  ---------------------------------------------
  -- 8. moving sums computation:             --
  ---------------------------------------------
  let (MOs, MOs_NN, breaks, means) = zip (zip4 Nss nss sigmas hs) (zip3 MO_fsts y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma, h), (MO_fst,y_error,val_inds) ) ->
            let MO = map (\j -> if j >= Ns-ns then 0.0
                                else if j == 0 then MO_fst
                                else  unsafe (-y_error[ns-h+j] + y_error[ns+j])
                         ) (0 ... N-n-1) |> scan (+) 0.0
	    
            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
            let val_inds' = map (\i ->  if i < Ns - ns 
                                        then (unsafe val_inds[i+ns]) - n
                                        else -1
                                ) (0 ... N-n-1)
	        let (is_break, fst_break) = 
		    map3 (\mo b j ->  if j < Ns - ns && !(f32.isnan mo)
				      then ( (f32.abs mo) > b, j )
				      else ( false, j )
		         ) MO' BOUND (iota (N-n))
		        |> reduce_comm (\ (b1,i1) (b2,i2) -> 
               				    if b1 then (b1,i1) 
              			          else if b2 then (b2, i2)
    				              else (b1,i1) 
              	      	     ) (false, -1)
	        let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (0...N-n-1)
			    |> reduce (+) 0.0

	        let fst_break' = if is_break then unsafe val_inds'[fst_break] else -1 
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'


            let MO'' = scatter (replicate (N-n) f32.nan) val_inds' MO'
            -- let breaks = map2 (\m b -> if (f32.isnan m) || (f32.isnan b)
            --                            then 0.0 else (f32.abs m) - b 
            --                            -- used to be nan instead of 0.0
            --                  ) MO'' BOUND
            in (MO'', MO', fst_break', mean)
        ) |> unzip4
  in (MO_fsts, Nss, nss, sigmas, MOs, MOs_NN, BOUND, breaks, means, y_errors, y_preds)

-- FUTHARK_INCREMENTAL_FLATTENING=1 ~/WORK/futhark/tools/futhark-autotune --compiler=futhark-opencl --pass-option=--default-tile-size=16 --stop-after 1500 --calc-timeout bfast-distr-work.fut --compiler=futhark-opencl

-- futhark-bench --skip-compilation bfast-distr-work.fut --pass-option --size=suff_intra_par_22492=4 --pass-option --size=suff_outer_par_21231=327779 --pass-option --size=suff_outer_par_23584=71005 --pass-option --size=suff_intra_par_19433=118 --pass-option --size=suff_intra_par_21269=16 --pass-option --size=suff_outer_par_17642=2174979 --pass-option --size=suff_intra_par_17688=114 --pass-option --size=suff_intra_par_23808=207 --pass-option --size=suff_outer_par_19992=271872 --pass-option --size=suff_intra_par_20032=114 --pass-option --size=suff_outer_par_22454=14069376 --pass-option --default-tile-size=16 -r 20

-- ./bfast-distr-work -t /dev/stderr  --size=suff_intra_par_22492=4  --size=suff_outer_par_21231=327779  --size=suff_outer_par_23584=71005  --size=suff_intra_par_19433=118  --size=suff_intra_par_21269=16  --size=suff_outer_par_17642=2174979  --size=suff_intra_par_17688=114  --size=suff_intra_par_23808=207  --size=suff_outer_par_19992=271872  --size=suff_intra_par_20032=114  --size=suff_outer_par_22454=14069376  --default-tile-size=16 -r 20 < data/sahara.in > /dev/null



-- | implementation is in this entry point
--   the outer map is distributed directly
entry remove_nans [m][n][p] (nan_value: i16) (images : [m][n][p]i16) = 
	map (\block -> 
			map (\row -> 
					map (\el -> if el == nan_value then f32.nan else f32.i16 el
						) row
				) block
		) images
		
entry reshapeTransp [m][n][p] (images : [m][n][p]f32) : [][m]f32 =
  let images' = map flatten images
  in  transpose images'
  
  