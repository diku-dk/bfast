-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/sahara.in
-- output @ data/sahara-irreg.out

import "/futlib/math"
import "/futlib/array"
import "/futlib/linalg"

module f32linalg = linalg(f32)

let logplus (x: f32) : f32 =
  if x > (f32.exp 1f32)
  then f32.log x else 1f32

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX (k: i32) (N: i32) (f: f32) : [][N]f32 =
  [ replicate N 1f32        -- first   row
  , map r32 (1 ... N)       -- second  row
  ] ++
  ( map (\ i ->           -- sin/cos rows
          map (\j -> let i' = r32 (i / 2)
                     let j' = r32 j
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle else f32.cos angle
              ) (1 ... N)
        ) (2 ... 2*k+1)
  )

-- | compute the actual number of values that precisely
--   encapsulate the first n valid values of y.
let findSplit [N] (n: i32) (y: [N]f32) : i32 =
  let flgs = map (\v -> if f32.isnan v then 0i32 else 1i32) y
  let scnf = scan (+) 0 flgs
  let pairs= map (\(i,v)->(i,v==n)) 
                 (zip (0 ... N-1) scnf)
  let (m,b)= reduce (\(i1,b1) (i2,b2) ->
                        if b1 then (i1, b1)
                              else (i2, b2)
                    ) (0,false) pairs
  in  if b then m+1 else N


let dotprod_filt [n] (flgs: [n]bool) (xs: [n]f32) (ys: [n]f32) : f32 =
    reduce (+) 0 (map3 (\flg x y -> if flg then x*y else 0f32) flgs xs ys)

let matsqr [m] [n] (flgs: [n]bool) (xss: [m][n]f32) : *[m][m]f32 =
    map(\xs -> map (\ys -> dotprod_filt flgs xs ys) xss) xss


let matvecmul_row_filt [n][m] (flgs: [m]bool) (xss: [n][m]f32) (ys: [m]f32) =
    map (dotprod_filt flgs ys) xss

-- | The core of the alg: the computation for a time series
--   for one pixel.
let bfast [N] (f: f32) (k: i32) (n: i32) 
              (hfrac: f32) (lam: f32)
              (y: [N]f32) :
              ([]f32, []f32, []bool, []f32, []f32) =
  -- it's ok, don't panick: whatever is invariant to the 
  -- outer map is gonna be hoisted out! 
  -- (Just to advertize the compiler a bit)
  let X = mkX k N f

  let m    = n -- findSplit n y
  let flgs = map (\v -> !(f32.isnan v)) y

  let flgsh = unsafe (flgs[:m])
  let yh    = unsafe (y[:m])
  let Xh    = unsafe (X[:,:m])
  
  -- line 2, beta-hat computation 
  -- fit linear regression model:
  let Xsqr = matsqr flgsh Xh              -- [2k+2][2k+2]
  let Xinv = f32linalg.inv Xsqr           -- [2k+2][2k+2]
  let beta0= matvecmul_row_filt flgsh Xh yh     -- [2k+2]
  let beta = f32linalg.matvecmul_row Xinv beta0 -- [2k+2]
  
  -- line 3, y-hat computation: get predictions and residuals.
  -- the entries in y_pred corresponding to indices `i` such
  -- that `flgs[i]==false` are invalid. 
  let y_pred = f32linalg.matvecmul_row (transpose X) beta -- [N]


  -- line 4: error
  let y_error= map3 (\flg ye yep -> if flg then (ye-yep) else 0) flgs y y_pred -- [N]

  -- moving sums: 
  let h = t32 ( (r32 m) * hfrac )
  let MO_fst = reduce (+) 0f32 ( unsafe (y_error[n-h+1 : n+1]) )
  let MO_ini = map (\j ->
                      if j == 0 then MO_fst
                      else  unsafe (-y_error[n-h+j] + y_error[n+j]) 
                   ) (0 ... N-n-1)
  let MO = scan (+) 0f32 MO_ini

  -- line 5: sigma
  let tmp = map (\ a -> a*a ) (unsafe (y_error[:m]))

  let sigma = reduce (+) 0f32 tmp
  let sigma = f32.sqrt ( sigma / (r32 (n-2*k-2)) )

  let MO = map (\mo -> mo / (sigma * (f32.sqrt (r32 n))) ) MO              

  -- line 10: BOUND computation (will be hoisted)
  let BOUND = map (\q -> let t   = n+1+q
                         let tmp = logplus ((r32 t) / (r32 n))
                         in  lam * (f32.sqrt tmp)
                  ) (0 ... N-n-1)
  let breaks = map2 (\m b -> (f32.abs m) > b) MO BOUND
  in  (MO, BOUND, breaks, y_error, y_pred)


-- | entry point
entry main [m][N] (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (images : [m][N]f32) : 
                  ([m][]f32, [m][]f32, [m][]bool, [m][]f32, [m][]f32) =
  let res = map (bfast freq k n hfrac lam) images
  in unzip5 res

-- futhark-dataset -b --i32-bounds=3:3 -g i32 --i32-bounds=228:228 -g i32 --f32-bounds=12:12 -g f32 --f32-bounds=0.25:0.25 -g f32 --f32-bounds=0.2:0.2 -g f32 --f32-bounds=0.001:0.35 -g [67968][414]f32 > data/rand_data_67968_414.in

-- futhark-dataset -b --i32-bounds=3:3 -g i32 --i32-bounds=228:228 -g i32 --f32-bounds=12:12 -g f32 --f32-bounds=0.25:0.25 -g f32 --f32-bounds=0.2:0.2 -g f32 --f32-bounds=0.001:0.35 -g [32768][414]f32 > data/rand_data_32768_414.in
