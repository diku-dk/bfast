nextodd <- function(x) {
  x <- round(x)
  x2 <- ifelse(x %% 2 == 0, x + 1, x)
  as.integer(x2)
}

stlplus.default <- function(Y, t = NULL, n.p) {
  n <- length(Y)

  y_idx <- !is.na(Y)
  noNA <- all(y_idx)

  l.window <- nextodd(n.p)
  l.jump = ceiling(l.window / 10)
  l.degree = 1

  t.window <- nextodd(ceiling(1.5 * n.p/(1 - 1.5 / (10 * n + 1))))
  t.jump = ceiling(t.window / 10)
  t.degree = 1

  # Trend vector - initialize to 0 or NA, depending on what's in Y
  trend <- 0

  # cycleSubIndices will keep track of what part of the
  # seasonal each observation belongs to
  cycleSubIndices <- rep(c(1:n.p), ceiling(n / n.p))[1:n]

  C <- rep(NA, n + 2 * n.p)

  w <- rep(1, n)

 for(iter in 1:2) {

   # step 1: detrending...
   Y.detrended <- Y - trend

   # step 2: smoothing of cycle-subseries
   for(i in 1:n.p) {
     cycleSub <- Y.detrended[cycleSubIndices == i]
     subWeights <- w[cycleSubIndices == i]
     cycleSub.length <- length(cycleSub)

     cs1 <- head(cycleSubIndices, n.p)
     cs2 <- tail(cycleSubIndices, n.p)

     C[c(cs1, cycleSubIndices, cs2) == i] <- rep(weighted.mean(cycleSub,
         w = w[cycleSubIndices == i], na.rm = TRUE), cycleSub.length + 2)
   }

   # Step 3: Low-pass filtering of collection of all the cycle-subseries
   # moving averages
   ma3 <- c_ma(C, n.p)

   l.ev <- seq(1, n, by = l.jump)
   if(tail(l.ev, 1) != n) l.ev <- c(l.ev, n)
   L <- .loess_stlplus(y = ma3, span = l.window, degree = l.degree,
     m = l.ev, weights = w, y_idx = y_idx, noNA = noNA, jump = l.jump, at = c(1:n))

   # Step 4: Detrend smoothed cycle-subseries
   # start and end indices for after adding in extra n.p before and after
   st <- n.p + 1
   nd <- n + n.p

   seasonal <- C[st:nd] - L

   # Step 5: Deseasonalize
   D <- Y - seasonal

   # Step 6: Trend Smoothing
   t.ev <- seq(1, n, by = t.jump)
   if(tail(t.ev, 1) != n) t.ev <- c(t.ev, n)
   trend <- .loess_stlplus(y = D, span = t.window, degree = t.degree,
     m = t.ev, weights = w, y_idx = y_idx, noNA = noNA,
     jump = t.jump, at = c(1:n))
 }

  # compute remainder
  R <- Y - seasonal - trend
}

