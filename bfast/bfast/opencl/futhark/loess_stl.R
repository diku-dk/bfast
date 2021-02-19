nextodd <- function(x) {
  x <- round(x)
  x2 <- ifelse(x %% 2 == 0, x + 1, x)
  as.integer(x2)
}

## n <- length(y[y_idx])
## span should be odd
## .loess_stlplus <- function(y, span, degree, weights, m, n, n_m, y_idx, noNA, jump, at) {
.loess_stlplus <- function(x = NULL, y, span, degree, weights = NULL,
                           m = c(1:length(y)), y_idx = !is.na(y), noNA = all(y_idx), blend = 0,
                           jump = ceiling(span / 10), at = c(1:length(y))) {
  x <- c(1:length(y))
  n <- length(y[y_idx])

  s2 <- (span + 1) / 2
  print(span)
  if(noNA) {
    if((diff(range(x))) < span) {
      l_idx <- rep(1, n_m) # ones n_m
      r_idx <- rep(n, n_m) # repeat n n_m
    } else{
      l_one <- rep(1, length(m[m < s2]))
      l_two <- m[m >= s2 & m <= n - s2] - s2 + 1
      l_three <- rep(n - span + 1, length(m[m > n - s2]))
      print(s2)

      l_idx <- c(l_one,
                 l_two,
                 l_three)

      r_idx <- l_idx + span - 1
    }
    aa <- abs(m - x[l_idx])
    bb <- abs(x[r_idx] - m)
    max_dist <- ifelse(aa > bb, aa, bb)

    print(m)
    print(l_idx)
    print(r_idx)
    print(max_dist)

  } else {
    span3 <- min(span, n)
    x2 <- x[y_idx]

    # another approach
    a <- yaImpute::ann(ref = as.matrix(x2), target = as.matrix(m), tree.type = "kd",
      k = span3, eps = 0, verbose = FALSE)$knnIndexDist[,1:span3]


    l_idx <- apply(a, 1, min)
    r_idx <- apply(a, 1, max)

    max_dist <- apply(cbind(abs(m - x2[l_idx]), abs(x2[r_idx] - m)), 1, max)
    print(m)
    print(y)
    ## print(l_idx)
    print(x2[l_idx])
    ## print(r_idx)
    print(x2[r_idx])
    print(max_dist)
  }
  if(span >= n)
    max_dist <- max_dist + (span - n) / 2

  ## out <- c_loess(x[y_idx], y[y_idx], degree, span, weights[y_idx],
  ##   m, l_idx - 1, as.double(max_dist))

  ## res1 <- out$result
  ## # do interpolation
  ## if(jump > 1)
  ##   res1 <- .interp(m, out$result, out$slope, at)

  ## res1
  NA
}


## ma3 <- c(1.2, 2.3, 3.2, 4.1, 5.3, 6.3, 7.1, 8.3, 9.1, 10.3)
## ma3 <- c(1.2, 2.3, 3.2, 4.1, 5.3, 6.3, NA, 8.3, 9.1, 10.3)
## ma3 <- c(1.2, 2.3, 3.2, NA, 5.3, NA, NA, 8.3, 9.1, 10.3)
## ma3 <- c(NA, 2.3, 3.2, NA, 5.3, NA, NA, 8.3, 9.1, 10.3)
ma3 <- c(1.2, 2.3, 3.2, 4.1, 5.3, 6.3, 7.1, 8.3, 9.1, 10.3)
l.degree <- 1
l.window <- 5
l.ev <- c(1,2,3,4,5,6,7,8,9,10)

L <- .loess_stlplus(y = ma3, span = l.window, degree = l.degree)
