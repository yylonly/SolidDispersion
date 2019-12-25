#' Maximum Dissimilarity Sampling
#' 
#' Functions to create a sub-sample by maximizing the dissimilarity between new
#' samples and the existing subset.
#' 
#' Given an initial set of m samples and a larger pool of n samples, this
#' function iteratively adds points to the smaller set by finding with of the n
#' samples is most dissimilar to the initial set. The argument \code{obj}
#' measures the overall dissimilarity between the initial set and a candidate
#' point. For example, maximizing the minimum or the sum of the m
#' dissimilarities are two common approaches.
#' 
#' This algorithm tends to select points on the edge of the data mainstream and
#' will reliably select outliers. To select more samples towards the interior
#' of the data set, set \code{randomFrac} to be small (see the examples below).
#' 
#' @aliases maxDissim minDiss sumDiss
#' @param a a matrix or data frame of samples to start
#' @param b a matrix or data frame of samples to sample from
#' @param n the size of the sub-sample
#' @param obj an objective function to measure overall dissimilarity
#' @param useNames a logical: should the function return the row names (as
#' opposed ot the row index)
#' @param randomFrac a number in (0, 1] that can be used to sub-sample from the
#' remaining candidate values
#' @param verbose a logical; should each step be printed?
#' @param \dots optional arguments to pass to dist
#' @param u a vector of dissimilarities
#' @return a vector of integers or row names (depending on \code{useNames})
#' corresponding to the rows of \code{b} that comprise the sub-sample.
#' @author Max Kuhn \email{max.kuhn@@pfizer.com}
#' @seealso \code{\link{dist}}
#' @references Willett, P. (1999), "Dissimilarity-Based Algorithms for
#' Selecting Structurally Diverse Sets of Compounds," \emph{Journal of
#' Computational Biology}, 6, 447-457.
#' @keywords utilities
#' @examples
#' 
#' 
#' example <- function(pct = 1, obj = minDiss, ...)
#' {
#'   tmp <- matrix(rnorm(200 * 2), nrow = 200)
#' 
#'   ## start with 15 data points
#'   start <- sample(1:dim(tmp)[1], 15)
#'   base <- tmp[start,]
#'   pool <- tmp[-start,]
#'   
#'   ## select 9 for addition
#'   newSamp <- maxDissim(
#'                        base, pool, 
#'                        n = 9, 
#'                        randomFrac = pct, obj = obj, ...)
#'   
#'   allSamp <- c(start, newSamp)
#'   
#'   plot(
#'        tmp[-newSamp,], 
#'        xlim = extendrange(tmp[,1]), ylim = extendrange(tmp[,2]), 
#'        col = "darkgrey", 
#'        xlab = "variable 1", ylab = "variable 2")
#'   points(base, pch = 16, cex = .7)
#'   
#'   for(i in seq(along = newSamp))
#'     points(
#'            pool[newSamp[i],1], 
#'            pool[newSamp[i],2], 
#'            pch = paste(i), col = "darkred") 
#' }
#' 
#' par(mfrow=c(2,2))
#' 
#' set.seed(414)
#' example(1, minDiss)
#' title("No Random Sampling, Min Score")
#' 
#' set.seed(414)
#' example(.1, minDiss)
#' title("10 Pct Random Sampling, Min Score")
#' 
#' set.seed(414)
#' example(1, sumDiss)
#' title("No Random Sampling, Sum Score")
#' 
#' set.seed(414)
#' example(.1, sumDiss)
#' title("10 Pct Random Sampling, Sum Score")
#' 
#' @export maxDissim
maxDissim <- function(a, b, n = 2, obj = minDiss, useNames = FALSE, randomFrac = 1, verbose = FALSE, alpha = 1, ...) 
{
  loadNamespace("proxy")
  if(nrow(b) < 2) stop("there must be at least 2 samples in b")
  if(ncol(a) != ncol(b)) stop("a and b must have the same number of columns")
  if(nrow(b) < n) stop("n must be less than nrow(b)")
  if(randomFrac > 1 | randomFrac <= 0) stop("randomFrac must be in (0, 1]")


  if(useNames)
    {
      if(is.null(rownames(b)))
        {
          warning("Cannot use rownames; swithcing to indices")
          free <- 1:nrow(b)
        } else free <- rownames(b)
    } else free <- 1:nrow(b)

  inSubset <- NULL
  newA <- a
  
  
  if(verbose) cat("  adding:")
  for(i in 1:n)
    {
      pool <- if(randomFrac == 1) free else sample(free, max(2, floor(randomFrac * length(free))))
      if(verbose)
        {
          cat("\nIter", i, "\n")
          cat("Number of candidates:", length(free), "\n")
          cat("Sampling from", length(pool), "samples\n")		
      }
      
      r <- b[pool,]
      
      diss <- proxy::dist(newA, b[pool, , drop = FALSE], ...)
     
      bNames <- colnames(b)[pool] 
      distanceAlltoSmallSet <- apply(diss, 2, obj)
      
      #take sub-group distance into account
      bigdatasetsize <- dim(r)[1]
      bigdatasetcolums <- dim(r)[2]
      
      subdisance <- NULL
      for (j in 1:bigdatasetsize) {
        #cat("j is: ", j, "\n")
        #cat("r[j] is:", r[j, ], "\n")
        subgroup <- NULL
        subgroup <- r[r[, 1] == r[j, 1] & r[, 2] == r[j, 2] & r[, 3] == r[j, 3] & r[, 3] == r[j, 3] & r[, 4] == r[j, 4] & r[, 5] == r[j, 5] & r[, 6] == r[j, 6] & r[, 7] == r[j, 7] & r[, 8] == r[j, 8] & r[, 9] == r[j, 9], 10:bigdatasetcolums]
        #subgroup <- data.frame(subgroup)
        #cat("Is subgroup vector: ", is.vector(subgroup))
        
        if (is.vector(subgroup))
        {
          subgroup <- t(data.frame(subgroup))
        }
        
        if (!is.null(subgroup)) {
          #cat ("r is ", dim(data.matrix(r[j, 10:bigdatasetcolums])), "\n")
          #cat ("subgroup is ", dim(subgroup), "\n")
          jdistance <- mean(proxy::dist(t(r[j, 10:bigdatasetcolums]), subgroup))
          subdisance <- c(subdisance, jdistance)
        }
        else
        {
           browser()
        }
       
       
      }
      
      #cat("i is ",  i, "\n")
      
      ## new cost
      cost <- distanceAlltoSmallSet - alpha*subdisance
      #cat("distanceAlltoSmallSet: ", distanceAlltoSmallSet, "\n")
      #cat("subdisance: ", subdisance, "\n")
      
      tmp <- pool[which.max(cost)]
      #browser()
      if(verbose)cat("new sample:", tmp, "\n")      
      inSubset <- c(inSubset, tmp)
      newA <- rbind(newA, b[tmp,, drop = FALSE])
      free <- free[!(free %in% inSubset)]
    }
  inSubset
}

#' @rdname maxDissim
#' @export
minDiss <- function(u) min(u, na.rm = TRUE)

#' @rdname maxDissim
#' @export
sumDiss <- function(u) sum(u, na.rm = TRUE)







splitter <- function(x, p = .8, start = NULL, ...)
  {
    n <- nrow(x)
    if(is.null(start)) start <- sample(1:n, 1)
    n2 <- n - length(start)
    m <- ceiling(p * n2)
    pool <- maxDissim(x[ start,,drop = FALSE],
                      x[-start,,drop = FALSE],
                      n = m,
                      ...)
    c(start, pool)
  }


splitByDissim <- function(x, p = .8, y = NULL, start = NULL, ...)
  {
    if(!is.data.frame(x)) x <- as.data.frame(x)
    
    if(!is.null(y))
      {
        if(!is.factor(y)) y <- as.factor(y)
        lvl <- levels(y)
        
        ind <- split(seq(along = y), y)
        ind2 <- lapply(ind, function(x) seq(along = x))
        start2 <- lapply(ind, function(x, start) which(x %in% start),
                         start = start)
        for(i in seq(along = lvl))
          {
            tmp <- splitter(x[ind[[i]],, drop = FALSE],
                            p = p,
                            start = start2[[i]],
                            ...)
            tmp2 <- ind[[i]][which(ind2[[i]] %in% tmp)]
            out <- if(i == 1) tmp2 else c(tmp2, out)
          }
      } else {
        out <- splitter(x, p = p, start = start, ...)
      }
    out
  }

