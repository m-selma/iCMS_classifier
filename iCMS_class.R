# iCMS_classifier
# Date: 2023-10-20

# Loading necessary libraries


# Loading table of up/downregulated genes for iCMS2/3
markers <- read.delim("~/Desktop/CRC/Selma/supp_table.csv", sep = ',', header = T, stringsAsFactors = F)

# Loading counts matrix
cnts <- read.delim("~/Desktop/CRC/CRC_TPM_1063_counts.csv", sep = ',', header = T, stringsAsFactors = F, row.names = "X")

# Checking that we have 716 unique markers
marker_list <- unique(c(markers$iCMS2_Up, markers$iCMS2_Down, markers$iCMS3_Up, markers$iCMS3_Down))
marker_list_iCMS2 <- unique(c(markers$iCMS2_Up, markers$iCMS2_Down))
marker_list_iCMS3 <- unique(c(markers$iCMS3_Up, markers$iCMS3_Down))

# Splitting markers by iCMS type
iCMS2 <- unique(c(markers$iCMS2_Up, markers$iCMS2_Down))
iCMS2_df <- data.frame(markers = iCMS2, status = 'iCMS2')
iCMS3 <- unique(c(markers$iCMS3_Up, markers$iCMS3_Down))
iCMS3_df <- data.frame(markers = iCMS3, status = 'iCMS3')
i2_i3 <- rbind(iCMS2_df, iCMS3_df)
i2_i3$status <- factor(i2_i3$status)

# The pred_iCMS function
pred_iCMS <- function(cnts_mat, i2_i3, nPerm = 2000, nCores = 0, setSeed = FALSE) {
    
    # Step 1: Clean cnts_mat (handle missing values)
    keepP <- stats::complete.cases(cnts_mat)
    if (sum(!keepP) > 0) {
        cnts_mat <- cnts_mat[keepP, , drop = FALSE]
    }
    
    # Step 2: Clean i2_i3 markers
    keepT <- i2_i3$markers %in% rownames(cnts_mat)
    if (sum(!keepT) > 0) {
        i2_i3 <- i2_i3[keepT, ]
    }

    # Step 3: Prepare inputs
    N <- ncol(cnts_mat)
    K <- nlevels(i2_i3$status)
    S <- nrow(i2_i3)
    P <- nrow(cnts_mat)
    class.names <- levels(i2_i3$status)
    i2_i3$status <- as.numeric(i2_i3$status)
    
    # Step 4: Check for normalization warning
    cnts_mat.mean <- round(mean(cnts_mat), 2)
    if (abs(cnts_mat.mean) > 1) {
        isnorm <- " <- check feature centering!"
        cnts_mat.sd <- round(stats::sd(cnts_mat), 2)
        warning(paste0("emat mean=", cnts_mat.mean, "; sd=", cnts_mat.sd, isnorm), call. = FALSE)
    }

    # Step 5: Matching markers and cnts_mat
    feat.class <- paste(range(table(i2_i3$status)), collapse = "-")
    mm <- match(i2_i3$markers, rownames(cnts_mat), nomatch = 0)
    if (!all(rownames(cnts_mat)[mm] == i2_i3$markers)) {
        stop("error matching probes, check rownames(cnts_mat) and i2_i3$markers")
    }
    pReplace <- length(i2_i3$markers) > length(unique(i2_i3$markers))

    # Step 6: Prepare templates
    tmat <- matrix(rep(i2_i3$status, K), ncol = K)
    for (k in seq_len(K)) tmat[, k] <- as.numeric(tmat[, k] == k)
    if (K == 2) tmat[tmat == 0] <- -1
    
    # Step 7: Define similarity and distance functions
    sim_fn <- function(x, y) corCosine(x, y)
    simToDist <- function(cos.sim) sqrt(1/2 * (1 - cos.sim))
    
    # Step 8: Define ntp_fn (nearest template prediction function)
    ntp_fn <- function(n) {
        n.sim <- as.vector(sim_fn(cnts_mat[mm, n, drop = FALSE], tmat))
        n.sim.perm.max <- apply(sim_fn(matrix(cnts_mat[, n][sample.int(P, S * nPerm, replace = TRUE)], ncol = nPerm), tmat), 1, max)
        n.ntp <- which.max(n.sim)
        n.sim.ranks <- rank(-c(n.sim[n.ntp], (n.sim.perm.max)))
        n.pval <- n.sim.ranks[1] / length(n.sim.ranks)
        return(c(n.ntp, simToDist(n.sim), n.pval))
    }

    # Step 9: Parallel or serial processing based on nCores and setSeed
    if (setSeed) {
        # Serialized prediction with seed
        set.seed(7)
        nCores <- 1
        res <- lapply(seq_len(N), ntp_fn)
        res <- data.frame(do.call(rbind, res))
    } else {
        # Parallelized prediction without seed
        nCores <- ifelse(nCores == 0, parallel::detectCores(), nCores)
        options(mc.cores = nCores)
        nParts <- split(seq_len(N), cut(seq_len(N), nCores, labels = FALSE))
        res <- parallel::mclapply(nParts, function(n) vapply(n, ntp_fn, numeric(2 + K)))
        res <- data.frame(t(do.call(cbind, res)))
    }

    # Step 10: Prepare output
    colnames(res) <- c("prediction", paste0("d.", class.names), "p.value")
    res$prediction <- factor(class.names[res$prediction], levels = class.names)
    rownames(res) <- colnames(cnts_mat)
    res$p.value[res$p.value < 1 / nPerm] <- 1 / nPerm
    res$FDR <- stats::p.adjust(res$p.value, "fdr")

    # Return result
    return(res)
}

# result <- pred_iCMS(cnts, i2_i3, nPerm = 2000, nCores = 0, setSeed = FALSE)
