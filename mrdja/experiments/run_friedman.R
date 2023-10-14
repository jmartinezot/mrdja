library(scmamp)

data <- read.csv("temp_data.csv", header = TRUE, row.names = NULL)
htest <- friedmanTest(data)
saveRDS(htest, file = "temp_htest.rds")
raw.pvalues <- friedmanPost(data)
saveRDS(raw.pvalues, file = "temp_raw_pvalues.rds")
adjusted.pvalues <- adjustShaffer(raw.pvalues)
saveRDS(adjusted.pvalues, file = "temp_adjusted_pvalues.rds")

#write.csv(adjusted.pvalues, "temp_result.csv")
