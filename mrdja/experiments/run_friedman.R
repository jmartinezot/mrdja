library(scmamp)

data <- read.csv("temp_data.csv", header = TRUE)
# remove the first column
data <- data[, -1]
htest <- friedmanTest(data)
raw.pvalues <- friedmanPost(data)
adjusted.pvalues <- adjustShaffer(raw.pvalues)

write.csv(adjusted.pvalues, "temp_result.csv")
