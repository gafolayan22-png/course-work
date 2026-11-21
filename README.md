# course-work
# install packages if not already installed:
# install.packages(c("DBI", "RSQLite"))
# install.packages("quantreg") once if not already installed
# install.packages(c("rsample", "mlr3measures")
install.packages("RSQLite") # install.packages("RSQLite")
install.packages(c("rsample", "mlr3measures"))
install.packages("quantreg")
library(DBI)
library(RSQLite)
library(rsample)
library(mlr3measures)
library(quantreg)

# 1. Connect to the SQLite database 
con <- dbConnect(
  RSQLite::SQLite(),
  dbname = "C:/Users/Admin/OneDrive/Documents/GIDEON FINA 1141/NEW_ECON_1205.db"
)
 
# 2. Verify table and colume name in the database
dbListTables(con)

# 3. Query and join the indicator table directly in SQL

query <- '
SELECT
  C.Country_Code,
  c.Country_Name,
  CAST(NULLIF(g."2025 [YR2025]","..") AS REAL) gdp,
  CAST(NULLIF(e."2025 [YR2025]","..") AS REAL) expenditure,
  CAST(NULLIF(p."2025 [YR2025]","..") AS REAL) population,
  CAST(NULLIF(u."2025 [YR2025]","..") AS REAL) unemployment,
  CAST(NULLIF(i."2025 [YR2025]","..") AS REAL) inflation
FROM countries c
INNER JOIN GDP g          ON c.country_code = g."Country Code"
INNER JOIN Expenditure e  ON c.country_code = e."Country Code"
INNER JOIN Inflation i    ON c.country_code = i."Country Code"
INNER JOIN Population p   ON c.country_code = p."Country Code"
INNER JOIN Unemployment u ON c.country_code = u."Country Code"
'

df_econ <- dbGetQuery(con, query)

# 4. Disconnect cleanly from the database
dbDisconnect(con)

# 5. Inspect the merged data and run OLS Linear regression
str(df_econ)

model <- lm(
  log_gdp ~ log_expenditure + log_population + unemployment + inflation,
  data = df_econ
)

summary(model)

# ==============================================================================
# Quantile Regression
# ==============================================================================

# 1. Estimate Quantile Regression across key percentiles (10th, 25th, 50th, 75th, 90th)
taus <- c(0.10, 0.25, 0.50, 0.75, 0.90)

model_qr <- rq(
  log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation,
  tau  = taus,
  data = df_econ
)

# 2. Print coefficient estimates and bootstrapped standard errors (robust to heteroscedasticity)
summary(model_qr, se = "boot", R = 500)

# 3. Compare Quantile Regression slopes directly against OLS
ols_baseline <- lm(
  log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation,
  data = df_econ
)

# Plot how each coefficient changes across quantiles vs. the flat OLS line
plot(summary(model_qr, se = "boot", R = 500))

# 4. Formal Hypothesis Test: Do slopes differ significantly across quantiles?
# Tests whether the 10th percentile (small economies) equals the 90th percentile (large economies)
qr_10 <- rq(log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation, tau = 0.10, data = df_econ)
qr_90 <- rq(log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation, tau = 0.90, data = df_econ)

anova(qr_10, qr_90)

# ==============================================================================
# LINEAR REGRESSION MONTE CARLO
# ==============================================================================
# set parameters
set.seed(42)          # For reproducibility
n           <- nrow(df_econ)     # Sample size per simulation
simulations <- 5000    # Number of simulated dataset

# Define known "True" theoretical parameters (NOT taken from model)

true_alpha      <- 0.5      # True intercept
true_beta_exp   <- 1.00     # True unit elasticity of GDP to Expenditure
true_beta_pop   <- 0.05     # True Population scaling effect 
true_beta_unemp <- -0.02    # True Okun-type unemployment penalty
true_beta_inf   <- -0.005   # True inflation drag

# set X (fixed empirical desgn matrix from database)
x_mat <- model.matrix(~ log(expenditure) + log(population) + unemployment + inflation, data = df_econ)
true_pars <- c(true_alpha, true_beta_exp,  true_beta_pop, true_beta_unemp, true_beta_inf)

# Initialize empty vectors to store results
sim_intercepts <- numeric(simulations)
sim_slopes     <- numeric(simulations)
coverage_count <- 0

# Run the Monte Carlo experiment with Heteroscedastic Errors
# Smaller economies (lower log(expenditure)) experience higher volatility than large once
het_sd <- 0.35 / (df_econ$log(expenditure) / mean(df_econ$log(expenditure)))^2

# run the simulation loop
for (i in 1:simulations) {
  
  # simulate random error term (normal distribution!)
  error <- rnorm(n, mean = 0, sd = het_sd)
  
  # generate dependent variable Y based on true model
  y <- as.vector(x_mat %*% true_pars) + error
  
  # fit a linear regression model
  model1 <- lm(
    log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation,
    data = df_econ
  )
  
  # Extract slope and classical OLS standard error for log_expenditure
  b_hat  <- coef(model)["log_expenditure"]
  se_hat <- summary(model)$coefficients["log_expenditure", "Std. Error"]
  
  sim_slopes_ols[i]   <- b_hat
  sim_se_classical[i] <- se_hat
  
  # Check if the 95% classical confidence interval captures the TRUE beta (1.00)
  ci_low  <- b_hat - qt(0.975, df = df.residual(model)) * se_hat
  ci_high <- b_hat + qt(0.975, df = df.residual(model)) * se_hat
  if (true_beta_exp >= ci_low && true_beta_exp <= ci_high) {
    coverage_count <- coverage_count + 1
  }
}

summary(model1)

# eyeball
head(sim_intercepts)
head(sim_slopes)

# plot the sampling distributions
par(mfrow = c(1, 2))

# histogram for intercept (alpha)
hist(sim_intercepts, breaks = 30, col = "lightblue", main = "Sampling Dist. of Intercept",
     xlab = expression(widehat(alpha)), border = "white")
abline(v = true_alpha, col = "red", lwd = 2, lty = 2)

# Histogram for Slope (beta)
hist(sim_slopes, breaks = 30, col = "lightgreen", main = "Sampling Dist. of Slope",
     xlab = expression(widehat(beta)), border = "white")
abline(v = true_beta, col = "red", lwd = 2, lty = 2)

# compare empirical and theoretical 5% critical values for beta

# calculate theoretical standard error of slope
theoretical_se <- 1 / sqrt(sum((x - mean(x))^2))

# get the t-scores for a two-tailed 5% test (df = n - 2)
t_critical <- qt(c(0.025, 0.975), df = n - 2)

# compute theoretical critical values
theoretical_critical_values <- true_beta + (t_critical * theoretical_se)

# print results to compare
cat("--- 5% Critical Values Comparison ---\n")
cat("Empirical (Simulated):  ", quantile(sim_slopes, probs = c(0.025, 0.975)), "\n")
cat("Theoretical (Formula):  ", theoretical_critical_values, "\n")

# ==============================================================================
# Linear Regression Bootstrapping
# ==============================================================================
# running a single instead of the linear regression model

set.seed(3)
error <- rnorm(n, mean = 0, sd = 1)
y <- true_alpha + true_beta*x + error
model2 <- lm(log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation,
            data = df_econ
)

B <- 1000
boot_intercepts <- numeric(B)
boot_slopes     <- numeric(B)

# create a dataframe of actual data
orig_data <- df_econ

# run the bootstrap
for (i in 1:B) {
  
  # sample rows with replacement
  boot_indices <- sample(1:n, size = n, replace = TRUE)
  boot_sample  <- orig_data[boot_indices, ]
  
  # fit model to the bootstrap sample
  boot_model <- lm(
    log(gdp) ~ log(expenditure) + log(population) + unemployment + inflation,
    data = boot_sample
  )
  
  # Save coefficients
  boot_intercepts[i] <- coef(boot_model)[1]
  boot_slopes[i]     <- coef(boot_model)[2]
}
summary(boot_model)
# Plot the bootstrap distributions 
par(mfrow = c(1,2))

# Histogram for intercept (alpha)
hist(boot_intercepts, breaks = 30, col = "lightblue", main = "Bootstrap Dist. of Intercept",
  xlab = expression(widehat(alpha)), border = "yellow")
abline(v = coef(model)[2], col = "purple", lwd = 2, lty = 2)

# Histogram for slope (beta: log_expenditure)
hist(boot_slopes, breaks = 30, col = "lightgreen", main = "Bootstrap Dist. of log(Expenditure) Slope",
     xlab = expression(widehat(beta)), border = "yellow")
abline(v = coef(model)[2], col = "purple", lwd = 2, lty = 2)

# Calculate and print bootstrap standard errors
cat("--- Case Bootstrap Standard Errors ---\n")  
cat("Bootstrap SE (Intercept):      ", sd(boot_intercepts), "\n")
cat("Bootstrap SE (log_expenditure):", sd(boot_slopes), "\n")


# ==============================================================================
# Linear Regression Holdout Method
# ==============================================================================

# Split dataset into a training and testing sample
set.seed(500)
split   <- initial_split(df_econ, prop = 0.75)
X_train <- training(split)
X_test  <- testing(split)

# Estimate model on training data
model_linreg_train <- lm(
  log(gdp) ~
    log(expenditure)
  + log(population)
  + unemployment
  + inflation,
  data = X_train
)

summary(model_linreg_train)

# Get predictions
y_train_pred <- predict(model_linreg_train, newdata = X_train)
y_test_pred  <- predict(model_linreg_train, newdata = X_test)

# Get RMSE (Full Model)
rmse_train <- rmse(X_train$log(gdp), y_train_pred)
rmse_test  <- rmse(X_test$log(gdp), y_test_pred)

tibble(
  prediction = c("Train", "Test"),
  rmse       = c(rmse_train, rmse_test)
)

# Compare to a different specification (parsimonious bivariate model)
model_linreg_train <- lm(log(gdp) ~ log(expenditure), data = X_train)

y_train_pred <- predict(model_linreg_train, newdata = X_train)
y_test_pred  <- predict(model_linreg_train, newdata = X_test)

rmse_train <- rmse(X_train$log(gdp), y_train_pred)
rmse_test  <- rmse(X_test$log(gdp), y_test_pred)

tibble(
  prediction = c("Train", "Test"),
  rmse       = c(rmse_train, rmse_test)
)


