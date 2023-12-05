import numpy as np
from scipy.stats import norm
# pip install sobol_seq (si le package est pas dans l'env)
from sobol_seq import i4_sobol_generate

def quasi_monte_carlo_option_pricing(spot_price, strike_price, risk_free_rate, volatility, maturity, num_paths):
    
    # Generate Sobol low-discrepancy sequences
    sobol_sequences = i4_sobol_generate(dim_num=1, n=num_paths)

    # Adjust sequences to normal distribution
    normal_samples = norm.ppf(sobol_sequences)

    # Simulate asset price paths
    drift = (risk_free_rate - 0.5 * volatility ** 2) * maturity
    diffusion = volatility * np.sqrt(maturity) * normal_samples
    asset_paths = spot_price * np.exp(drift + diffusion)

    # Calculate option payoffs
    payoffs = np.maximum(asset_paths - strike_price, 0)

    # Discount payoffs to present value and average
    option_price = np.mean(payoffs) * np.exp(-risk_free_rate * maturity)
    return option_price


# Example parameters
spot_price = 100  #S
strike_price = 100  #K
risk_free_rate = 0.05 #R 
volatility = 0.2  #sigma
maturity = 1  #T
num_paths = 50000

d1 = (np.log(spot_price/strike_price) + ((risk_free_rate + (volatility**2)/2)*maturity))/(volatility*np.sqrt(maturity))
d2 = (np.log(spot_price/strike_price) + ((risk_free_rate - (volatility**2)/2)*maturity))/(volatility*np.sqrt(maturity))
C = spot_price*norm.cdf(d1)-strike_price*np.exp(-risk_free_rate * maturity)*norm.cdf(d2)
print("option price", C)

# Calculate option price
option_price = quasi_monte_carlo_option_pricing(spot_price, strike_price, risk_free_rate, volatility, maturity, num_paths)
print("Estimated European Call Option Price:", option_price)



def quasi_monte_carlo_delta(spot_price, strike_price, risk_free_rate, volatility, maturity, num_paths, delta_shift):
    sobol_sequences = i4_sobol_generate(dim_num=1, n=num_paths)
    normal_samples = norm.ppf(sobol_sequences)

    # Original asset paths
    drift = (risk_free_rate - 0.5 * volatility ** 2) * maturity
    diffusion = volatility * np.sqrt(maturity) * normal_samples
    original_paths = spot_price * np.exp(drift + diffusion)

    # Perturbed asset paths
    perturbed_paths = (spot_price * (1 + delta_shift)) * np.exp(drift + diffusion)

    # Calculate original and perturbed payoffs
    original_payoffs = np.maximum(original_paths - strike_price, 0)
    perturbed_payoffs = np.maximum(perturbed_paths - strike_price, 0)

    # Discounted payoffs
    original_value = np.mean(original_payoffs) * np.exp(-risk_free_rate * maturity)
    perturbed_value = np.mean(perturbed_payoffs) * np.exp(-risk_free_rate * maturity)

    # Estimate of Delta
    delta = (perturbed_value - original_value) / (spot_price * delta_shift)
    return original_value, delta

# Example parameters
delta_shift = 0.01  # 1% shift for Delta calculation

# Calculate option price and Delta
option_price, delta = quasi_monte_carlo_delta(spot_price, strike_price, risk_free_rate, volatility, maturity, num_paths, delta_shift)
print("Option Price:", option_price, "Delta:", delta)
d1 = (np.log(spot_price/strike_price) + ((risk_free_rate + (volatility**2)/2)*maturity))/(volatility*np.sqrt(maturity))


print("Expected delta : ", norm.cdf(d1))




def quasi_monte_carlo_gamma(spot_price, strike_price, risk_free_rate, volatility, maturity, num_paths, gamma_shift):
    sobol_sequences = i4_sobol_generate(dim_num=1, n=num_paths)
    normal_samples = norm.ppf(sobol_sequences)

    # Chemins d'actifs originaux, augmentés et diminués
    drift = (risk_free_rate - 0.5 * volatility ** 2) * maturity
    diffusion = volatility * np.sqrt(maturity) * normal_samples
    original_paths = spot_price * np.exp(drift + diffusion)
    up_paths = (spot_price * (1 + gamma_shift)) * np.exp(drift + diffusion)
    down_paths = (spot_price * (1 - gamma_shift)) * np.exp(drift + diffusion)

    # Calcul des payoffs
    original_payoffs = np.maximum(original_paths - strike_price, 0)
    up_payoffs = np.maximum(up_paths - strike_price, 0)
    down_payoffs = np.maximum(down_paths - strike_price, 0)

    # Valeurs actualisées
    original_value = np.mean(original_payoffs) * np.exp(-risk_free_rate * maturity)
    up_value = np.mean(up_payoffs) * np.exp(-risk_free_rate * maturity)
    down_value = np.mean(down_payoffs) * np.exp(-risk_free_rate * maturity)

    # Estimation de Gamma
    gamma = (up_value - 2 * original_value + down_value) / (spot_price ** 2 * gamma_shift ** 2)
    return original_value, gamma

# Paramètres exemple
gamma_shift = 0.01

# Calcul du prix de l'option et du Gamma
option_price, gamma = quasi_monte_carlo_gamma(spot_price, strike_price, risk_free_rate, volatility, maturity, num_paths, gamma_shift)
print("Option price:", option_price, "Gamma :", gamma)
g = (np.exp((-d1*2)/2)/np.sqrt(2*np.pi))/(spot_price*volatility*np.sqrt(maturity))
print("expected gamma", g)
