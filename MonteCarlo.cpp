#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numbers>
#include <algorithm>
#include <iterator>
#include <numeric>

// Hardcoded direction numbers for the first dimension
std::vector<unsigned int> directionNumbers = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
};

double inverseCumulativeNormal(double p) {
    // Coefficients for the Beasley-Springer-Moro algorithm
    static const double a[4] = {2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637};
    static const double b[4] = {-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};
    static const double c[9] = {0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                                0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
                                0.0000321767881768, 0.0000002888167364, 0.0000003960315187};

    if (p <= 0 || p >= 1) {
        std::cerr << "Argument to inverseCumulativeNormal out of bounds" << std::endl;
        return 0; // Or handle error appropriately
    }

    // See if we're in the tails
    if (p < 0.02425) {
        // Left tail
        double q = sqrt(-2 * log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) * q + c[7]) * q + c[8];
    } else if (p > 1 - 0.02425) {
        // Right tail
        double q = sqrt(-2 * log(1 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) * q + c[7]) * q + c[8];
    } else {
        // Central region
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * q + b[0]) * r + b[1]) * r + b[2]) * r + b[3];
    }
}


class SobolSequenceGenerator {
private:
    unsigned int seed;
    unsigned int count;

public:
    SobolSequenceGenerator() : seed(0), count(0) {}

    double next() {
        unsigned int c = 1;
        unsigned int value = count++;
        while (value & 1) {
            value >>= 1;
            c++;
        }

        if (c >= directionNumbers.size()) {
            // Handle error: c exceeds the size of precomputed direction numbers
            std::cerr << "Index out of bounds in direction numbers array." << std::endl;
            return 0.0;
        }

        seed ^= directionNumbers[c - 1];
        return static_cast<double>(seed) / static_cast<double>(1 << 16);  // Normalize to [0, 1]
    }
};


double boxMullerTransform(double u1, double u2) {
    return sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

std::vector<double> simulateAssetPathsSobol(int numPaths, double spotPrice, double riskFreeRate, double volatility, double maturity) {
    std::vector<double> assetPaths(numPaths);
    SobolSequenceGenerator sobol;
    double drift = (riskFreeRate - 0.5 * volatility * volatility) * maturity;
    double diffusionCoefficient = volatility * sqrt(maturity);

    for (int i = 0; i < numPaths; ++i) {
        double u1 = sobol.next() + 1e-10;
        double u2 = sobol.next() + 1e-10;
        double z = boxMullerTransform(u1, u2);
        double diffusion = diffusionCoefficient * z;
        assetPaths[i] = spotPrice * exp(drift + diffusion);
    }

    return assetPaths;
}


double normalCDF(double value) {
    return 0.5 * erfc(-value * M_SQRT1_2);
}

double normalPDF(double value) {
    return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * value * value);
}

void blackScholesCall(double S, double K, double r, double sigma, double T,
                      double &callPrice, double &delta, double &gamma) {
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    callPrice = S * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
    delta = normalCDF(d1);
    gamma = normalPDF(d1) / (S * sigma * sqrt(T));
}



// Fonction pour simuler les chemins de prix de l'actif
std::vector<double> simulateAssetPaths(int numPaths, double spotPrice, double riskFreeRate, double volatility, double maturity) {
    std::vector<double> assetPaths(numPaths);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    double drift = (riskFreeRate - 0.5 * volatility * volatility) * maturity;
    double diffusionCoefficient = volatility * sqrt(maturity);

    for (int i = 0; i < numPaths; ++i) {
        double diffusion = diffusionCoefficient * d(gen);
        assetPaths[i] = spotPrice * exp(drift + diffusion);
    }

    return assetPaths;
}

// Function to generate a sequence of random numbers in the range [0, 1]
std::vector<double> random_generate(int num_paths) {
    std::vector<double> sequence;
    std::mt19937 rng(std::random_device{}()); // random-number engine
    std::uniform_real_distribution<double> dist(0.0, 1.0); // distribution

    for (int i = 0; i < num_paths; ++i) {
        sequence.push_back(dist(rng));
    }

    return sequence;
}


// Fonction pour calculer le prix de l'option
double calculateOptionPrice(const std::vector<double>& assetPaths, double strikePrice, double riskFreeRate, double maturity) {
    double payoffSum = 0;
    for (double price : assetPaths) {
        payoffSum += std::max(price - strikePrice, 0.0);
    }

    double averagePayoff = payoffSum / assetPaths.size();
    return averagePayoff * exp(-riskFreeRate * maturity);
}

// Fonction pour calculer Gamma
double calculateGamma(int numPaths, double spotPrice, double strikePrice, double riskFreeRate, double volatility, double maturity, double gammaShift) {
    auto originalPaths = simulateAssetPathsSobol(numPaths, spotPrice, riskFreeRate, volatility, maturity);
    auto upPaths = simulateAssetPathsSobol(numPaths, spotPrice * (1 + gammaShift), riskFreeRate, volatility, maturity);
    auto downPaths = simulateAssetPathsSobol(numPaths, spotPrice * (1 - gammaShift), riskFreeRate, volatility, maturity);

    double originalOptionPrice = calculateOptionPrice(originalPaths, strikePrice, riskFreeRate, maturity);
    double upOptionPrice = calculateOptionPrice(upPaths, strikePrice, riskFreeRate, maturity);
    double downOptionPrice = calculateOptionPrice(downPaths, strikePrice, riskFreeRate, maturity);

    return (upOptionPrice - 2 * originalOptionPrice + downOptionPrice) / (spotPrice * spotPrice * gammaShift * gammaShift);
}

// Function to calculate Delta
double calculateDelta(int numPaths, double spotPrice, double strikePrice, double riskFreeRate, double volatility, double maturity, double deltaShift) {
    auto originalPaths = simulateAssetPathsSobol(numPaths, spotPrice, riskFreeRate, volatility, maturity);
    auto shiftedPaths = simulateAssetPathsSobol(numPaths, spotPrice * (1 + deltaShift), riskFreeRate, volatility, maturity);

    double originalOptionPrice = calculateOptionPrice(originalPaths, strikePrice, riskFreeRate, maturity);
    double shiftedOptionPrice = calculateOptionPrice(shiftedPaths, strikePrice, riskFreeRate, maturity);

    return (shiftedOptionPrice - originalOptionPrice) / (spotPrice * deltaShift);
}

double calculateQuasiMonteCarloDelta(double spotPrice, double strikePrice, double riskFreeRate, double volatility, double maturity, int numPaths, double deltaShift) {
    SobolSequenceGenerator sobol;
    double drift = (riskFreeRate - 0.5 * volatility * volatility) * maturity;
    double diffusionCoefficient = volatility * sqrt(maturity);

    double originalValueSum = 0.0;
    double perturbedValueSum = 0.0;

    for (int i = 0; i < numPaths; ++i) {
        double sobolValue = sobol.next();
        double normalSample = inverseCumulativeNormal(sobolValue);

        double diffusion = diffusionCoefficient * normalSample;
        double originalPath = spotPrice * exp(drift + diffusion);
        double perturbedPath = (spotPrice * (1 + deltaShift)) * exp(drift + diffusion);

        originalValueSum += std::max(originalPath - strikePrice, 0.0);
        perturbedValueSum += std::max(perturbedPath - strikePrice, 0.0);
    }

    double originalValue = (originalValueSum / numPaths) * exp(-riskFreeRate * maturity);
    double perturbedValue = (perturbedValueSum / numPaths) * exp(-riskFreeRate * maturity);

    double delta = (perturbedValue - originalValue) / (spotPrice * deltaShift);
    return delta;
}


int main() {
    // Paramètres
    int numPaths = 10000;
    double spotPrice = 100;
    double strikePrice = 100;
    double riskFreeRate = 0.05;
    double volatility = 0.2;
    double maturity = 1;
    double gammaShift = 0.01; 
    double deltaShift = 0.0005;

    // Calcul du Gamma
    double gamma = calculateGamma(numPaths, spotPrice, strikePrice, riskFreeRate, volatility, maturity, gammaShift);
    double delta = calculateDelta(numPaths, spotPrice, strikePrice, riskFreeRate, volatility, maturity, deltaShift);
    double delta_qmc = calculateQuasiMonteCarloDelta(spotPrice,strikePrice,riskFreeRate,volatility,maturity,numPaths,deltaShift);

    // Affichage des résultats
    std::cout << "Gamma: " << gamma << std::endl;
    std::cout << "Delta: " << delta << std::endl;
    std::cout << "Delta QMC: " << delta_qmc << std::endl;
    

    double callPrice, deltaBS, gammaBS;
    blackScholesCall(spotPrice, strikePrice, riskFreeRate, volatility, maturity, callPrice, deltaBS, gammaBS);
    std::cout << "Call Option Price Black Scholes: " << callPrice << std::endl;
    std::cout << "Delta Black Scholes: " << deltaBS << std::endl;
    std::cout << "Gamma Black Scholes: " << gammaBS << std::endl;

    return 0;
}
